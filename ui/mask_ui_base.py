"""
# mask_ui_base.py - v1.1747845070
# Updated: Wednesday, May 21, 2025 at 09:29:52 PST
# Changes in this version:
# - Applied "move forward then backward" workaround to fix coordinates
# - Ensured initial view is properly fit to window size
# - Added explicit reset_view at initialization
# - Implemented frame navigation sequence to fix coordinate systems
# - Used the exact user-confirmed approach that works correctly

Base UI components for the SAM-based mask generator.
Handles initialization and window setup.
"""

import os
import tkinter as tk
from tkinter import ttk, StringVar, IntVar
from PIL import Image, ImageTk
import platform
import cv2
import json

# Import UI components
from ui.ui_components import Tooltip

# Import from other mask modules
from ui.mask_ui_interactions import MaskUIInteractions
from ui.mask_ui_frame_manager import MaskFrameManager
from ui.mask_ui_editor import MaskEditor


class MaskUIBase:
    """Base class for mask generator UI with initialization functionality"""
    
    def __init__(self, root, on_mask_generated=None):
        """
        Initialize the mask generator UI
        
        Args:
            root: Tkinter root or Toplevel window
            on_mask_generated: Callback function for when mask is generated
        """
        self.root = root
        self.on_mask_generated = on_mask_generated
        
        # Initialize instance variables
        self.video_path = None
        self.mask_window = None
        self.canvas = None
        self.tk_image = None
        self.overlay_image = None
        self.image = None
        self.display_frame = None
        self.scale = 1.0
        self.status_var = None
        
        # Initialize UI state variables
        self.points = []
        self.point_markers = []
        self.box_coords = []
        self.box_markers = []
        self.overlay_visible = False
        self.generated_mask = None
        self.edited_mask = None
        self.click_radius = 10
        self.paint_radius = 5
        self.min_brush_size = 1
        self.max_brush_size = 50
        self.is_painting = False
        self.drawing_box = False
        self.box_start_x = None
        self.box_start_y = None
        self.temp_box = None
        self.paint_cursor = None
        self.last_overlay_update = 0
        self.overlay_update_debounce = 50
        self.update_pending = False
        self.accumulated_paint_actions = []
        
        # View state management
        self.fit_to_window = True  # Track if we're in "fit to window" mode
        self.fit_scale = 1.0       # Scale used for fit to window
        self.last_canvas_width = 0 # Last known canvas width for fit calculations
        self.last_canvas_height = 0 # Last known canvas height for fit calculations
        
        # Zoom and pan variables
        self.zoom_factor = 1.0
        self.zoom_min = 0.1
        self.zoom_max = 10.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self.is_panning = False
        self.pan_mode_active = False
        
        # Undo/redo stack
        self.command_stack = []
        self.redo_stack = []
        self.max_undo_steps = 20
        
        # Frame navigation variables
        self.current_frame_index = 0
        self.total_frames = 0
        self.checkpoints = {}
        self.video_info = None
        self.updating_frame = False
        
        # Settings file path
        self.settings_file = os.path.join(os.path.expanduser("~"), ".matanyone_mask_settings.json")
        
        # Create helper classes for functionality
        self.interactions = MaskUIInteractions(self)
        self.frame_manager = MaskFrameManager(self)
        self.editor = MaskEditor(self)
        
        # Load saved settings
        self.load_settings()
        
    def initialize_ui(self, video_path, mask_save_path=None):
        """
        Initialize the UI components
        
        Args:
            video_path: Path to the video file
            mask_save_path: Path to save the generated mask (optional)
        """
        self.video_path = video_path
        
        # Create and setup mask window
        self.mask_window = tk.Toplevel(self.root)
        self.mask_window.title("Generate Mask")
        
        # Calculate appropriate window size (max 80% of screen)
        screen_width = self.mask_window.winfo_screenwidth()
        screen_height = self.mask_window.winfo_screenheight()
        
        # Get original image dimensions
        img_height, img_width = self.display_frame.shape[:2]
        
        # Calculate scale factor to fit within screen
        max_width = int(screen_width * 0.8)
        max_height = int(screen_height * 0.8) - 200  # Leave more space for controls
        
        self.scale = min(max_width / img_width, max_height / img_width, 1.0)
        
        # If image is too large, resize it for display
        if self.scale < 1:
            display_width = int(img_width * self.scale)
            display_height = int(img_height * self.scale)
        else:
            display_width = img_width
            display_height = img_height
        
        # Create top-level container
        main_frame = ttk.Frame(self.mask_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a canvas container with scrollbars
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Add scrollbars
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create a canvas for the image
        self.canvas = tk.Canvas(canvas_frame, width=display_width, height=display_height,
                              xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure scrollbars
        h_scrollbar.config(command=self.canvas.xview)
        v_scrollbar.config(command=self.canvas.yview)
        
        # Convert numpy image to PhotoImage for display
        pil_image = Image.fromarray(self.display_frame)
        self.tk_image = ImageTk.PhotoImage(image=pil_image)
        
        # Create a container frame for the image
        self.image_container = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image, tags="base_image")
        
        # Configure canvas scrollable area
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        
        # Set up variables for interactions
        self.interaction_mode = tk.StringVar(value="point")
        
        # Determine masks directory path
        video_dir = os.path.dirname(video_path)
        self.masks_dir = os.path.join(video_dir, "masks")
        os.makedirs(self.masks_dir, exist_ok=True)
        
        # Create a status label
        self.status_var = tk.StringVar(value="Click to select foreground points (green)")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.pack(pady=5)
        
        # Create option frame for mode selection
        self.create_mode_selection_frame(main_frame)
        
        # Create paint settings frame
        self.create_paint_settings_frame(main_frame)
        
        # Create button frames
        self.create_button_frames(main_frame, mask_save_path)
        
        # Create frame slider and navigation controls
        self.create_navigation_controls(main_frame)
        
        # Create confirm/cancel buttons
        self.create_confirm_buttons(main_frame, video_path, mask_save_path)
        
        # Bind mouse events
        self.bind_mouse_events()
        
        # Set up keyboard shortcuts
        self.mask_window.bind("<Key>", self.interactions.on_key_press)
        self.mask_window.bind("<Control-z>", self.editor.undo)
        self.mask_window.bind("<Control-y>", self.editor.redo)
        
        # Hide cursor in paint mode (will show it when mode changes)
        self.empty_cursor = "none"
        self.default_cursor = self.canvas['cursor']
        
        # Update UI to show checkpoints
        self.frame_manager.update_checkpoint_markers()
        
        # Add info label
        self.add_info_label(main_frame)
        
        # Update brush size from settings
        self.interactions.update_brush_size()
        
        # Force the canvas to update so we have correct dimensions
        self.canvas.update_idletasks()
        
        # IMPORTANT: Make sure the view is initially fit to window
        # This guarantees the image is properly sized when first loaded
        self.interactions.reset_view()
        
        # Mark coordinate systems as initialized right away
        self.coordinate_systems_initialized = True
        
        # Schedule forward/backward navigation after a longer delay 
        # to ensure the initial view is properly rendered
        self.canvas.after(1000, self._simulate_frame_change)
    
    def initialize_view(self):
        """
        Initialize the view and coordinate systems
        This is critical for proper coordinate transformation when the mask generator first opens
        """
        # First reset the view to fit the image
        self.interactions.reset_view()
        
        # Force an immediate update of the canvas to ensure coordinate systems are synchronized
        self.canvas.update_idletasks()
        
        # Force another reset and update to ensure everything is synchronized
        # This is needed because the first reset might happen before the canvas is fully rendered
        self.canvas.after(100, self._complete_initialization)
    
    def _simulate_frame_change(self):
        """
        Fix coordinate systems by actually navigating forward and backward
        This ensures the exact same code path that works during normal frame changes
        """
        # Make sure the view fits to window size first
        self.interactions.reset_view()
        
        # Let the window know we're in the middle of processing
        if not hasattr(self, 'updating_frame'):
            self.updating_frame = False
        
        # Move forward then backward to fix coordinate systems
        # This is what the user confirmed works 100% of the time
        print("Moving forward and backward to fix coordinate systems...")
        
        # Schedule the moves with a small delay between them
        # to ensure each completes before the next one starts
        def forward_then_back():
            # Navigate forward one frame
            if hasattr(self, 'next_frame_button'):
                print("Simulating next frame button click...")
                self.frame_manager.navigate_frame(1)
                
                # Then schedule moving back after a short delay
                self.canvas.after(500, lambda: go_back_to_first())
        
        def go_back_to_first():
            # Navigate back to first frame
            if hasattr(self, 'prev_frame_button'):
                print("Simulating previous frame button click...")
                self.frame_manager.navigate_frame(-1)
                print("Navigation completed - coordinate systems should now be fixed")
        
        # Start the sequence after a short delay to ensure UI is ready
        self.canvas.after(500, forward_then_back)
    
    def create_mode_selection_frame(self, parent):
        """Create the mode selection radio buttons"""
        option_frame = ttk.Frame(parent)
        option_frame.pack(pady=5)
        
        # Radio buttons for interaction mode
        ttk.Radiobutton(option_frame, text="Point Selection", 
                       variable=self.interaction_mode, value="point", 
                       command=self.interactions.switch_mode).pack(side=tk.LEFT, padx=10)
        
        ttk.Radiobutton(option_frame, text="Box Selection", 
                       variable=self.interaction_mode, value="box", 
                       command=self.interactions.switch_mode).pack(side=tk.LEFT, padx=10)
        
        ttk.Radiobutton(option_frame, text="Paint Mode", 
                       variable=self.interaction_mode, value="paint", 
                       command=self.interactions.switch_mode).pack(side=tk.LEFT, padx=10)
        
        ttk.Radiobutton(option_frame, text="Move View", 
                       variable=self.interaction_mode, value="move", 
                       command=self.interactions.switch_mode).pack(side=tk.LEFT, padx=10)
    
    def create_paint_settings_frame(self, parent):
        """Create the paint brush settings frame"""
        self.paint_frame = ttk.LabelFrame(parent, text="Paint Brush Settings")
        self.paint_frame.pack(pady=5, fill=tk.X, padx=20)
        
        # Paint info
        paint_info_frame = ttk.Frame(self.paint_frame)
        paint_info_frame.pack(side=tk.LEFT, padx=20, pady=5)
        
        ttk.Label(paint_info_frame, text="Left-click: Add to mask  |  Right-click: Remove from mask").pack(side=tk.LEFT, padx=5)
        
        # Brush size
        brush_size_frame = ttk.Frame(self.paint_frame)
        brush_size_frame.pack(side=tk.LEFT, padx=20, pady=5)
        
        ttk.Label(brush_size_frame, text="Brush Size:").pack(side=tk.LEFT, padx=5)
        
        # Create a list of brush sizes
        brush_sizes = [1, 3, 5, 10, 15, 20, 30, 50]
        self.brush_size_combo = ttk.Combobox(brush_size_frame, values=brush_sizes, width=5)
        self.brush_size_combo.current(2)  # Default to 5px
        self.brush_size_combo.pack(side=tk.LEFT, padx=5)
        self.brush_size_combo.bind("<<ComboboxSelected>>", self.interactions.update_brush_size)
        
        # Keyboard shortcuts help
        ttk.Label(brush_size_frame, text="(Use [ and ] keys to adjust size)").pack(side=tk.LEFT, padx=5)
        
        # Undo/redo buttons
        undo_frame = ttk.Frame(self.paint_frame)
        undo_frame.pack(side=tk.RIGHT, padx=20, pady=5)
        
        undo_button = ttk.Button(undo_frame, text="Undo", command=self.editor.undo)
        undo_button.pack(side=tk.LEFT, padx=5)
        Tooltip(undo_button, "Undo last action (Ctrl+Z)")
        
        redo_button = ttk.Button(undo_frame, text="Redo", command=self.editor.redo)
        redo_button.pack(side=tk.LEFT, padx=5)
        Tooltip(redo_button, "Redo last undone action (Ctrl+Y)")
        
        # Initially hide paint settings
        self.paint_frame.pack_forget()
    
    def create_button_frames(self, parent, mask_save_path):
        """Create the button frames for controls"""
        # Buttons frame
        button_frame = ttk.Frame(parent)
        button_frame.pack(pady=10)
        
        # First row of buttons
        self.generate_button = ttk.Button(button_frame, text="Generate Mask", 
                                       command=self.editor.generate_and_preview_mask)
        self.generate_button.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(button_frame, text="Clear", 
                 command=self.editor.clear_selections).pack(side=tk.LEFT, padx=10)
        
        # Add toggle overlay button
        self.overlay_button = ttk.Button(button_frame, text="Toggle Overlay", 
                                      command=self.editor.toggle_overlay, state=tk.DISABLED)
        self.overlay_button.pack(side=tk.LEFT, padx=10)
        
        # Add zoom controls
        zoom_frame = ttk.Frame(button_frame)
        zoom_frame.pack(side=tk.LEFT, padx=(20, 5))
        
        ttk.Label(zoom_frame, text="| Zoom:").pack(side=tk.LEFT)
        
        zoom_in_button = ttk.Button(zoom_frame, text="+", width=3, 
                                  command=lambda: self.interactions.zoom(1.25))
        zoom_in_button.pack(side=tk.LEFT, padx=5)
        
        zoom_out_button = ttk.Button(zoom_frame, text="-", width=3, 
                                   command=lambda: self.interactions.zoom(0.8))
        zoom_out_button.pack(side=tk.LEFT, padx=5)
        
        reset_view_button = ttk.Button(zoom_frame, text="Reset View", 
                                     command=self.interactions.reset_view)
        reset_view_button.pack(side=tk.LEFT, padx=5)
        
    def create_navigation_controls(self, parent):
        """Create frame navigation controls"""
        # Create frame slider frame
        slider_frame = ttk.Frame(parent)
        slider_frame.pack(fill=tk.X, pady=10)
        
        # Add frame navigation controls
        navigation_frame = ttk.Frame(slider_frame)
        navigation_frame.pack(fill=tk.X)
        
        # Previous frame button
        self.prev_frame_button = ttk.Button(navigation_frame, text="←", width=3,
                                         command=lambda: self.frame_manager.navigate_frame(-1))
        self.prev_frame_button.pack(side=tk.LEFT, padx=5)
        
        # Frame slider
        self.frame_slider = tk.Scale(navigation_frame, from_=0, to=self.total_frames-1, 
                                   orient=tk.HORIZONTAL, showvalue=False, 
                                   command=self.frame_manager.on_slider_changed)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        # Next frame button
        self.next_frame_button = ttk.Button(navigation_frame, text="→", width=3,
                                         command=lambda: self.frame_manager.navigate_frame(1))
        self.next_frame_button.pack(side=tk.LEFT, padx=5)
        
        # Frame counter and input
        frame_counter_frame = ttk.Frame(navigation_frame)
        frame_counter_frame.pack(side=tk.LEFT, padx=15)
        
        ttk.Label(frame_counter_frame, text="Frame:").pack(side=tk.LEFT)
        
        # Frame number entry
        self.current_frame_var = StringVar(value="0")
        self.frame_entry = ttk.Entry(frame_counter_frame, textvariable=self.current_frame_var, width=6)
        self.frame_entry.pack(side=tk.LEFT, padx=5)
        self.frame_entry.bind("<Return>", self.frame_manager.on_frame_entry)
        self.frame_entry.bind("<FocusOut>", self.frame_manager.on_frame_entry)
        
        # Total frames label
        ttk.Label(frame_counter_frame, text=f"/ {self.total_frames-1}").pack(side=tk.LEFT)
        
        # Create a container for checkpoint markers
        self.checkpoint_markers_frame = ttk.Frame(slider_frame)
        self.checkpoint_markers_frame.pack(fill=tk.X, pady=5)
    
    def create_confirm_buttons(self, parent, video_path, mask_save_path):
        """Create confirm and cancel buttons"""
        # Second row of buttons for confirm/cancel
        confirm_frame = ttk.Frame(parent)
        confirm_frame.pack(pady=5)
        
        # Confirm button (initially disabled)
        self.confirm_button = ttk.Button(confirm_frame, text="Confirm & Save", 
                                      command=lambda: self.frame_manager.save_and_close(video_path, mask_save_path),
                                      state=tk.DISABLED)
        self.confirm_button.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(confirm_frame, text="Cancel", 
                 command=self.mask_window.destroy).pack(side=tk.LEFT, padx=10)
    
    def bind_mouse_events(self):
        """Bind mouse events to canvas"""
        # Ensure these work for both Windows and Mac
        self.canvas.bind("<Button-1>", self.interactions.on_canvas_click)
        self.canvas.bind("<ButtonRelease-1>", self.interactions.on_button_release)
        self.canvas.bind("<Motion>", self.interactions.on_mouse_move)
        
        # For right-click, we need different bindings depending on the platform
        if platform.system() == "Darwin":  # macOS
            # On macOS, right-click is typically Control+Click
            self.canvas.bind("<Control-Button-1>", self.interactions.on_canvas_right_click)
            self.canvas.bind("<Control-B1-Motion>", self.interactions.on_right_paint_motion)
            self.canvas.bind("<Control-ButtonRelease-1>", self.interactions.on_button_release)
            # Also bind the actual right button for mice with right buttons
            self.canvas.bind("<Button-2>", self.interactions.on_canvas_right_click)
            self.canvas.bind("<B2-Motion>", self.interactions.on_right_paint_motion)
            self.canvas.bind("<ButtonRelease-2>", self.interactions.on_button_release)
            self.canvas.bind("<Button-3>", self.interactions.on_canvas_right_click)
            self.canvas.bind("<B3-Motion>", self.interactions.on_right_paint_motion)
            self.canvas.bind("<ButtonRelease-3>", self.interactions.on_button_release)
        else:
            # On Windows/Linux, it's usually Button-3
            self.canvas.bind("<Button-3>", self.interactions.on_canvas_right_click)
            self.canvas.bind("<B3-Motion>", self.interactions.on_right_paint_motion)
            self.canvas.bind("<ButtonRelease-3>", self.interactions.on_button_release)
        
        # Shortcut to switch to Move mode with spacebar (not hold)
        self.mask_window.bind("<KeyPress-space>", lambda e: [self.interaction_mode.set("move"), self.interactions.switch_mode()])
        
        # Bind for mouse wheel zoom - simplified approach
        # First bind to the canvas directly
        self.canvas.bind("<MouseWheel>", self.interactions.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.interactions.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.interactions.on_mouse_wheel)
        
        # Then bind to all widgets to catch events that might be missed
        self.mask_window.bind_all("<MouseWheel>", self.interactions.on_mouse_wheel)
        self.mask_window.bind_all("<Button-4>", self.interactions.on_mouse_wheel)
        self.mask_window.bind_all("<Button-5>", self.interactions.on_mouse_wheel)
        
        # Add mouse motion bindings
        self.canvas.bind("<B1-Motion>", self.interactions.on_mouse_motion)
    
    def add_info_label(self, parent):
        """Add information label to the UI"""
        info_text = ("Segment Anything Model (SAM) Mask Generator\n"
                    "Point mode: Left-click for foreground, right-click for background\n"
                    "Box mode: Click and drag to define a bounding box\n"
                    "Paint mode: Left-click to add to mask, right-click to remove from mask\n"
                    "Move mode: Click and drag to pan/move the view\n"
                    "Use + and - buttons to zoom in/out, 'Reset View' to fit to window\n"
                    "Use [ and ] keys to adjust brush size\n"
                    "Use Ctrl+Z to undo, Ctrl+Y to redo\n"
                    "Note: Paint mode requires a generated mask first\n"
                    "Generate masks at different frames to create checkpoints")
        ttk.Label(parent, text=info_text, justify=tk.CENTER).pack(pady=5)
        
    def load_settings(self):
        """Load user settings from file"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    
                    # Load brush size
                    if 'brush_size' in settings:
                        self.paint_radius = settings['brush_size']
                    
                    print(f"Loaded settings: brush size = {self.paint_radius}")
            else:
                print("No settings file found, using defaults")
        except Exception as e:
            print(f"Error loading settings: {str(e)}")
            
    def save_settings(self):
        """Save user settings to file"""
        try:
            settings = {
                'brush_size': self.paint_radius
                # Add more settings as needed
            }
            
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f)
                
            print("Settings saved successfully")
        except Exception as e:
            print(f"Error saving settings: {str(e)}")
