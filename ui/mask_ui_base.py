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
from ui.ui_components import Tooltip, create_message_dialog

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
        self._updating_slider = False
        self.mask_generated_frames = set()  # Track frames where masks have been generated
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
        
        # Create SAM threshold settings frame
        self.create_threshold_settings_frame(main_frame)
        
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
        
        # Info label removed - now accessible via Help button
        
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
        
        # Set brush size from main app config if available
        if hasattr(self, 'main_app') and self.main_app:
            try:
                saved_size = self.main_app.mask_brush_size
                if saved_size in brush_sizes:
                    self.brush_size_combo.set(saved_size)
                else:
                    self.brush_size_combo.current(2)  # Default to 5px
            except:
                self.brush_size_combo.current(2)  # Default to 5px
        else:
            self.brush_size_combo.current(2)  # Default to 5px
            
        self.brush_size_combo.pack(side=tk.LEFT, padx=5)
        self.brush_size_combo.bind("<<ComboboxSelected>>", self._on_brush_size_changed)
        
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
    
    def create_threshold_settings_frame(self, parent):
        """Create the SAM threshold settings frame"""
        self.threshold_frame = ttk.LabelFrame(parent, text="SAM Threshold Settings (Advanced)")
        self.threshold_frame.pack(pady=5, fill=tk.X, padx=20)
        
        # Create variables for thresholds
        self.mask_threshold_var = tk.DoubleVar(value=0.0)
        self.stability_threshold_var = tk.DoubleVar(value=0.95)
        
        # Mask threshold slider
        mask_thresh_frame = ttk.Frame(self.threshold_frame)
        mask_thresh_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(mask_thresh_frame, text="Mask Threshold:").pack(side=tk.LEFT, padx=5)
        mask_thresh_slider = ttk.Scale(mask_thresh_frame, from_=-3.0, to=3.0, 
                                     variable=self.mask_threshold_var,
                                     orient=tk.HORIZONTAL, length=200,
                                     command=self._on_threshold_changed)
        mask_thresh_slider.pack(side=tk.LEFT, padx=5)
        
        self.mask_thresh_label = ttk.Label(mask_thresh_frame, text="0.0")
        self.mask_thresh_label.pack(side=tk.LEFT, padx=5)
        
        # Add tooltip
        Tooltip(mask_thresh_slider, "Controls mask sensitivity. Lower = more inclusive, Higher = more selective")
        
        # Stability threshold slider
        stability_frame = ttk.Frame(self.threshold_frame)
        stability_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(stability_frame, text="Stability Filter:").pack(side=tk.LEFT, padx=5)
        stability_slider = ttk.Scale(stability_frame, from_=0.0, to=1.0,
                                   variable=self.stability_threshold_var,
                                   orient=tk.HORIZONTAL, length=200,
                                   command=self._on_stability_changed)
        stability_slider.pack(side=tk.LEFT, padx=5)
        
        self.stability_label = ttk.Label(stability_frame, text="0.95")
        self.stability_label.pack(side=tk.LEFT, padx=5)
        
        # Add tooltip
        Tooltip(stability_slider, "Filters unstable mask edges. Higher = cleaner edges but may lose detail")
        
        # Reset button
        reset_button = ttk.Button(self.threshold_frame, text="Reset to Defaults", 
                                command=self._reset_thresholds)
        reset_button.pack(pady=5)
        
        # Initially hide the threshold frame (can be toggled with a button)
        self.threshold_frame.pack_forget()
        
        # Add toggle button in the main UI
        self.show_thresholds = False
    
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
        
        # Add Load Existing Mask button here (moved from bottom)
        ttk.Button(button_frame, text="Load Existing Mask", 
                 command=lambda: self._load_existing_mask(mask_save_path)).pack(side=tk.LEFT, padx=10)
        
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
        
        # Add help button
        ttk.Button(zoom_frame, text="Help", 
                 command=self.show_help_dialog).pack(side=tk.LEFT, padx=5)
        
        # Add threshold settings toggle button
        self.threshold_button = ttk.Button(zoom_frame, text="Thresholds", 
                                         command=self._toggle_threshold_settings)
        self.threshold_button.pack(side=tk.LEFT, padx=5)
        
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
        
        # Frame slider with hashmarks
        slider_container = ttk.Frame(navigation_frame)
        slider_container.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        self.frame_slider = ttk.Scale(slider_container, from_=0, to=self.total_frames-1, 
                                    orient=tk.HORIZONTAL,
                                    command=self._on_slider_changed_discrete)
        self.frame_slider.pack(fill=tk.X)
        
        # Create hashmarks canvas (no explicit background to match theme)
        self.hashmarks_canvas = tk.Canvas(slider_container, height=30)
        self.hashmarks_canvas.pack(fill=tk.X)
        self.create_hashmarks()
        
        # Bind resize event to redraw hashmarks
        self.hashmarks_canvas.bind('<Configure>', lambda e: self.create_hashmarks())
        
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
    
    def create_hashmarks(self, current_frame=None):
        """Create hashmarks under the scrub bar"""
        if not hasattr(self, 'hashmarks_canvas'):
            return
            
        # Clear any existing hashmarks
        self.hashmarks_canvas.delete("all")
        
        # Get canvas width after it's been packed
        self.hashmarks_canvas.update_idletasks()
        canvas_width = self.hashmarks_canvas.winfo_width()
        
        if canvas_width <= 1:  # Canvas not yet drawn
            self.hashmarks_canvas.after(100, lambda: self.create_hashmarks(current_frame))
            return
        
        # Calculate spacing between frames
        if self.total_frames <= 1:
            return
            
        frame_spacing = canvas_width / (self.total_frames - 1)
        
        # Get current frame if not provided
        if current_frame is None and hasattr(self, 'current_frame'):
            current_frame = self.current_frame
        
        # Draw hashmarks
        for frame in range(self.total_frames):
            x_pos = frame * frame_spacing
            
            # Determine color priority: red for mask generated, yellow for current, white/grey for normal
            is_current = (current_frame is not None and frame == current_frame)
            has_mask = frame in self.mask_generated_frames
            
            # Use different colors for Windows light theme
            if platform.system() == "Windows":
                if has_mask:
                    color = "#d00000"  # Dark red for frames with generated masks
                elif is_current:
                    color = "#0078d4"  # Windows blue for current frame
                else:
                    color = "black" if frame % 5 == 0 else "#808080"  # Black/gray for normal
            else:
                if has_mask:
                    color = "red"  # Persistent red for frames with generated masks
                elif is_current:
                    color = "yellow"  # Yellow for current frame position
                else:
                    color = "white" if frame % 5 == 0 else "lightgray"  # Normal colors
            
            # Long hashmark every 5th frame (0, 5, 10, etc.)
            if frame % 5 == 0:
                # Long hashmark
                self.hashmarks_canvas.create_line(x_pos, 0, x_pos, 15, fill=color, width=2)
                # Frame number
                self.hashmarks_canvas.create_text(x_pos, 22, text=str(frame), anchor="center", 
                                                font=("Arial", 10, "bold"), fill=color)
            else:
                # Short hashmark
                self.hashmarks_canvas.create_line(x_pos, 0, x_pos, 8, fill=color, width=1)
    
    def _on_slider_changed_discrete(self, value):
        """Handle slider changes with discrete integer stepping"""
        # Prevent recursion when we set the slider value
        if self._updating_slider:
            return
            
        # Convert to integer for discrete stepping
        int_value = int(round(float(value)))
        
        # Set flag to prevent recursion and update slider to integer value
        self._updating_slider = True
        self.frame_slider.set(int_value)
        self._updating_slider = False
        
        # Call the original callback with integer value
        self.frame_manager.on_slider_changed(str(int_value))
    
    def mark_frame_with_mask(self, frame_index):
        """Mark a frame as having a generated mask"""
        self.mask_generated_frames.add(frame_index)
        # Refresh hashmarks to show the red highlighting
        self.create_hashmarks()
    
    def _on_brush_size_changed(self, event=None):
        """Handle brush size change from combo box"""
        # Update the interactions
        self.interactions.update_brush_size(event)
        
        # Save to main app config if available
        if hasattr(self, 'main_app') and self.main_app:
            try:
                brush_size = int(self.brush_size_combo.get())
                self.main_app.mask_brush_size = brush_size
                # Save settings to file
                self.main_app.config_manager.save_settings(self.main_app)
            except Exception as e:
                print(f"Error saving brush size to config: {e}")
    
    def _toggle_threshold_settings(self):
        """Toggle visibility of threshold settings"""
        self.show_thresholds = not self.show_thresholds
        if self.show_thresholds:
            # Pack it in the main frame where it was originally created
            self.threshold_frame.pack(pady=5, fill=tk.X, padx=20)
            # Move it to the right position by re-packing other frames
            if hasattr(self, 'button_frame_container'):
                self.button_frame_container.pack_forget()
                self.button_frame_container.pack(pady=10)
        else:
            self.threshold_frame.pack_forget()
    
    def _on_threshold_changed(self, value):
        """Handle mask threshold change"""
        threshold = float(value)
        self.mask_thresh_label.config(text=f"{threshold:.2f}")
        
        # Update the mask generator if it exists
        if hasattr(self, 'mask_generator') and self.mask_generator:
            self.mask_generator.set_thresholds(mask_threshold=threshold)
    
    def _on_stability_changed(self, value):
        """Handle stability threshold change"""
        stability = float(value)
        self.stability_label.config(text=f"{stability:.2f}")
        
        # Update the mask generator if it exists
        if hasattr(self, 'mask_generator') and self.mask_generator:
            self.mask_generator.set_thresholds(stability_threshold=stability)
    
    def _reset_thresholds(self):
        """Reset thresholds to default values"""
        self.mask_threshold_var.set(0.0)
        self.stability_threshold_var.set(0.95)
        
        # Update labels
        self.mask_thresh_label.config(text="0.0")
        self.stability_label.config(text="0.95")
        
        # Update the mask generator
        if hasattr(self, 'mask_generator') and self.mask_generator:
            self.mask_generator.set_thresholds(mask_threshold=0.0, stability_threshold=0.95)
    
    def _load_existing_mask(self, mask_save_path):
        """Load the existing mask from the main GUI mask input field"""
        from tkinter import messagebox
        import os
        
        try:
            # Use the existing mask path passed from main GUI
            if not hasattr(self, 'existing_mask_path') or not self.existing_mask_path:
                messagebox.showerror("Error", "No mask path specified in main GUI.\nPlease enter a mask path in the main interface first.")
                return
                
            mask_file = self.existing_mask_path.strip()
            if not mask_file:
                messagebox.showerror("Error", "No mask path specified in main GUI.\nPlease enter a mask path in the main interface first.")
                return
                
            if not os.path.exists(mask_file):
                messagebox.showerror("Error", f"Mask file does not exist:\n{mask_file}")
                return
            
            # Read keyframe metadata to determine which frame this mask belongs to
            from mask.mask_utils import get_keyframe_metadata_from_mask
            keyframe = get_keyframe_metadata_from_mask(mask_file)
            
            if keyframe is not None:
                # Navigate to the keyframe before loading the mask
                print(f"Navigating to keyframe {keyframe}")
                self.frame_manager.set_current_frame(keyframe)
            
            # Load the mask using the existing checkpoint system
            self.editor.apply_loaded_mask(mask_file)
            
            # Mark the frame as having a mask after loading (this should happen in apply_loaded_mask but ensure it happens)
            target_frame = keyframe if keyframe is not None else self.current_frame_index
            self.mark_frame_with_mask(target_frame)
            
            # No popup needed - the frame jump and loaded mask are sufficient indication
            if keyframe is not None:
                print(f"Mask loaded successfully from: {mask_file}, navigated to keyframe {keyframe}")
            else:
                print(f"Mask loaded successfully from: {mask_file}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load mask:\n{str(e)}")
    
    def create_confirm_buttons(self, parent, video_path, mask_save_path):
        """Create confirm and cancel buttons"""
        # Bottom row of buttons for confirm/cancel only
        confirm_frame = ttk.Frame(parent)
        confirm_frame.pack(pady=15)
        
        # Confirm button (initially disabled)
        self.confirm_button = ttk.Button(confirm_frame, text="Confirm & Save", 
                                      command=lambda: self.frame_manager.save_and_close(video_path, mask_save_path),
                                      state=tk.DISABLED)
        self.confirm_button.pack(side=tk.LEFT, padx=20)
        
        ttk.Button(confirm_frame, text="Cancel", 
                 command=self.mask_window.destroy).pack(side=tk.LEFT, padx=20)
    
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
    
    def show_help_dialog(self):
        """Show detailed help information in a dialog"""
        help_text = """MatAnyone Mask Generator Help

🎯 OVERVIEW
The SAM (Segment Anything Model) mask generator allows you to create precise masks for video processing. You can generate masks on any frame in your video - there's no need to have your subject in frame 0!

📝 INTERACTION MODES

Point Mode:
• Left-click: Add foreground points (green)
• Right-click: Add background points (red)
• Best for: Precise object selection with clear boundaries

Box Mode:
• Click and drag: Draw a bounding box around your subject
• Best for: Quick selection of well-defined objects

Paint Mode:
• Left-click and drag: Paint to add areas to the mask
• Right-click and drag: Paint to remove areas from the mask
• Requires: A generated mask must exist first
• Best for: Fine-tuning mask boundaries

Move Mode:
• Click and drag: Pan/move the view around the image
• Spacebar: Quick toggle to Move mode
• Best for: Navigating large images

🎮 CONTROLS & SHORTCUTS

Zoom Controls:
• + button: Zoom in
• - button: Zoom out
• Reset View: Fit image to window size
• Mouse wheel: Zoom in/out at cursor position

Brush Size (Paint Mode):
• [ key: Decrease brush size
• ] key: Increase brush size
• Size range: 1-50 pixels

Edit Controls:
• Ctrl+Z: Undo last action
• Ctrl+Y: Redo last undone action
• Clear: Remove all points and selections
• Toggle Overlay: Show/hide the generated mask overlay

🎬 FRAME NAVIGATION

Navigation:
• ← / → buttons: Move between frames
• Frame slider: Jump to any frame directly
• Frame input: Type specific frame number

Checkpoints:
• Generate masks on different frames to create checkpoints
• Checkpoints appear as markers on the frame slider
• Allows you to refine masks at key moments in your video

💾 WORKFLOW

1. Navigate to a frame where your subject is clearly visible
2. Choose your interaction mode (Point/Box recommended to start)
3. Make your selections (points or box)
4. Click "Generate Mask" to create the initial mask
5. Use Paint mode to refine the mask if needed
6. Generate masks on additional frames for better results
7. Click "Confirm & Save" when satisfied

💡 TIPS

• Generate masks on frames with clear subject visibility
• Use multiple frames for complex movements or shape changes
• Point mode works best for objects with clear boundaries
• Box mode is fastest for well-defined rectangular subjects
• Paint mode is perfect for fine-tuning difficult areas
• The system processes forward and backward from your keyframes automatically

🎛️ THRESHOLD SETTINGS (Advanced)

Click "Thresholds" button to adjust mask generation sensitivity:
• Mask Threshold: Controls edge sensitivity (-3 to 3)
  - Lower values = more inclusive masks
  - Higher values = more selective masks
• Stability Filter: Reduces flickering/shuttering (0 to 1)
  - Higher values = cleaner edges but may lose detail
  - Lower values = preserve all details but may flicker"""

        create_message_dialog(
            self.mask_window, 
            "Mask Generator Help", 
            help_text
        )
        
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
        """Save user settings to main app config"""
        try:
            # Save to main app config if available
            if hasattr(self, 'main_app') and self.main_app:
                self.main_app.mask_brush_size = self.paint_radius
                self.main_app.config_manager.save_settings(self.main_app)
                print(f"Brush size saved to main config: {self.paint_radius}")
            else:
                # Fallback to local file if main app not available
                settings = {
                    'brush_size': self.paint_radius
                    # Add more settings as needed
                }
                
                with open(self.settings_file, 'w') as f:
                    json.dump(settings, f)
                    
                print("Settings saved to local file")
        except Exception as e:
            print(f"Error saving settings: {str(e)}")
