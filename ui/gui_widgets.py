# gui_widgets.py - v1.1737774900
# Updated: Friday, January 24, 2025 at 17:45:00 PST
# Changes in this version:
# - Added video quality/bitrate controls to improve output quality and reduce compression artifacts
# - Added codec selection dropdown (H.264, H.265, VP9) with automatic fallbacks
# - Added video quality preset dropdown (Low, Medium, High, Very High, Lossless)
# - Added custom bitrate option with slider control
# - Organized video settings into dedicated section within Basic Controls
# - Updated tooltips to explain new video quality options

"""
Widget creation and layout management for MatAnyone GUI.
Handles the creation and layout of all GUI widgets.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, StringVar, BooleanVar, IntVar, DoubleVar
from tkinter.scrolledtext import ScrolledText
import platform

# Import our UI components
from ui.ui_components import TextRedirector, Tooltip, CustomSlider


class WidgetManager:
    """Manages widget creation and layout for the MatAnyone GUI"""
    
    def __init__(self, app):
        """
        Initialize the widget manager
        
        Args:
            app: Reference to the main MatAnyoneApp instance
        """
        self.app = app
        self.is_dark_theme = False
    
    def apply_theme(self):
        """Apply platform-specific theme"""
        style = ttk.Style()
        
        if platform.system() == "Windows":
            # For Windows, apply a dark theme
            try:
                # Use the theme file from the UI directory
                self.app.root.tk.call('source', os.path.join(os.path.dirname(__file__), 'azure_dark_tcl.tcl'))
                style.theme_use('azure-dark')
                self.app.root.configure(bg='#333333')
                self.is_dark_theme = True
            except tk.TclError:
                # Fallback if custom theme fails
                style.theme_use('vista')
                self.is_dark_theme = False
        elif platform.system() == "Darwin":  # macOS
            # Use native macOS look with dark mode detection
            style.theme_use('aqua')
            
            # Check if dark mode is active (macOS)
            try:
                # Attempt to detect dark mode
                import subprocess
                cmd = 'defaults read -g AppleInterfaceStyle'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                self.is_dark_theme = result.stdout.strip() == 'Dark'
            except:
                # Default to light mode if detection fails
                self.is_dark_theme = False
                
            # Set background based on theme
            if self.is_dark_theme:
                self.app.root.configure(bg='#333333')
            else:
                self.app.root.configure(bg='#F0F0F0')
        else:  # Linux and others
            # Use a modern theme or the default
            try:
                style.theme_use('clam')
            except:
                pass  # Use default theme if clam is not available
            self.is_dark_theme = False
            
        # Configure common styles
        style.configure('TLabel', font=('Segoe UI', 10))
        style.configure('TButton', font=('Segoe UI', 10))
        style.configure('TCheckbutton', font=('Segoe UI', 10))
        style.configure('TRadiobutton', font=('Segoe UI', 10))
        
        # Configure special styles
        style.configure('Process.TButton', font=('Segoe UI', 11, 'bold'))
        style.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'))
        style.configure('Section.TLabel', font=('Segoe UI', 10, 'bold'))
    
    def create_all_widgets(self):
        """Create all widgets for the GUI"""
        # Configure grid weights
        self.app.main_frame.columnconfigure(0, weight=1)
        
        # Create all sections
        self.create_input_section()
        self.create_options_section()
        self.create_enhanced_section()
        self.create_process_section()
        self.create_progress_section()
        self.create_console_section()
        self.create_button_section()
        
        # Configure row weights for main frame
        self.app.main_frame.rowconfigure(5, weight=1)  # Make console expandable
    
    def create_input_section(self):
        """Create the input/output section"""
        # Input section - Row 0
        input_section = ttk.LabelFrame(self.app.main_frame, text="Input/Output", padding=(10, 5))
        input_section.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        input_section.columnconfigure(1, weight=1)  # Make entry fields expandable
        
        # Input type selection
        input_type_frame = ttk.Frame(input_section)
        input_type_frame.grid(row=0, column=0, columnspan=3, sticky="w", pady=5)
        
        ttk.Label(input_type_frame, text="Input Type:").pack(side=tk.LEFT, padx=(0,10))
        ttk.Radiobutton(input_type_frame, text="Video File", variable=self.app.input_type, value="video", 
                       command=self.app.event_handler.update_input_label).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(input_type_frame, text="Image Sequence", variable=self.app.input_type, value="sequence", 
                       command=self.app.event_handler.update_input_label).pack(side=tk.LEFT, padx=5)
        
        # Video/Sequence input selection - Row 1
        self.app.input_label = ttk.Label(input_section, text="Input Video:")
        self.app.input_label.grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(input_section, textvariable=self.app.video_path).grid(row=1, column=1, sticky="ew", pady=5)
        
        # Input Video browse button
        self.app.browse_input_btn = ttk.Button(input_section, text="Browse...", command=self.app.event_handler.browse_input, width=8)
        self.app.browse_input_btn.grid(row=1, column=2, padx=5, pady=5, sticky="e")
        
        # Mask file selection - Row 2
        ttk.Label(input_section, text="Input Mask:").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(input_section, textvariable=self.app.mask_path).grid(row=2, column=1, sticky="ew", pady=5)
        
        # Mask button - browse only
        ttk.Button(input_section, text="Browse...", command=self.app.event_handler.browse_mask, width=8).grid(row=2, column=2, padx=5, pady=5, sticky="e")
        
        # Keyframe status label - Row 2.5 (spans columns)
        self.app.keyframe_status = tk.StringVar()
        self.app.keyframe_status_label = ttk.Label(input_section, textvariable=self.app.keyframe_status, foreground="yellow", font=("TkDefaultFont", 9, "italic"))
        self.app.keyframe_status_label.grid(row=3, column=1, sticky="w", pady=(0, 5))
        
        # Output directory selection - Row 4
        ttk.Label(input_section, text="Output Folder:").grid(row=4, column=0, sticky="w", pady=5)
        ttk.Entry(input_section, textvariable=self.app.output_path).grid(row=4, column=1, sticky="ew", pady=5)
        
        # Output folder browse button
        ttk.Button(input_section, text="Browse...", command=self.app.event_handler.browse_output, width=8).grid(row=4, column=2, padx=5, pady=5, sticky="e")
    
    def create_options_section(self):
        """Create the processing options section"""
        # Processing options section - Row 1
        options_section = ttk.LabelFrame(self.app.main_frame, text="Processing Options", padding=(10, 5))
        options_section.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # Create three distinct sub-sections
        # Basic Controls (left)
        basic_options = ttk.LabelFrame(options_section, text="Basic Controls", padding=(5, 3))
        basic_options.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        basic_options.columnconfigure(1, weight=1)
        
        # Mask Controls (center)
        mask_options = ttk.LabelFrame(options_section, text="Mask Controls", padding=(5, 3))
        mask_options.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        mask_options.columnconfigure(1, weight=1)
        
        # Advanced Controls (right)
        advanced_options = ttk.LabelFrame(options_section, text="Advanced Controls", padding=(5, 3))
        advanced_options.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        advanced_options.columnconfigure(1, weight=1)
        
        # Configure weights for the three sections
        options_section.columnconfigure(0, weight=1)
        options_section.columnconfigure(1, weight=1)
        options_section.columnconfigure(2, weight=1)
        
        # Populate sections
        self.populate_basic_controls(basic_options)
        self.populate_mask_controls(mask_options)
        self.populate_advanced_controls(advanced_options)
    
    def populate_basic_controls(self, parent):
        """Populate the basic controls section"""
        # Warmup Frames
        self.app.n_warmup = IntVar(value=10)
        ttk.Label(parent, text="Warmup Frames:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Spinbox(parent, from_=1, to=30, textvariable=self.app.n_warmup, width=5).grid(row=0, column=1, sticky="w", pady=5)
        
        # Max Size
        self.app.max_size = IntVar(value=512)
        ttk.Label(parent, text="Max Size:").grid(row=1, column=0, sticky="w", pady=5)
        max_size_frame = ttk.Frame(parent)
        max_size_frame.grid(row=1, column=1, sticky="w", pady=5)
        ttk.Spinbox(max_size_frame, from_=-1, to=2000, textvariable=self.app.max_size, width=5).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(max_size_frame, text="(-1 for native)").pack(side=tk.LEFT)
        
        # Video Quality Section
        ttk.Label(parent, text="Video Quality:", style="Section.TLabel").grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 5))
        
        # Video Codec
        self.app.video_codec = StringVar(value="H.264")
        ttk.Label(parent, text="Codec:").grid(row=3, column=0, sticky="w", pady=5)
        codec_combo = ttk.Combobox(parent, textvariable=self.app.video_codec, width=10, state="readonly")
        codec_combo['values'] = ('Auto', 'H.264', 'H.265', 'VP9')
        codec_combo.grid(row=3, column=1, sticky="w", pady=5)
        codec_info = ttk.Label(parent, text="ⓘ")
        codec_info.grid(row=3, column=2, sticky="w", padx=5)
        Tooltip(codec_info, "Video codec for encoding:\n• Auto: Automatic codec selection\n• H.264: Wide compatibility, good quality\n• H.265: Better compression, newer players\n• VP9: Google's codec, very efficient")
        
        # Video Quality Preset
        self.app.video_quality = StringVar(value="High")
        ttk.Label(parent, text="Quality:").grid(row=4, column=0, sticky="w", pady=5)
        quality_combo = ttk.Combobox(parent, textvariable=self.app.video_quality, width=10, state="readonly")
        quality_combo['values'] = ('Low', 'Medium', 'High', 'Very High', 'Lossless')
        quality_combo.grid(row=4, column=1, sticky="w", pady=5)
        quality_info = ttk.Label(parent, text="ⓘ")
        quality_info.grid(row=4, column=2, sticky="w", padx=5)
        Tooltip(quality_info, "Video quality preset:\n• Low: Fast encoding, small files\n• Medium: Balanced quality/size\n• High: Good quality, recommended\n• Very High: Excellent quality, larger files\n• Lossless: No quality loss, largest files")
        
        # Custom Bitrate (only shown when not using Lossless)
        self.app.custom_bitrate_enabled = BooleanVar(value=False)
        self.app.custom_bitrate = IntVar(value=8000)  # 8 Mbps default
        
        self.app.custom_bitrate_frame = ttk.Frame(parent)
        self.app.custom_bitrate_frame.grid(row=5, column=0, columnspan=3, sticky="w", pady=5)
        
        ttk.Checkbutton(self.app.custom_bitrate_frame, text="Custom Bitrate:", 
                       variable=self.app.custom_bitrate_enabled,
                       command=self.toggle_custom_bitrate).pack(side=tk.LEFT)
        
        self.app.bitrate_slider = CustomSlider(
            self.app.custom_bitrate_frame,
            variable=self.app.custom_bitrate,
            from_=1000,
            to=50000,  # Up to 50 Mbps
            length=120,
            show_value=False,
            show_spinbox=True
        )
        self.app.bitrate_slider.pack(side=tk.LEFT, padx=(10, 5))
        
        ttk.Label(self.app.custom_bitrate_frame, text="kbps").pack(side=tk.LEFT)
        
        bitrate_info = ttk.Label(self.app.custom_bitrate_frame, text="ⓘ")
        bitrate_info.pack(side=tk.LEFT, padx=5)
        Tooltip(bitrate_info, "Custom bitrate in kbps:\n• 1000-3000: Low quality\n• 3000-8000: Good quality\n• 8000-15000: High quality\n• 15000+: Very high quality\nHigher bitrates = better quality but larger files")
        
        # Initially disable custom bitrate controls
        self.toggle_custom_bitrate()
        
        # Save Individual Frames option
        self.app.save_image = BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Save Individual Frames", variable=self.app.save_image).grid(row=6, column=0, columnspan=2, sticky="w", pady=5)
        
        # Cleanup Temporary Files
        self.app.cleanup_temp = BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Cleanup Temporary Files", variable=self.app.cleanup_temp).grid(row=7, column=0, columnspan=2, sticky="w", pady=5)
    
    def toggle_custom_bitrate(self):
        """Toggle the custom bitrate controls based on checkbox state"""
        if self.app.custom_bitrate_enabled.get():
            # Enable custom bitrate slider
            for widget in self.app.bitrate_slider.winfo_children():
                widget.configure(state='normal')
        else:
            # Disable custom bitrate slider
            for widget in self.app.bitrate_slider.winfo_children():
                try:
                    widget.configure(state='disabled')
                except:
                    pass  # Some widgets might not support state
    
    def populate_mask_controls(self, parent):
        """Populate the mask controls section"""
        # Erosion Radius with slider
        self.app.r_erode = IntVar(value=10)
        ttk.Label(parent, text="Erosion Radius:").grid(row=0, column=0, sticky="w", pady=5)
        
        erode_frame = ttk.Frame(parent)
        erode_frame.grid(row=0, column=1, sticky="ew", pady=5)
        
        # Create custom slider - with spinbox only (no label)
        self.app.erode_slider = CustomSlider(
            erode_frame, 
            variable=self.app.r_erode,
            from_=0,
            to=50,
            length=150,
            show_value=False,  # Don't show value label
            show_spinbox=True
        )
        self.app.erode_slider.pack(fill="x", expand=True)
        
        # Dilation Radius with slider
        self.app.r_dilate = IntVar(value=15)
        ttk.Label(parent, text="Dilation Radius:").grid(row=1, column=0, sticky="w", pady=5)
        
        dilate_frame = ttk.Frame(parent)
        dilate_frame.grid(row=1, column=1, sticky="ew", pady=5)
        
        # Create custom slider - with spinbox only (no label)
        self.app.dilate_slider = CustomSlider(
            dilate_frame, 
            variable=self.app.r_dilate,
            from_=0,
            to=50,
            length=150,
            show_value=False,  # Don't show value label
            show_spinbox=True
        )
        self.app.dilate_slider.pack(fill="x", expand=True)
        
        # Add Generate Mask button at the bottom of the Mask Controls section
        # Check if mask generator is available
        try:
            from mask.mask_generator import MaskGeneratorUI
            has_mask_generator = True
        except ImportError:
            has_mask_generator = False
        
        if has_mask_generator:
            generate_frame = ttk.Frame(parent)
            generate_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=10)
            ttk.Button(generate_frame, text="Generate Mask", command=self.app.event_handler.generate_mask).pack(fill="x", expand=True)
    
    def populate_advanced_controls(self, parent):
        """Populate the advanced controls section"""
        # Horizontal Chunks
        self.app.num_chunks = IntVar(value=2)
        self.app.chunks_label = ttk.Label(parent, text="Number of Chunks:")
        self.app.chunks_label.grid(row=0, column=0, sticky="w", pady=5)
        
        self.app.chunks_frame = ttk.Frame(parent)
        self.app.chunks_frame.grid(row=0, column=1, sticky="ew", pady=5)
        
        self.app.chunks_spinbox = ttk.Spinbox(self.app.chunks_frame, from_=1, to=16, textvariable=self.app.num_chunks, width=5)
        self.app.chunks_spinbox.pack(side=tk.LEFT, padx=(0, 5))
        
        self.app.chunks_info_label = ttk.Label(self.app.chunks_frame, text="(2+ for chunking)")
        self.app.chunks_info_label.pack(side=tk.LEFT)
        
        # Chunk Type - Radio Buttons
        self.app.chunk_type = StringVar(value="strips")
        chunk_type_frame = ttk.Frame(parent)
        chunk_type_frame.grid(row=1, column=0, columnspan=2, sticky="w", pady=5)
        ttk.Label(chunk_type_frame, text="Chunk Type:").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(chunk_type_frame, text="Horizontal Strips", variable=self.app.chunk_type, value="strips").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(chunk_type_frame, text="Aspect-ratio Grid", variable=self.app.chunk_type, value="grid").pack(side=tk.LEFT, padx=5)
        
        # Add tooltip for chunk types
        chunk_info_text = ("Horizontal Strips: Divides image into horizontal strips (traditional method)\n"
                         "Aspect-ratio Grid: Divides image into a grid of chunks that maintain aspect ratio")
        chunk_info = ttk.Label(chunk_type_frame, text="ⓘ")
        chunk_info.pack(side=tk.LEFT, padx=5)
        Tooltip(chunk_info, chunk_info_text)
        
        # Bidirectional Processing option
        self.app.bidirectional = BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Bidirectional Processing", variable=self.app.bidirectional).grid(row=2, column=0, columnspan=2, sticky="w", pady=5)
        
        # Reverse Dilation with slider
        self.app.reverse_dilate = IntVar(value=15)
        ttk.Label(parent, text="Reverse Dilation:").grid(row=3, column=0, sticky="w", pady=5)
        
        rev_dilate_frame = ttk.Frame(parent)
        rev_dilate_frame.grid(row=3, column=1, sticky="ew", pady=5)
        
        # Create custom slider - with spinbox only (no label)
        self.app.rev_dilate_slider = CustomSlider(
            rev_dilate_frame, 
            variable=self.app.reverse_dilate,
            from_=0,
            to=50,
            length=150,
            show_value=False,  # Don't show value label
            show_spinbox=True
        )
        self.app.rev_dilate_slider.pack(fill="x", expand=True)
        
        # Blend Method
        self.app.blend_method = StringVar(value="weighted")
        ttk.Label(parent, text="Blend Method:").grid(row=4, column=0, sticky="w", pady=5)
        blend_combo = ttk.Combobox(parent, textvariable=self.app.blend_method, width=15, state="readonly")
        blend_combo['values'] = ('weighted', 'max_alpha', 'min_alpha', 'average')
        blend_combo.grid(row=4, column=1, sticky="w", pady=5)
    
    def create_enhanced_section(self):
        """Create the enhanced chunk processing section"""
        # Enhanced chunk processing section - Row 2
        enhanced_section = ttk.LabelFrame(self.app.main_frame, text="Enhanced Chunk Processing", padding=(10, 5))
        enhanced_section.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        enhanced_section.columnconfigure(0, weight=1)
        enhanced_section.columnconfigure(1, weight=2)
        
        # Enable enhanced processing
        self.app.use_enhanced_chunks = BooleanVar(value=True)
        ttk.Checkbutton(enhanced_section, text="Use Enhanced Chunk Processing", 
                       variable=self.app.use_enhanced_chunks, 
                       command=self.app.event_handler.toggle_enhanced_options).grid(row=0, column=0, sticky="w", pady=5)
        
        # Tooltip for enhanced processing
        enhanced_help_text = ("Enhanced chunk processing creates a continuous bidirectional mask for the entire video at reduced resolution,\n"
                             "then identifies optimal keyframes with maximum mask coverage for each range of frames.\n"
                             "It processes only the frame ranges where objects are visible with bidirectional processing around optimal keyframes.\n"
                             "This addresses issues with objects moving between chunks and reduces processing time.")
        
        enhanced_info = ttk.Label(enhanced_section, text="ⓘ")
        enhanced_info.grid(row=0, column=1, sticky="w", pady=5)
        Tooltip(enhanced_info, enhanced_help_text)
        
        # Enhanced options frame (will be shown/hidden based on checkbox)
        self.app.enhanced_options_frame = ttk.Frame(enhanced_section)
        self.app.enhanced_options_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        self.app.enhanced_options_frame.columnconfigure(1, weight=1)
        
        self.populate_enhanced_options()
    
    def populate_enhanced_options(self):
        """Populate the enhanced processing options"""
        # Add Autochunk mode
        self.app.use_autochunk = BooleanVar(value=False)
        autochunk_frame = ttk.Frame(self.app.enhanced_options_frame)
        autochunk_frame.grid(row=0, column=0, columnspan=2, sticky="w", pady=5)
        
        self.app.autochunk_checkbox = ttk.Checkbutton(autochunk_frame, text="Use Auto-chunk Mode", 
                                         variable=self.app.use_autochunk,
                                         command=self.app.event_handler.toggle_autochunk)
        self.app.autochunk_checkbox.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add tooltip for autochunk
        autochunk_help_text = ("Auto-chunk mode automatically determines the ideal number and size of chunks based on the low-resolution mask.\n"
                             "Each chunk will be the same size as the low-res dimensions, with appropriate overlap.\n"
                             "When enabled, the 'Number of Chunks' setting is ignored and calculated automatically.\n"
                             "This produces optimal results by ensuring chunks are properly sized for the model.")
        autochunk_info = ttk.Label(autochunk_frame, text="ⓘ")
        autochunk_info.pack(side=tk.LEFT)
        Tooltip(autochunk_info, autochunk_help_text)
        
        # Low-res scale factor
        self.app.low_res_scale = DoubleVar(value=0.25)
        ttk.Label(self.app.enhanced_options_frame, text="Low-res Scale:").grid(row=1, column=0, sticky="w", pady=5)
        
        scale_frame = ttk.Frame(self.app.enhanced_options_frame)
        scale_frame.grid(row=1, column=1, sticky="ew", pady=5)
        
        # Create a combobox for preset scales with added 3/4 option
        scale_combo = ttk.Combobox(scale_frame, textvariable=self.app.low_res_scale, width=5, state="readonly")
        scale_combo['values'] = (0.125, 0.25, 0.5, 0.75)  # Added 0.75 (3/4) option
        scale_combo.pack(side=tk.LEFT, padx=(0, 5))
        
        # Add low-res blend method selection
        self.app.lowres_blend_method = StringVar(value="")  # Empty means use same as final blend
        ttk.Label(self.app.enhanced_options_frame, text="Low-res Blend Method:").grid(row=2, column=0, sticky="w", pady=5)
        
        lowres_blend_frame = ttk.Frame(self.app.enhanced_options_frame)
        lowres_blend_frame.grid(row=2, column=1, sticky="ew", pady=5)
        
        lowres_blend_combo = ttk.Combobox(lowres_blend_frame, textvariable=self.app.lowres_blend_method, width=15, state="readonly")
        lowres_blend_combo['values'] = ('(Same as final)', 'weighted', 'max_alpha', 'min_alpha', 'average')
        lowres_blend_combo.current(0)  # Default to "Same as final"
        lowres_blend_combo.pack(side=tk.LEFT, padx=(0, 5))
        
        # Add tooltip for low-res blend
        lowres_blend_help = "Different blend methods can provide better results for the low-resolution mask vs. final output"
        lowres_blend_info = ttk.Label(lowres_blend_frame, text="ⓘ")
        lowres_blend_info.pack(side=tk.LEFT)
        Tooltip(lowres_blend_info, lowres_blend_help)
        
        # Mask threshold
        self.app.mask_threshold = IntVar(value=5)
        ttk.Label(self.app.enhanced_options_frame, text="Mask Threshold:").grid(row=3, column=0, sticky="w", pady=5)
        
        threshold_frame = ttk.Frame(self.app.enhanced_options_frame)
        threshold_frame.grid(row=3, column=1, sticky="ew", pady=5)
        
        # Create threshold slider
        self.app.threshold_slider = CustomSlider(
            threshold_frame,
            variable=self.app.mask_threshold,
            from_=1,
            to=50,  # Increased max value to allow more control
            length=150,
            show_value=False,
            show_spinbox=True
        )
        self.app.threshold_slider.pack(side=tk.LEFT, fill="x", expand=True)
        ttk.Label(threshold_frame, text="% (minimum mask coverage required)").pack(side=tk.LEFT, padx=5)
        
        # Add face detection option
        self.app.prioritize_faces = BooleanVar(value=True)
        face_frame = ttk.Frame(self.app.enhanced_options_frame)
        face_frame.grid(row=4, column=0, columnspan=2, sticky="w", pady=5)
        
        face_checkbox = ttk.Checkbutton(face_frame, text="Prioritize Frames with Faces", 
                                    variable=self.app.prioritize_faces)
        face_checkbox.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add parallel processing option
        self.app.use_parallel_processing = BooleanVar(value=True)
        parallel_frame = ttk.Frame(self.app.enhanced_options_frame)
        parallel_frame.grid(row=5, column=0, columnspan=2, sticky="w", pady=5)
        
        parallel_checkbox = ttk.Checkbutton(parallel_frame, text="Use Parallel Processing", 
                                       variable=self.app.use_parallel_processing)
        parallel_checkbox.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add tooltip for parallel processing
        parallel_help_text = ("Enable parallel processing of video chunks for faster performance.\n"
                            "This can significantly speed up processing on multi-core systems.\n"
                            "Disable if you encounter memory issues or unstable behavior.")
        parallel_info = ttk.Label(parallel_frame, text="ⓘ")
        parallel_info.pack(side=tk.LEFT)
        Tooltip(parallel_info, parallel_help_text)
        
        # Add tooltip for face detection
        face_help_text = ("When enabled, the system will detect faces in frames and prioritize them as keyframes.\n"
                         "This can significantly improve the quality of human subjects in the final output.\n"
                         "If no faces are detected, it falls back to using frames with maximum mask coverage.")
        face_info = ttk.Label(face_frame, text="ⓘ")
        face_info.pack(side=tk.LEFT)
        Tooltip(face_info, face_help_text)
    
    def create_process_section(self):
        """Create the process button section"""
        # Process button section - Row 3
        process_section = ttk.Frame(self.app.main_frame)
        process_section.grid(row=3, column=0, sticky="ew", padx=5, pady=10)
        process_section.columnconfigure(0, weight=1)
        
        # Process button
        self.app.process_button = ttk.Button(
            process_section, 
            text="Process Video", 
            command=self.app.processing_manager.process_video,
            style="Process.TButton"
        )
        self.app.process_button.grid(row=0, column=0)
    
    def create_progress_section(self):
        """Create the progress bar and status section"""
        # Progress bar and status - Row 4
        progress_section = ttk.Frame(self.app.main_frame)
        progress_section.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        progress_section.columnconfigure(0, weight=1)
        
        # Create a frame for the progress bar and label
        self.app.progress_frame = ttk.Frame(progress_section)
        self.app.progress_frame.grid(row=0, column=0, sticky="ew", pady=5)
        self.app.progress_frame.columnconfigure(0, weight=1)
        
        # Progress bar
        self.app.progress = ttk.Progressbar(
            self.app.progress_frame, 
            orient=tk.HORIZONTAL, 
            length=100, 
            mode='determinate'
        )
        self.app.progress.grid(row=0, column=0, sticky="ew")
        
        # Progress status frame (to hold percentage and stage indicator)
        progress_status_frame = ttk.Frame(progress_section)
        progress_status_frame.grid(row=1, column=0, sticky="ew")
        progress_status_frame.columnconfigure(0, weight=1)  # Left-align percentage
        progress_status_frame.columnconfigure(1, weight=1)  # Right-align stage
        
        # Progress percentage (left side)
        self.app.progress_text = StringVar(value="")
        self.app.progress_label = ttk.Label(
            progress_status_frame, 
            textvariable=self.app.progress_text,
            anchor="w"  # Left-align
        )
        self.app.progress_label.grid(row=0, column=0, sticky="w", padx=(5, 0))
        
        # Stage indicator (right side)
        self.app.progress_stage = StringVar(value="")
        self.app.stage_label = ttk.Label(
            progress_status_frame,
            textvariable=self.app.progress_stage,
            anchor="e"  # Right-align
        )
        self.app.stage_label.grid(row=0, column=1, sticky="e", padx=(0, 5))
        
        # Status label
        self.app.status_var = StringVar(value="Ready")
        ttk.Label(progress_section, textvariable=self.app.status_var).grid(row=2, column=0, pady=5)
    
    def create_console_section(self):
        """Create the console output section"""
        # Console output - Row 5
        self.app.console_section = ttk.LabelFrame(self.app.main_frame, text="Console Output")
        self.app.console_section.grid(row=5, column=0, sticky="nsew", padx=5, pady=5)
        self.app.console_section.columnconfigure(0, weight=1)
        self.app.console_section.rowconfigure(0, weight=1)
        
        # Create the console with platform-specific styling for better visibility
        console_bg = "#2D2D2D" if platform.system() == "Darwin" and self.is_dark_theme else "white"
        console_fg = "white" if platform.system() == "Darwin" and self.is_dark_theme else "black"
        console_font = ("Menlo" if platform.system() == "Darwin" else "Courier", 11 if platform.system() == "Darwin" else 9)
        
        self.app.console = ScrolledText(self.app.console_section, height=12, wrap=tk.WORD, 
                                   bg=console_bg, fg=console_fg, font=console_font)
        self.app.console.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Initialize console with a message
        self.app.console.configure(state=tk.NORMAL)
        self.app.console.insert(tk.END, "Console initialized. Ready for output...\n")
        self.app.console.configure(state=tk.DISABLED)
    
    def create_button_section(self):
        """Create the bottom button bar"""
        # Bottom button bar - Row 6
        button_section = ttk.Frame(self.app.main_frame)
        button_section.grid(row=6, column=0, sticky="ew", padx=5, pady=5)
        button_section.columnconfigure(1, weight=1)  # Middle space expands
        
        # Add help button (left)
        help_button = ttk.Button(button_section, text="Help", command=self.app.event_handler.show_help)
        help_button.grid(row=0, column=0, pady=5, padx=5, sticky="w")
        
        # Add about button (right)
        about_button = ttk.Button(button_section, text="About", command=self.app.event_handler.show_about)
        about_button.grid(row=0, column=2, pady=5, padx=5, sticky="e")
