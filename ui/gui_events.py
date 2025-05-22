# gui_events.py - v1.1748000000
# Updated: Thursday, May 22, 2025 at 08:56:00 PDT
# Changes in this version:
# - Added keyframe metadata system documentation to help text
# - Implemented keyframe status display under mask input
# - Added automatic keyframe metadata checking when masks are selected or generated
# - Updated GUI to show "Frame [N] being used" for masks with keyframe metadata
# - Enhanced mask browsing and generation callbacks to update keyframe status

"""
Event handling for MatAnyone GUI.
Handles button clicks, callbacks, and user interactions.
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

# Try to import mask generator module
try:
    from mask.mask_generator import MaskGeneratorUI
    HAS_MASK_GENERATOR = True
except ImportError:
    print("Mask generator module not found. Mask generation will be disabled.")
    print("Make sure mask_generator.py is in the same directory as this script.")
    HAS_MASK_GENERATOR = False

# Import video utilities for mask generation fallback
try:
    from utils.image_processing import generate_mask_for_video
except ImportError:
    generate_mask_for_video = None

from ui.ui_components import create_message_dialog


class EventHandler:
    """Handles GUI events and user interactions"""
    
    def __init__(self, app):
        """
        Initialize the event handler
        
        Args:
            app: Reference to the main MatAnyoneApp instance
        """
        self.app = app
    
    def toggle_enhanced_options(self):
        """Show or hide enhanced chunk processing options based on checkbox"""
        if self.app.use_enhanced_chunks.get():
            self.app.enhanced_options_frame.grid()
            # Check if autochunk is enabled to update chunks spinbox state
            self.toggle_autochunk()
        else:
            self.app.enhanced_options_frame.grid_remove()
            # Enable the chunks spinbox when enhanced processing is disabled
            self.app.chunks_spinbox.config(state=tk.NORMAL)
            self.app.chunks_label.config(state=tk.NORMAL)
            self.app.chunks_info_label.config(state=tk.NORMAL)
            # Also disable auto-chunk if enhanced processing is disabled
            self.app.use_autochunk.set(False)

    def toggle_autochunk(self):
        """Toggle the state of the Number of Chunks input based on Auto-chunk checkbox"""
        if self.app.use_enhanced_chunks.get() and self.app.use_autochunk.get():
            # Disable the chunks spinbox when autochunk is enabled
            self.app.chunks_spinbox.config(state=tk.DISABLED)
            self.app.chunks_label.config(state=tk.DISABLED)
            self.app.chunks_info_label.config(state=tk.DISABLED)
            # Change info text to indicate chunks will be calculated automatically
            self.app.chunks_info_label.config(text="(Auto-calculated)")
        else:
            # Enable the chunks spinbox when autochunk is disabled
            self.app.chunks_spinbox.config(state=tk.NORMAL)
            self.app.chunks_label.config(state=tk.NORMAL)
            self.app.chunks_info_label.config(state=tk.NORMAL)
            # Reset info text
            self.app.chunks_info_label.config(text="(2+ for chunking)")

    def update_input_label(self):
        """Update the input label based on selected input type"""
        if self.app.input_type.get() == "video":
            self.app.input_label.config(text="Input Video:")
            self.app.browse_input_btn.config(command=self.browse_input)
        else:
            self.app.input_label.config(text="Image Sequence Folder:")
            self.app.browse_input_btn.config(command=self.browse_input)
    
    def browse_input(self):
        """Generic browse function that calls the appropriate browse method"""
        if self.app.input_type.get() == "video":
            self.browse_video()
        else:
            self.browse_sequence()
            
    def browse_video(self):
        """Browse for a video file"""
        filetypes = [
            ("Video files", "*.mp4 *.mov *.avi"),
            ("All files", "*.*")
        ]
        filename = filedialog.askopenfilename(
            title="Select Input Video",
            filetypes=filetypes
        )
        if filename:
            self.app.video_path.set(filename)
            
    def browse_sequence(self):
        """Browse for an image sequence folder"""
        dirname = filedialog.askdirectory(
            title="Select Image Sequence Folder"
        )
        if dirname:
            self.app.video_path.set(dirname)

    def browse_mask(self):
        """Browse for mask image file"""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg"),
            ("All files", "*.*")
        ]
        filename = filedialog.askopenfilename(
            title="Select Mask Image",
            filetypes=filetypes
        )
        if filename:
            self.app.mask_path.set(filename)
            print(f"Selected mask: {filename}")
            
            # Check for keyframe metadata and update status
            self.update_keyframe_status(filename)

    def update_keyframe_status(self, mask_path):
        """Update the keyframe status display based on mask metadata"""
        try:
            from mask.mask_utils import get_keyframe_metadata_from_mask
            keyframe = get_keyframe_metadata_from_mask(mask_path)
            
            if keyframe is not None:
                self.app.keyframe_status.set(f"Frame {keyframe} being used")
                print(f"Mask contains keyframe metadata: frame {keyframe}")
            else:
                self.app.keyframe_status.set("")  # Clear the status
                
        except Exception as e:
            print(f"Error checking keyframe metadata: {e}")
            self.app.keyframe_status.set("")  # Clear the status on error

    def check_existing_mask_keyframe(self):
        """Check keyframe metadata for existing mask path (called on app startup)"""
        mask_path = self.app.mask_path.get()
        if mask_path and os.path.exists(mask_path):
            self.update_keyframe_status(mask_path)

    def generate_mask(self):
        """Open the SAM mask generator interface"""
        # Check if a video is selected
        if not self.app.video_path.get():
            messagebox.showerror("Error", "Please select an input video first")
            return
        
        # Make sure the video file exists
        if not os.path.exists(self.app.video_path.get()):
            messagebox.showerror("Error", "The selected video file does not exist")
            return
            
        # Check if input type is video (not image sequence)
        if self.app.input_type.get() != "video":
            messagebox.showerror("Error", "Mask generation is only available for video files, not image sequences")
            return
        
        try:
            # Create a callback function for when the mask is generated
            def on_mask_generated(mask_path):
                if mask_path and os.path.exists(mask_path):
                    self.app.mask_path.set(mask_path)
                    print(f"Mask generated and set: {mask_path}")
                    
                    # Update keyframe status for newly generated mask
                    self.update_keyframe_status(mask_path)
            
            # Determine mask save path
            video_name = os.path.basename(self.app.video_path.get())
            if video_name.endswith(('.mp4', '.mov', '.avi')):
                video_name = os.path.splitext(video_name)[0]
            
            mask_filename = f"{video_name}_sam_mask.png"
            mask_dir = os.path.dirname(self.app.video_path.get())
            mask_save_path = os.path.join(mask_dir, mask_filename)
            
            # Check if MaskGeneratorUI is available
            if HAS_MASK_GENERATOR:
                print("Opening SAM mask generator interface...")
                
                # Create the mask generator UI
                mask_generator = MaskGeneratorUI(self.app.root, on_mask_generated=on_mask_generated, main_app=self.app)
                
                # Open the mask generator window
                mask_generator.open_mask_generator(
                    video_path=self.app.video_path.get(),
                    mask_save_path=mask_save_path,
                    existing_mask_path=self.app.mask_path.get()
                )
            else:
                # Fall back to command-line version
                print("The SAM mask generator UI is not available.")
                print("Using command-line version instead.")
                
                if generate_mask_for_video:
                    result = generate_mask_for_video(
                        video_path=self.app.video_path.get(),
                        output_mask_path=mask_save_path
                    )
                    
                    if result:
                        self.app.mask_path.set(result)
                else:
                    messagebox.showerror("Error", "Mask generation functionality is not available")
        
        except ImportError as e:
            messagebox.showerror("Error", f"Missing required packages: {str(e)}\n\n"
                                "Please install the required packages:")
            print("Required packages:")
            print("  pip install torch torchvision")
            print("  pip install git+https://github.com/facebookresearch/segment-anything.git")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize mask generation: {str(e)}")
            import traceback
            traceback.print_exc()

    def browse_output(self):
        """Browse for output directory"""
        dirname = filedialog.askdirectory(
            title="Select Output Directory"
        )
        if dirname:
            self.app.output_path.set(dirname)
    
    def show_help(self):
        """Show help information"""
        help_text = """
MatAnyone Video Processor Help

Input Options:
- Video File: Select a video file (.mp4, .mov, .avi)
- Image Sequence: Select a folder containing image files

Mask:
- Select a binary mask image that identifies the object you want to extract
- Generate Mask: Opens the SAM mask generator to create a mask by selecting areas
- White pixels (255) indicate the foreground object
- Black pixels (0) indicate the background

Basic Controls:
- Warmup Frames: Number of frames to use for initialization
- Max Size: Maximum dimension for processing (smaller = faster but lower quality)
  Set to -1 for native resolution, though this may require more memory.
- Save Individual Frames: Save each frame as separate image files
- Cleanup Temporary Files: Removes intermediate files when complete

Video Quality Settings (NEW):
- Codec: Choose between H.264 (best compatibility), H.265 (better compression), VP9 (very efficient), or Auto
  * Auto: Automatically selects the best available codec with fallbacks
  * H.264: Widely supported, good quality, recommended for compatibility
  * H.265: Better compression than H.264, newer players required
  * VP9: Google's codec, very efficient but requires modern players
- Quality: Select from Low (fast/small), Medium (balanced), High (recommended), Very High (excellent), or Lossless
  * Low: 2 Mbps - Fast encoding, smallest files
  * Medium: 5 Mbps - Good balance of quality and file size
  * High: 8 Mbps - Recommended for most uses
  * Very High: 15 Mbps - Excellent quality, larger files
  * Lossless: 50 Mbps - Maximum quality, largest files
- Custom Bitrate: Override quality presets with a specific bitrate (1000-50000 kbps)
  * Higher bitrates = better quality but larger files
  * Enable the checkbox to use the slider for precise control

Mask Controls:
- Erosion Radius: Reduces the mask size (higher = smaller mask)
- Dilation Radius: Expands the mask size (higher = larger mask)
- Generate Mask: Create a mask using SAM from the input video

Advanced Controls:
- Number of Chunks: Split the video into N chunks for processing.
  This can help process high-resolution videos by breaking them into smaller parts.
  (Ignored when Auto-chunk mode is enabled)
- Chunk Type: Choose between horizontal strips or aspect-ratio preserving grid
  - Horizontal Strips: Traditional method, divides into horizontal bands
  - Aspect-ratio Grid: NEW! Divides video into rectangular chunks preserving aspect ratio
- Bidirectional Processing: Processes video in both directions for better results
- Reverse Dilation: Expands the final mask for reverse processing
- Blend Method:
  * Weighted: Combines passes with weights based on frame position
  * Max Alpha: Uses maximum alpha value from either pass
  * Min Alpha: Uses minimum alpha value from either pass
  * Average: Simple average of both passes

Enhanced Chunk Processing (Improved):
- Creates a continuous bidirectional mask for the entire video at reduced resolution
- Identifies optimal keyframes with maximum mask coverage for each chunk
- Processes only the frame ranges where objects are visible
- Uses bidirectional processing around optimal keyframes for each range
- Auto-chunk Mode: NEW! Creates chunks based on low-res dimensions for optimal sizing
  * When enabled, the 'Number of Chunks' input is disabled and calculated automatically
  * The system determines the ideal number and size of chunks for best results
- Low-res Scale: Scale factor for preprocessing (smaller = faster but less precise)
  * Options now include 0.125 (1/8), 0.25 (1/4), 0.5 (1/2), and 0.75 (3/4)
- Low-res Blend Method: Separate blending option just for the low-resolution mask
  * Often a different blend method works better for the low-res mask vs final output
  * For example, 'weighted' often works better for low-res mask even when 'max_alpha' is best for final output
- Mask Threshold: Percentage threshold for considering a frame to have meaningful mask content
  * Higher values (e.g., 20-30%) require more mask coverage for a region to be processed
  * Lower values (e.g., 5%) will process regions with minimal mask coverage
  * Use higher values to skip regions with small/insignificant content
- Prioritize Frames with Faces: Uses face detection to select optimal keyframes
  * When enabled, frames with faces are chosen as keyframes for better quality
  * Falls back to maximum mask coverage if no faces are detected
  * Particularly useful for videos with human subjects

Grid vs. Strips Chunking:
- Strip chunking divides the image into horizontal bands which works well for most videos
- Grid chunking divides the image into a grid pattern that preserves aspect ratio
- Grid chunking often produces better results for:
  * Videos with objects moving both horizontally and vertically
  * Scenes with multiple moving objects
  * Videos with complex camera movements
  * High-resolution landscape videos that would otherwise need many horizontal strips

Auto-chunking Mode:
- NEW! Automatically calculates the optimal number and size of chunks
- Each chunk will match the low-res dimensions with appropriate overlap
- Ensures chunks are properly sized for the model
- Can produce better results especially for high-resolution videos
- When enabled, the 'Number of Chunks' input is disabled and determined automatically

Video Quality & Codec Optimization:
- Choose appropriate codecs based on your needs:
  * H.264: Best compatibility - works on all devices and players
  * H.265: Better compression - 30-50% smaller files than H.264
  * VP9: Very efficient - excellent for web streaming, requires modern players
  * Auto: Tries H.264 first, then H.265, VP9, with mp4v fallback
- Quality presets automatically set optimal bitrates:
  * Use 'High' for most applications
  * Use 'Very High' for professional work
  * Use 'Lossless' only when file size is not a concern
- Custom bitrate for precise control:
  * 1000-3000 kbps: Good for web/mobile viewing
  * 5000-8000 kbps: Standard HD quality
  * 10000-15000 kbps: High-quality for editing/archival
  * 20000+ kbps: Professional/broadcast quality

Memory Management Strategies:
- For large videos that cause memory errors, try these options:
  1. Enable Enhanced Chunk Processing (most effective)
  2. Enable Auto-chunk Mode
  3. Try Grid chunking instead of Strips
  4. Increase the number of Chunks
  5. Reduce the Max Size to scale down the video
  6. Avoid using native resolution (-1)
  7. Try a different codec/quality combination (some use less memory)

Troubleshooting Chunk Errors:
- If you get tensor dimension errors with chunks:
  1. Enable Enhanced Chunk Processing with Auto-chunk Mode
  2. Try Grid chunking instead of Strips (or vice versa)
  3. Use a max size like 512 instead of native resolution (-1)
  4. Process without chunks (set Number of Chunks to 1)
- If you get video encoding errors:
  1. Try a different codec (Auto mode will find the best available)
  2. Reduce the quality preset or custom bitrate
  3. Make sure ffmpeg is properly installed on your system

Video Quality Troubleshooting:
- If videos won't open in your player:
  * Try H.264 codec for maximum compatibility
  * Some older players don't support H.265 or VP9
- If file sizes are too large:
  * Use H.265 or VP9 for better compression
  * Reduce quality preset or custom bitrate
  * Consider lower resolution (Max Size setting)
- If quality is poor:
  * Increase quality preset or custom bitrate
  * Use 'Very High' or 'Lossless' quality
  * Enable Enhanced Chunk Processing for better results

Controls:
- Process Video: Start processing (becomes Cancel during processing)

Mask Generation:
- The Generate Mask button opens SAM (Segment Anything Model) to create a mask
- Point Mode: Click to set foreground (green) points, right-click for background (red)
- Box Mode: Click to set box corners to define a region
- Paint Mode: Available after generating a mask to refine it with brush tools
  * Left-click adds to mask, right-click removes from mask
  * Adjust brush size using [ and ] keys or dropdown selector

Output:
- Foreground video (_fgr.mp4): RGB video with background removed
- Alpha matte video (_pha.mp4): Grayscale video of the alpha channel
- Individual frames saved in subfolders if option is enabled
- Bidirectional results have _bidirectional suffix
- Enhanced chunk results include "enhanced" in the filename
- All videos are now encoded with high-quality settings for better results

Keyframe Metadata System (NEW):
- Masks generated on frames other than 0 now store keyframe metadata
- When a mask with keyframe metadata is used, enhanced chunk processing uses a specialized approach:
  * Cuts the video at the keyframe as the pivot point
  * Processes forward from keyframe to end
  * Processes backward from keyframe to start
  * Recombines with perfect frame alignment to avoid sequence corruption
- Frame 0 masks continue to work exactly as before (no behavior change)
- The GUI will display "Frame [N] being used" under masks with keyframe metadata
- This results in better temporal consistency for masks generated on specific frames

New in this version:
- NEW! Keyframe metadata system for intelligent mask-based processing
- Improved auto-chunk mode to automatically calculate optimal number of chunks
- The Number of Chunks input is now disabled when Auto-chunk mode is enabled
- Each chunk will match the low-res dimensions with appropriate overlap
- Better handling of chunk sizing and overlap for optimal results
- Improved diagnostics showing the actual number of chunks created
- Advanced video quality controls with codec selection
- Quality presets (Low, Medium, High, Very High, Lossless)
- Custom bitrate control for precise quality settings
- Intelligent codec fallback system for maximum compatibility
- Significantly improved output video quality with reduced compression artifacts
- Modularized mask generation code for better organization and maintainability
- Enhanced mask editing tools with improved paint mode performance
- Fixed frame navigation to maintain consistent view and zoom level
- Resolved issues with frame resizing when scrubbing through timeline
- Improved "fit to window" mode that persists across frame changes
- Better handling of aspect ratio when changing frames
"""
        create_message_dialog(self.app.root, "Help", help_text, "Close")
    
    def show_about(self):
        """Show about information"""
        about_text = """
MatAnyone Video Processor GUI v1.1716998903

A graphical interface for the MatAnyone video segmentation and matting model with SAM mask generation.

MatAnyone is a deep learning model for video segmentation and matting.
This GUI allows easy processing of videos to extract foreground objects with transparency.

Major New Features (May 2025):
- Modularized Mask Generator: Split into smaller, more maintainable components
- Enhanced mask editing tools with improved performance
- Advanced Video Quality Controls: Choose from H.264, H.265, VP9 codecs or Auto mode
- Quality Presets: Low, Medium, High, Very High, and Lossless options
- Custom Bitrate Control: Precise bitrate settings from 1-50 Mbps
- Intelligent Codec Fallbacks: Automatic codec selection with compatibility fallbacks
- Significantly Improved Output Quality: Reduced compression artifacts and better color reproduction

Enhanced Chunk Processing Improvements:
- Refactored codebase into multiple modules for better maintainability
- Improved auto-chunk mode to automatically calculate optimal number of chunks
- Number of Chunks input is now disabled when Auto-chunk is enabled
- Each chunk matches the low-res dimensions with appropriate overlap
- Better handling of chunk sizing for optimal results
- Added 3/4 resolution option for low-res preprocessing
- Improved face detection with better mask coverage balancing
- Fixed optimal keyframe selection with both face detection and coverage
- Fixed bidirectional processing for cases where keyframe is first or last
- Better grid chunk layout with proper aspect ratio handling
- Improved weight mask handling with proper temp directory usage

Technical Improvements:
- All video output now uses high-quality encoding with proper codec support
- Better memory management and error handling
- Improved progress monitoring and user feedback
- Enhanced cross-platform compatibility

For more information about MatAnyone, visit:
https://github.com/PeiqingYang/MatAnyone

For more information about SAM (Segment Anything Model), visit:
https://segment-anything.com/

System Requirements:
- Python 3.8+ with required packages installed
- FFmpeg for video processing (automatically uses system installation)
- CUDA-capable GPU recommended for faster processing
- Sufficient RAM (8GB+ recommended for high-resolution videos)

Video Quality Notes:
- H.264: Best compatibility, supported by all players and devices
- H.265: Better compression (smaller files), requires newer players
- VP9: Google's codec, very efficient, good for web streaming
- Auto mode: Tries codecs in order of preference with automatic fallbacks
- Higher quality/bitrate settings produce better results but larger files
"""
        create_message_dialog(self.app.root, "About", about_text, "Close")
    
    def on_closing(self):
        """Handle window close event"""
        # Cancel processing if in progress
        if self.app.processing_manager.processing:
            if messagebox.askyesno("Quit", "Processing is in progress. Are you sure you want to quit?"):
                self.app.processing_manager.cancel_processing = True
                self.app.root.after(1000, self.force_quit)  # Force quit after 1 second if still processing
            else:
                return
            
        # Clear any existing progress monitor timer
        if self.app.processing_manager.progress_timer_id:
            self.app.root.after_cancel(self.app.processing_manager.progress_timer_id)
            self.app.processing_manager.progress_timer_id = None
        
        # Save settings before closing
        self.app.config_manager.save_settings(self.app)
        
        # Restore stdout
        sys.stdout = self.app.original_stdout
        
        # Destroy window
        self.app.root.destroy()
    
    def force_quit(self):
        """Force quit the application"""
        sys.stdout = self.app.original_stdout
        self.app.root.destroy()