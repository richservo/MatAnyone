"""
# mask_generator_ui_main.py - v1.1716998903
# Created: Tuesday, May 21, 2025 at 17:15:03 PST
# Changes in this version:
# - Created as the main entry point for the modularized mask generator UI
# - Imports functionality from other mask generator modules
# - Maintains the same public API as the original mask_generator_ui.py
# - Coordinates between the base UI, interactions, frame manager, and editor

Main UI class for SAM-based mask generation.
Provides a user interface for generating masks using SAM.
"""

import numpy as np
from PIL import Image

# Import modular components
from ui.mask_ui_base import MaskUIBase
from mask.sam_generator import SAMMaskGenerator


class MaskGeneratorUI(MaskUIBase):
    """User interface for generating masks using SAM"""
    
    def __init__(self, root, on_mask_generated=None, main_app=None):
        """
        Initialize the mask generator UI
        
        Args:
            root: Tkinter root or Toplevel window
            on_mask_generated: Callback function for when mask is generated
            main_app: Reference to the main application for config access
        """
        # Initialize the base UI
        super().__init__(root, on_mask_generated)
        
        # Store reference to main app for config access
        self.main_app = main_app
        
        # Create the SAM mask generator
        self.mask_generator = SAMMaskGenerator()
    
    def open_mask_generator(self, video_path, mask_save_path=None, existing_mask_path=None):
        """
        Open the mask generator UI
        
        Args:
            video_path: Path to the video file
            mask_save_path: Path to save the generated mask (optional)
            existing_mask_path: Path to existing mask from main GUI (optional)
        """
        try:
            # Store paths
            self.video_path = video_path
            self.existing_mask_path = existing_mask_path
            
            # Get video information
            self.video_info = self.mask_generator.get_video_info(video_path)
            self.total_frames = self.video_info['frame_count']
            
            # Reset current frame and checkpoints
            self.current_frame_index = 0
            self.checkpoints = {}
            
            # Extract the first frame from the video
            print("Extracting first frame from video...")
            first_frame = self.mask_generator.extract_frame(video_path, self.current_frame_index)
            self.image = first_frame
            self.display_frame = first_frame  # Will be resized if needed during UI initialization
            
            # Initialize the UI components
            self.initialize_ui(video_path, mask_save_path)
            
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Error", f"Failed to initialize mask generation: {str(e)}")
            import traceback
            traceback.print_exc()
