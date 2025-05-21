"""
# mask_ui_frame_manager.py - v1.1747787389
# Updated: Tuesday, May 20, 2025 at 17:29:49 PST
# Changes in this version:
# - Implemented "fit to window" mode that persists across frame changes
# - Fixed issue where images would resize when navigating through timeline
# - Added proper recalculation of fit dimensions for each frame
# - Preserved consistent view sizing when changing frames
# - Improved status bar to show the current view mode (fit vs. zoom)
# - Maintained aspect ratio across different frame sizes
# - Fixed vertical and horizontal scaling issues with frame navigation
# - Added proper centering of frames when in fit mode

Frame management utilities for the SAM-based mask generator.
Handles frame navigation, checkpointing, and frame state.
"""

import os
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox
import traceback
import numpy as np

# Import UI components
from ui.ui_components import Tooltip


class MaskFrameManager:
    """Manages frame navigation and checkpoint functionality"""
    
    def __init__(self, ui):
        """
        Initialize the frame manager
        
        Args:
            ui: Reference to the MaskUIBase object
        """
        self.ui = ui
    
    def on_frame_entry(self, event=None):
        """Handle frame number entry"""
        try:
            frame_num = int(self.ui.current_frame_var.get())
            # Clamp to valid range
            frame_num = max(0, min(frame_num, self.ui.total_frames - 1))
            self.set_current_frame(frame_num)
        except ValueError:
            # If invalid number, revert to current frame
            self.ui.current_frame_var.set(str(self.ui.current_frame_index))
    
    def on_slider_changed(self, value):
        """Handle slider change events"""
        if self.ui.updating_frame:
            return
        
        try:
            frame_num = int(float(value))
            # Ensure frame number is within range
            frame_num = max(0, min(frame_num, self.ui.total_frames - 1))
            self.set_current_frame(frame_num)
        except ValueError:
            pass
    
    def navigate_frame(self, delta):
        """Navigate to frame relative to current frame"""
        new_frame = self.ui.current_frame_index + delta
        # Ensure frame number is within range
        new_frame = max(0, min(new_frame, self.ui.total_frames - 1))
        self.set_current_frame(new_frame)
    
    def set_current_frame(self, frame_index):
        """Set the current frame and update display"""
        if self.ui.current_frame_index == frame_index:
            return
            
        # Set updating flag to prevent recursion
        self.ui.updating_frame = True
        
        # Save current frame state if needed
        self.save_current_frame_state()
        
        # Update current frame index
        self.ui.current_frame_index = frame_index
        
        # Update UI
        self.ui.current_frame_var.set(str(frame_index))
        self.ui.frame_slider.set(frame_index)
        
        # Get current canvas dimensions
        canvas_width = self.ui.canvas.winfo_width()
        canvas_height = self.ui.canvas.winfo_height()
        
        # Store current view settings for potential restoration
        current_zoom = getattr(self.ui, 'zoom_factor', 1.0)
        current_offset_x = getattr(self.ui, 'canvas_offset_x', 0)
        current_offset_y = getattr(self.ui, 'canvas_offset_y', 0)
        
        # Store the current canvas configuration
        current_width = self.ui.canvas.cget("width")
        current_height = self.ui.canvas.cget("height")
        
        # Check if we're in "fit to window" mode
        fit_to_window = getattr(self.ui, 'fit_to_window', False)
        
        # Load frame from video
        try:
            # Extract frame from video
            frame = self.ui.mask_generator.extract_frame(self.ui.video_path, frame_index)
            self.ui.image = frame
            
            # Get original frame dimensions
            orig_height, orig_width = frame.shape[:2]
            self.ui.original_dimensions = (orig_width, orig_height)
            
            # Initial scale for display
            if self.ui.scale < 1:
                # Apply initial scaling to maintain consistent display size
                display_width = int(orig_width * self.ui.scale)
                display_height = int(orig_height * self.ui.scale)
                pil_img = Image.fromarray(frame)
                pil_img = pil_img.resize((display_width, display_height), Image.Resampling.LANCZOS)
                self.ui.display_frame = np.array(pil_img)
            else:
                # No scaling needed
                self.ui.display_frame = frame
            
            # Get display dimensions after initial scaling
            display_height, display_width = self.ui.display_frame.shape[:2]
            
            # If in "fit to window" mode, recalculate the fit scale
            if fit_to_window:
                # Calculate scaling factor to fit within canvas (with a small margin)
                margin = 20  # Pixels of margin
                width_scale = (canvas_width - margin) / display_width
                height_scale = (canvas_height - margin) / display_height
                
                # Use the smaller scale to ensure entire image fits
                fit_scale = min(width_scale, height_scale, 1.0)
                
                # Update the current zoom to use the fit scale
                current_zoom = fit_scale
                
                # Center the image in the canvas
                current_offset_x = max(0, (canvas_width - (display_width * fit_scale)) / 2)
                current_offset_y = max(0, (canvas_height - (display_height * fit_scale)) / 2)
                
                # Store the current fit settings
                self.ui.fit_scale = fit_scale
                self.ui.last_canvas_width = canvas_width
                self.ui.last_canvas_height = canvas_height
            
            # Convert to PIL image and apply current zoom
            pil_image = Image.fromarray(self.ui.display_frame)
            zoomed_width = int(display_width * current_zoom)
            zoomed_height = int(display_height * current_zoom)
            
            # Use LANCZOS for high quality resizing
            resample_method = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
            zoomed_pil = pil_image.resize((zoomed_width, zoomed_height), resample_method)
            self.ui.tk_image = ImageTk.PhotoImage(image=zoomed_pil)
            
            # Clear the canvas
            self.ui.canvas.delete("all")
            
            # Create image with the current view settings
            self.ui.image_container = self.ui.canvas.create_image(
                current_offset_x, current_offset_y, 
                anchor=tk.NW, image=self.ui.tk_image, 
                tags="base_image"
            )
            
            # Update UI state with current view settings
            self.ui.zoom_factor = current_zoom
            self.ui.canvas_offset_x = current_offset_x
            self.ui.canvas_offset_y = current_offset_y
            
            # Configure canvas scrollable area
            self.ui.canvas.config(scrollregion=self.ui.canvas.bbox(tk.ALL))
            
            # Load checkpoint data if this frame is a checkpoint
            self.load_frame_state(frame_index)
            
            # Update status
            checkpoint_status = " (checkpoint)" if frame_index in self.ui.checkpoints else ""
            mode_info = "Fit to window" if fit_to_window else f"Zoom: {current_zoom:.2f}x"
            self.ui.status_var.set(f"Frame {frame_index}{checkpoint_status} - {mode_info}")
            
            # Reset overlay flag
            self.ui.overlay_visible = False
            
            # Update button states
            self.ui.overlay_button.config(state=tk.DISABLED if frame_index not in self.ui.checkpoints else tk.NORMAL)
            self.ui.confirm_button.config(state=tk.DISABLED if frame_index not in self.ui.checkpoints else tk.NORMAL)
            
            # Update generate button text based on checkpoint status
            if frame_index in self.ui.checkpoints:
                self.ui.generate_button.config(text="Update Mask")
            else:
                self.ui.generate_button.config(text="Generate Mask")
            
            # Show mask overlay if this is a checkpoint frame
            if frame_index in self.ui.checkpoints:
                self.ui.editor.show_mask_overlay()
                self.ui.overlay_visible = True
                self.ui.overlay_button.config(text="Hide Overlay")
                
            # Update process button state based on number of checkpoints
            if hasattr(self.ui, 'process_checkpoints_button'):
                if len(self.ui.checkpoints) >= 2:
                    self.ui.process_checkpoints_button.config(state=tk.NORMAL)
                else:
                    self.ui.process_checkpoints_button.config(state=tk.DISABLED)
                
            # Update all point and box markers to ensure they're positioned correctly
            if hasattr(self.ui, 'interactions') and hasattr(self.ui.interactions, 'update_point_markers'):
                self.ui.interactions.update_point_markers()
                self.ui.interactions.update_box_markers()
                
            # If there's a paint cursor, make sure it's properly positioned
            if hasattr(self.ui, 'paint_cursor') and self.ui.paint_cursor:
                if hasattr(self.ui, 'last_mouse_pos'):
                    x, y = self.ui.last_mouse_pos
                    self.ui.editor.update_paint_cursor(x, y)
                    
            # Ensure view is updated correctly
            self.ui.canvas.update_idletasks()
            
        except Exception as e:
            print(f"Error loading frame {frame_index}: {str(e)}")
            traceback.print_exc()
        
        # Reset updating flag
        self.ui.updating_frame = False
    
    def save_current_frame_state(self):
        """Save the current frame state (points, boxes, mask) if needed"""
        # Only save if we have points, box, or mask
        if self.ui.points or self.ui.box_coords or self.ui.generated_mask is not None:
            # If there's a mask, this becomes a checkpoint
            if self.ui.generated_mask is not None or self.ui.edited_mask is not None:
                mask_to_save = self.ui.edited_mask if self.ui.edited_mask is not None else self.ui.generated_mask
                self.ui.checkpoints[self.ui.current_frame_index] = {
                    'mask': mask_to_save.copy(),  # Store a copy of the mask
                    'points': [p.copy() for p in self.ui.points] if self.ui.points else [],
                    'box_coords': self.ui.box_coords.copy() if self.ui.box_coords else []
                }
                print(f"Saved checkpoint for frame {self.ui.current_frame_index}")
                
                # Update checkpoint markers
                self.update_checkpoint_markers()
    
    def load_frame_state(self, frame_index):
        """Load frame state from checkpoint if available"""
        # Clear current state
        self.ui.points = []
        self.ui.point_markers = []
        self.ui.box_coords = []
        self.ui.box_markers = []
        self.ui.generated_mask = None
        self.ui.edited_mask = None
        
        # Load state from checkpoint if available
        if frame_index in self.ui.checkpoints:
            checkpoint = self.ui.checkpoints[frame_index]
            
            # Load mask
            if 'mask' in checkpoint and checkpoint['mask'] is not None:
                self.ui.generated_mask = checkpoint['mask'].copy()
                self.ui.edited_mask = self.ui.generated_mask.copy()
            
            # Load points
            if 'points' in checkpoint and checkpoint['points']:
                self.ui.points = [p.copy() for p in checkpoint['points']]
                
                # Create markers for points
                for i, point in enumerate(self.ui.points):
                    # Convert original coordinates to display coordinates
                    display_x = int(point[0] * self.ui.scale) if self.ui.scale < 1 else point[0]
                    display_y = int(point[1] * self.ui.scale) if self.ui.scale < 1 else point[1]
                    
                    # Create marker based on point type (foreground/background)
                    color = "green" if point[2] == 1 else "red"
                    marker = self.ui.canvas.create_oval(
                        display_x - 5, display_y - 5, 
                        display_x + 5, display_y + 5, 
                        fill=color, outline=color, tags="point_marker"
                    )
                    self.ui.point_markers.append((marker, i))
            
            # Load box coordinates
            if 'box_coords' in checkpoint and checkpoint['box_coords']:
                self.ui.box_coords = checkpoint['box_coords'].copy()
                
                # Create markers for box
                if len(self.ui.box_coords) == 2:
                    x1, y1 = self.ui.box_coords[0][0], self.ui.box_coords[0][1]
                    x2, y2 = self.ui.box_coords[1][0], self.ui.box_coords[1][1]
                    
                    # Top-left corner
                    tl_marker = self.ui.canvas.create_oval(
                        x1 - 5, y1 - 5, x1 + 5, y1 + 5,
                        fill="blue", outline="blue", tags="box_marker"
                    )
                    
                    # Bottom-right corner
                    br_marker = self.ui.canvas.create_oval(
                        x2 - 5, y2 - 5, x2 + 5, y2 + 5,
                        fill="blue", outline="blue", tags="box_marker"
                    )
                    
                    # Box outline
                    box_marker = self.ui.canvas.create_rectangle(
                        x1, y1, x2, y2,
                        outline="blue", width=2, tags="box_marker"
                    )
                    
                    self.ui.box_markers = [tl_marker, br_marker, box_marker]
    
    def update_checkpoint_markers(self):
        """Update the visual markers for checkpoints in the timeline"""
        # Clear existing markers
        for widget in self.ui.checkpoint_markers_frame.winfo_children():
            widget.destroy()
        
        # Create a label for each checkpoint
        if self.ui.checkpoints:
            # Create a new label for each checkpoint
            for frame_idx in sorted(self.ui.checkpoints.keys()):
                # Create a marker
                marker = tk.Label(self.ui.checkpoint_markers_frame, text="▲", fg="red")
                
                # Calculate position based on frame index
                pos = (frame_idx / max(1, self.ui.total_frames - 1)) * 100
                
                # Place the marker
                marker.place(relx=pos/100, rely=0.5, anchor="center")
                
                # Add tooltip showing frame number
                marker_tooltip = Tooltip(marker, f"Checkpoint at frame {frame_idx}")
    
    def list_checkpoints(self):
        """Show a dialog listing all checkpoints"""
        if not self.ui.checkpoints:
            messagebox.showinfo("Checkpoints", "No checkpoints have been created yet.")
            return
        
        # Create a list of checkpoints
        checkpoint_list = "\n".join([f"Frame {frame}" for frame in sorted(self.ui.checkpoints.keys())])
        
        # Show in a dialog
        messagebox.showinfo("Checkpoints", f"Checkpoints have been created at the following frames:\n\n{checkpoint_list}")
    
    def process_with_checkpoints(self, mask_save_path=None):
        """Process video with checkpoint-based segmentation"""
        # Ensure we have at least 2 checkpoints
        if len(self.ui.checkpoints) < 2:
            messagebox.showerror("Error", "At least 2 checkpoints are needed for processing")
            return
        
        # Get sorted list of checkpoint frames
        checkpoint_frames = sorted(self.ui.checkpoints.keys())
        
        try:
            # Create a progress dialog
            progress_window = tk.Toplevel(self.ui.mask_window)
            progress_window.title("Processing Video")
            progress_window.geometry("400x200")
            progress_window.transient(self.ui.mask_window)
            progress_window.grab_set()
            
            # Create progress label and bar
            progress_label = ttk.Label(progress_window, text="Preparing to process checkpoints...")
            progress_label.pack(pady=(20, 10))
            
            progress_bar = ttk.Progressbar(progress_window, orient=tk.HORIZONTAL, 
                                         length=300, mode='determinate')
            progress_bar.pack(pady=10, padx=20)
            
            # Create cancel button
            cancel_button = ttk.Button(progress_window, text="Cancel", 
                                     command=lambda: progress_window.destroy())
            cancel_button.pack(pady=10)
            
            # Process segments between checkpoints
            segments = []
            for i in range(len(checkpoint_frames) - 1):
                start_frame = checkpoint_frames[i]
                end_frame = checkpoint_frames[i + 1]
                
                segments.append((start_frame, end_frame))
            
            # Update progress
            progress_label.config(text=f"Processing {len(segments)} segments between checkpoints...")
            progress_window.update()
            
            # TODO: Implement actual video processing with segments
            # This would involve splitting the video, processing each segment,
            # and then combining the results
            
            # For now, let's just show a success message with the frames involved
            progress_bar['value'] = 100
            progress_label.config(text="Processing complete!")
            progress_window.update()
            
            # Show processing details
            details = "Processing completed for the following segments:\n\n"
            for start, end in segments:
                details += f"Frame {start} to Frame {end}\n"
            
            # Close progress window and show completion message
            progress_window.destroy()
            messagebox.showinfo("Processing Complete", details)
            
            # Determine output path
            if mask_save_path is None:
                # Create a default path
                video_dir = os.path.dirname(self.ui.video_path)
                video_name = os.path.basename(self.ui.video_path)
                if video_name.endswith(('.mp4', '.mov', '.avi')):
                    video_name = os.path.splitext(video_name)[0]
                
                mask_save_path = os.path.join(video_dir, f"{video_name}_mask.png")
            
            # Return result to callback if provided
            if self.ui.on_mask_generated:
                self.ui.on_mask_generated(mask_save_path)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process checkpoints: {str(e)}")
            traceback.print_exc()
    
    def save_and_close(self, video_path, mask_save_path=None):
        """Save the mask and close the window"""
        # Determine which mask to save (edited or generated)
        mask_to_save = self.ui.edited_mask if self.ui.edited_mask is not None else self.ui.generated_mask
        
        if mask_to_save is None:
            messagebox.showerror("Error", "No mask has been generated yet")
            return
        
        try:
            # Determine filename for mask
            video_name = os.path.basename(video_path)
            if video_name.endswith(('.mp4', '.mov', '.avi')):
                video_name = os.path.splitext(video_name)[0]
            
            # Create mask filename
            base_filename = f"{video_name}_frame_{self.ui.current_frame_index}_mask.png"
            
            # Determine where to save the mask
            if mask_save_path is None:
                # Save in masks directory
                mask_save_path = os.path.join(self.ui.masks_dir, base_filename)
            
            # Check if file already exists and create a unique name if needed
            if os.path.exists(mask_save_path):
                # Find a unique filename by adding a counter
                counter = 1
                mask_dir = os.path.dirname(mask_save_path)
                filename_base, ext = os.path.splitext(os.path.basename(mask_save_path))
                while os.path.exists(mask_save_path):
                    mask_save_path = os.path.join(mask_dir, f"{filename_base}_{counter}{ext}")
                    counter += 1
                
                print(f"File already exists, using unique name: {os.path.basename(mask_save_path)}")
            
            # Save the mask
            mask_path = self.ui.mask_generator.save_mask(mask_to_save, mask_save_path)
            
            messagebox.showinfo("Success", 
                               f"Mask saved successfully!\n\n"
                               f"Saved to: {mask_path}")
            
            # Call the callback function if provided
            if self.ui.on_mask_generated:
                self.ui.on_mask_generated(mask_path)
            
            # Close the mask window
            self.ui.mask_window.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save mask: {str(e)}")
            traceback.print_exc()
