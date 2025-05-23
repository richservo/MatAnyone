"""
# mask_ui_editor.py - v1.1716998907
# Updated: Monday, May 20, 2025 at 14:39:15 PST
# Changes in this version:
# - Optimized apply_paint_stroke function with adaptive debouncing based on zoom level
# - Replaced pixel-by-pixel processing with NumPy vectorized operations for faster performance
# - Added caching for brush masks to improve painting performance when zoomed in
# - Optimized paint cursor updates to reduce canvas operations
# - Improved performance with larger brush sizes

Mask editing tools for the SAM-based mask generator.
Handles mask generation, visualization, and editing operations.
"""

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import traceback
import copy


class MaskEditor:
    """Handles mask generation and editing"""
    
    def __init__(self, ui):
        """
        Initialize the mask editor
        
        Args:
            ui: Reference to the MaskUIBase object
        """
        self.ui = ui
        self.current_paint_action = None
        # Cache for brush masks at different radii
        self.brush_mask_cache = {}
        # Last cursor position and radius for optimized updates
        self._last_cursor_radius = None
        self._last_cursor_x = None
        self._last_cursor_y = None
    
    def generate_and_preview_mask(self):
        """Generate the mask and show preview overlay for the current frame"""
        if not self.ui.points and not self.ui.box_coords:
            from tkinter import messagebox
            messagebox.showerror("Error", "Please select points, define a box, or paint areas first")
            return
            
        try:
            # Show loading status
            self.ui.status_var.set("Loading SAM model... This may take a moment.")
            self.ui.mask_window.update()
            
            # Generate the mask based on the points and box
            try:
                points_np = np.array(self.ui.points) if self.ui.points else None
                
                # Set up box if present
                box = None
                if self.ui.box_coords and len(self.ui.box_coords) == 2:
                    # Get original coordinates from box_coords
                    # box_coords structure: [(display_x1, display_y1, orig_x1, orig_y1), (display_x2, display_y2, orig_x2, orig_y2)]
                    orig_x1 = self.ui.box_coords[0][2]
                    orig_y1 = self.ui.box_coords[0][3]
                    orig_x2 = self.ui.box_coords[1][2]
                    orig_y2 = self.ui.box_coords[1][3]
                    
                    # Ensure we have the min/max values for x and y to create a proper box
                    x1 = min(orig_x1, orig_x2)
                    y1 = min(orig_y1, orig_y2)
                    x2 = max(orig_x1, orig_x2)
                    y2 = max(orig_y1, orig_y2)
                    box = [x1, y1, x2, y2]
                
                # Keep the current edited mask for paint persistence
                previous_edited_mask = self.ui.edited_mask
                
                # Generate mask - use both points and box together for better results
                mask, score = self.ui.mask_generator.generate_mask_from_image(
                    self.ui.image, points=points_np, box=box, multimask_output=True
                )
                print(f"Generated mask with confidence {score:.4f}")
                
                # Store the generated mask
                self.ui.generated_mask = mask
                
                # If we had a previous edited mask, combine it with the new generated mask
                if previous_edited_mask is not None:
                    # Use previous edited mask as starting point
                    self.ui.edited_mask = previous_edited_mask.copy()
                    
                    # Apply the new generated mask where appropriate
                    # Strategy: Use a weighted combination of the previous edit and new generation
                    # We favor the previous edits more (they were intentional)
                    # Get a binary mask where the new mask is different from the previous edited mask
                    diff_mask = (self.ui.generated_mask != previous_edited_mask)
                    
                    # Where the new mask has content (1s) and we didn't previously edit those pixels,
                    # use the new mask value
                    # This preserves previous edits while allowing the new mask to add new content
                    self.ui.edited_mask[diff_mask & self.ui.generated_mask] = True
                else:
                    # Initialize edited mask with the generated mask if no previous edits
                    self.ui.edited_mask = mask.copy()
                
            except Exception as e:
                print(f"Error during mask generation: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
            
            # Store the mask score
            self.ui.mask_score = score
            
            # Create the overlay image and display it
            self.show_mask_overlay()
            
            # Ensure the overlay_visible flag is set and buttons updated
            self.ui.overlay_visible = True
            
            # Enable overlay toggle and confirm buttons
            self.ui.overlay_button.config(state=tk.NORMAL, text="Hide Overlay")
            self.ui.confirm_button.config(state=tk.NORMAL)
            
            # Save as checkpoint
            self.ui.checkpoints[self.ui.current_frame_index] = {
                'mask': mask.copy(),  # Store a copy of the mask
                'points': [p.copy() for p in self.ui.points] if self.ui.points else [],
                'box_coords': self.ui.box_coords.copy() if self.ui.box_coords else [],
                'edited_mask': self.ui.edited_mask.copy() if self.ui.edited_mask is not None else None
            }
            print(f"Created checkpoint for frame {self.ui.current_frame_index}")
            
            # Update checkpoint markers
            self.ui.frame_manager.update_checkpoint_markers()
            
            # Mark this frame as having a generated mask (for red highlighting)
            if hasattr(self.ui, 'mark_frame_with_mask'):
                self.ui.mark_frame_with_mask(self.ui.current_frame_index)
            
            # Update generate button text
            self.ui.generate_button.config(text="Update Mask")
            
            # Update status
            self.ui.status_var.set(f"Mask generated with confidence {score:.4f} and saved as checkpoint.")
            
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Error", f"Failed to generate mask: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def show_mask_overlay(self):
        """Show the mask as a red overlay on the image"""
        try:
            # Determine which mask to use
            mask_to_use = self.ui.edited_mask if self.ui.edited_mask is not None else self.ui.generated_mask
            
            if mask_to_use is None:
                return
            
            # Create a copy of the display frame
            overlay_img = self.ui.display_frame.copy()
            
            # Resize the mask to match the display frame if needed
            if self.ui.scale < 1:
                # Resize the boolean mask to match display dimensions
                display_h, display_w = self.ui.display_frame.shape[:2]
                mask_img = Image.fromarray(mask_to_use.astype(np.uint8) * 255)
                mask_img = mask_img.resize((display_w, display_h), Image.Resampling.NEAREST)
                resized_mask = np.array(mask_img) > 0
            else:
                resized_mask = mask_to_use
            
            # Apply red overlay at 50% opacity where mask is True
            overlay_img[resized_mask] = overlay_img[resized_mask] * 0.5 + np.array([255, 0, 0]) * 0.5
            
            # Convert to PhotoImage and update canvas
            pil_image = Image.fromarray(overlay_img.astype(np.uint8))
            
            # Apply current zoom factor
            if self.ui.zoom_factor != 1.0:
                zoomed_width = int(pil_image.width * self.ui.zoom_factor)
                zoomed_height = int(pil_image.height * self.ui.zoom_factor)
                pil_image = pil_image.resize((zoomed_width, zoomed_height), Image.Resampling.LANCZOS)
            
            self.ui.overlay_image = ImageTk.PhotoImage(image=pil_image)
            
            # Update image without removing markers
            # First, remove only the base image
            self.ui.canvas.delete("base_image")
            # Then add new overlay image with tag for future identification
            self.ui.image_container = self.ui.canvas.create_image(
                self.ui.canvas_offset_x, self.ui.canvas_offset_y, 
                anchor=tk.NW, image=self.ui.overlay_image, tags="base_image"
            )
            
            # Make sure markers remain on top
            for marker, _ in self.ui.point_markers:
                self.ui.canvas.tag_raise(marker)
            for marker in self.ui.box_markers:
                self.ui.canvas.tag_raise(marker)
                
            # Make sure paint cursor remains on top if present
            if hasattr(self.ui, 'paint_cursor') and self.ui.paint_cursor:
                self.ui.canvas.tag_raise(self.ui.paint_cursor)
                
            # Set overlay visible flag and update button text if it exists
            self.ui.overlay_visible = True
            if hasattr(self.ui, 'overlay_button'):
                self.ui.overlay_button.config(text="Hide Overlay")
            
        except Exception as e:
            print(f"Error showing overlay: {str(e)}")
            traceback.print_exc()
    
    def hide_mask_overlay(self):
        """Hide the overlay and show the original image"""
        try:
            # Create a zoomed version of the original display frame
            pil_image = Image.fromarray(self.ui.display_frame)
            
            # Apply current zoom factor
            if self.ui.zoom_factor != 1.0:
                zoomed_width = int(pil_image.width * self.ui.zoom_factor)
                zoomed_height = int(pil_image.height * self.ui.zoom_factor)
                pil_image = pil_image.resize((zoomed_width, zoomed_height), Image.Resampling.LANCZOS)
            
            self.ui.tk_image = ImageTk.PhotoImage(image=pil_image)
            
            # Show the original display frame
            self.ui.canvas.delete("base_image")  # Remove just the base image
            self.ui.image_container = self.ui.canvas.create_image(
                self.ui.canvas_offset_x, self.ui.canvas_offset_y, 
                anchor=tk.NW, image=self.ui.tk_image, tags="base_image"
            )
            
            # Make sure points and boxes stay on top
            for marker, _ in self.ui.point_markers:
                self.ui.canvas.tag_raise(marker)
            for marker in self.ui.box_markers:
                self.ui.canvas.tag_raise(marker)
            
            # Make sure paint cursor remains on top if present
            if hasattr(self.ui, 'paint_cursor') and self.ui.paint_cursor:
                self.ui.canvas.tag_raise(self.ui.paint_cursor)
                
            self.ui.overlay_visible = False
        except Exception as e:
            print(f"Error hiding overlay: {str(e)}")
            traceback.print_exc()
    
    def toggle_overlay(self):
        """Toggle between showing and hiding the mask overlay"""
        if hasattr(self.ui, 'overlay_visible') and self.ui.overlay_visible:
            self.hide_mask_overlay()
            self.ui.overlay_button.config(text="Show Overlay")
        else:
            self.show_mask_overlay()
            self.ui.overlay_button.config(text="Hide Overlay")
    
    def clear_selections(self):
        """Clear all selections (points, box, mask) for the current frame"""
        # Add to command stack before clearing
        if self.ui.points or self.ui.box_coords or self.ui.edited_mask is not None:
            self.add_to_command_stack(
                "clear_all", 
                {
                    "points": [p.copy() for p in self.ui.points] if self.ui.points else [],
                    "box_coords": self.ui.box_coords.copy() if self.ui.box_coords else [],
                    "edited_mask": self.ui.edited_mask.copy() if self.ui.edited_mask is not None else None,
                    "generated_mask": self.ui.generated_mask.copy() if self.ui.generated_mask is not None else None
                }
            )
        
        # Remove all point markers from canvas
        for marker, _ in self.ui.point_markers:
            self.ui.canvas.delete(marker)
        
        # Remove all box markers from canvas
        for marker in self.ui.box_markers:
            self.ui.canvas.delete(marker)
            
        # Clear any paint cursor
        if hasattr(self.ui, 'paint_cursor') and self.ui.paint_cursor:
            self.ui.canvas.delete(self.ui.paint_cursor)
            self.ui.paint_cursor = None
            
        # Clear temporary box if exists
        if self.ui.temp_box:
            self.ui.canvas.delete(self.ui.temp_box)
            self.ui.temp_box = None
        
        # Clear data for current frame
        self.ui.points = []
        self.ui.point_markers = []
        self.ui.box_coords = []
        self.ui.box_markers = []
        self.ui.drawing_box = False
        
        # Clear generated mask and edited mask
        self.ui.generated_mask = None
        self.ui.edited_mask = None
        
        # If this was a checkpoint, remove it
        if self.ui.current_frame_index in self.ui.checkpoints:
            del self.ui.checkpoints[self.ui.current_frame_index]
            # Update checkpoints
            self.ui.frame_manager.update_checkpoint_markers()
        
        # Reset UI buttons
        self.ui.overlay_button.config(state=tk.DISABLED)
        self.ui.confirm_button.config(state=tk.DISABLED)
        self.ui.generate_button.config(text="Generate Mask")
        
        # Switch to point mode
        self.ui.interaction_mode.set("point")
        self.ui.interactions.switch_mode()
        
        # Hide any mask overlay
        if self.ui.overlay_visible:
            self.hide_mask_overlay()
            self.ui.overlay_visible = False
        
        self.ui.status_var.set("Selections cleared. Start again.")
    
    def delete_point(self, point_index):
        """
        Delete a point by its index
        
        Args:
            point_index: Index of the point to delete
        """
        try:
            # Find the canvas marker for this point
            marker_to_delete = None
            marker_idx_to_delete = None
            
            for i, (marker, idx) in enumerate(self.ui.point_markers):
                if idx == point_index:
                    marker_to_delete = marker
                    marker_idx_to_delete = i
                    break
            
            # Delete the point if found
            if marker_to_delete is not None:
                # Remove from canvas
                self.ui.canvas.delete(marker_to_delete)
                
                # Remove from our lists
                self.ui.point_markers.pop(marker_idx_to_delete)
                point_coords = self.ui.points.pop(point_index)
                
                # Update indices for remaining markers
                for i in range(len(self.ui.point_markers)):
                    marker, idx = self.ui.point_markers[i]
                    if idx > point_index:
                        self.ui.point_markers[i] = (marker, idx - 1)
                
                # Update status message
                point_type = "foreground" if point_coords[2] == 1 else "background"
                self.ui.status_var.set(f"Deleted {point_type} point at ({point_coords[0]}, {point_coords[1]}). "
                                   f"{len(self.ui.points)} points remaining.")
        except Exception as e:
            print(f"Error deleting point: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def apply_paint_stroke(self, x, y, add_to_mask=True):
        """
        Apply paint stroke to edit the mask
        
        Args:
            x, y: Screen coordinates (directly from mouse events)
            add_to_mask: True to add to mask (foreground), False to remove (background)
        """
        # Make sure we have a mask to edit
        if self.ui.edited_mask is None:
            if self.ui.generated_mask is None:
                self.ui.status_var.set("No mask to edit. Please generate a mask first.")
                return
            # Initialize edited mask with the generated mask
            self.ui.edited_mask = self.ui.generated_mask.copy()
            
            # Start a new paint action for undo
            if self.current_paint_action is None:
                self.start_paint_action()
        
        # First convert screen coordinates (from mouse events) to display coordinates
        display_x, display_y = self.ui.interactions.screen_to_display(x, y)
        
        # Then convert display coordinates to original image coordinates
        orig_x, orig_y = self.ui.interactions.display_to_original(display_x, display_y)
        
        # Scale radius for original image coordinates
        orig_radius = self.ui.paint_radius
        if self.ui.scale < 1:
            orig_radius = int(self.ui.paint_radius / self.ui.scale)
        
        # Store this paint action for batch processing
        self.ui.accumulated_paint_actions.append((orig_x, orig_y, orig_radius, add_to_mask))
        
        # Adjust debounce time based on zoom - more zoomed in = more debounce
        debounce_time = self.ui.overlay_update_debounce
        if self.ui.zoom_factor > 2.0:
            # Scale debounce time with zoom factor for smoother experience when zoomed in
            debounce_time = int(self.ui.overlay_update_debounce * (self.ui.zoom_factor / 2.0))
            
        # Check if we should update now or wait
        current_time = self.ui.canvas.winfo_id()  # Use as a unique timestamp for each call
        elapsed = current_time - self.ui.last_overlay_update
        
        # If it's time to update or we're not painting (single click)
        if not self.ui.is_painting or elapsed > debounce_time:
            self.process_accumulated_paint_actions()
            self.ui.last_overlay_update = current_time
        else:
            # Schedule a deferred update if one isn't already pending
            if not self.ui.update_pending:
                self.ui.update_pending = True
                self.ui.canvas.after(debounce_time, self.process_accumulated_paint_actions)
    
    def start_paint_action(self):
        """Start a new paint action for undo tracking"""
        # Store the current state of the mask before any edits
        if self.ui.edited_mask is not None:
            self.current_paint_action = {
                "type": "paint",
                "before": self.ui.edited_mask.copy()
            }
        elif self.ui.generated_mask is not None:
            self.current_paint_action = {
                "type": "paint",
                "before": self.ui.generated_mask.copy()
            }
    
    def end_paint_action(self):
        """End the current paint action and add it to the command stack"""
        if self.current_paint_action is not None:
            # Store the state after edits
            self.current_paint_action["after"] = self.ui.edited_mask.copy()
            
            # Add to command stack
            self.add_to_command_stack("paint", self.current_paint_action)
            
            # Reset current paint action
            self.current_paint_action = None
    
    def get_brush_mask(self, radius):
        """
        Get a circular brush mask from cache or create a new one
        
        Args:
            radius: Brush radius
            
        Returns:
            Boolean mask of the brush shape
        """
        # Round radius to nearest integer to improve cache hits
        radius = int(radius)
        
        # Return from cache if available
        if radius in self.brush_mask_cache:
            return self.brush_mask_cache[radius]
            
        # Create a new brush mask
        size = 2 * radius + 1
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x*x + y*y <= radius*radius
        
        # Store in cache
        self.brush_mask_cache[radius] = mask
        
        return mask
    
    def process_accumulated_paint_actions(self):
        """Process all accumulated paint actions at once for efficiency"""
        if not self.ui.accumulated_paint_actions:
            self.ui.update_pending = False
            return
            
        # Start a new paint action if we haven't already
        if self.current_paint_action is None:
            self.start_paint_action()
            
        # Get mask dimensions
        h, w = self.ui.edited_mask.shape
        
        # Create a temporary mask for all add operations and another for all remove operations
        add_mask = np.zeros((h, w), dtype=bool)
        remove_mask = np.zeros((h, w), dtype=bool)
        
        # Apply all paint actions in batch using NumPy operations
        for orig_x, orig_y, orig_radius, add_to_mask in self.ui.accumulated_paint_actions:
            # Get integer coordinates
            center_x, center_y = int(orig_x), int(orig_y)
            radius = int(orig_radius)
            
            # Get the brush mask
            brush = self.get_brush_mask(radius)
            brush_h, brush_w = brush.shape
            
            # Calculate the position to place the brush in the mask
            y_min = max(0, center_y - radius)
            y_max = min(h, center_y + radius + 1)
            x_min = max(0, center_x - radius)
            x_max = min(w, center_x + radius + 1)
            
            # Skip if brush is completely outside the mask
            if y_max <= y_min or x_max <= x_min:
                continue
                
            # Calculate the portion of the brush to use
            brush_y_min = max(0, radius - center_y)
            brush_y_max = min(brush_h, brush_h - (center_y + radius + 1 - h))
            brush_x_min = max(0, radius - center_x)
            brush_x_max = min(brush_w, brush_w - (center_x + radius + 1 - w))
            
            # Place brush in appropriate mask
            if add_to_mask:
                add_mask[y_min:y_max, x_min:x_max] |= brush[brush_y_min:brush_y_max, brush_x_min:brush_x_max]
            else:
                remove_mask[y_min:y_max, x_min:x_max] |= brush[brush_y_min:brush_y_max, brush_x_min:brush_x_max]
                
        # Apply changes to the mask
        if np.any(add_mask):
            self.ui.edited_mask |= add_mask
        if np.any(remove_mask):
            self.ui.edited_mask &= ~remove_mask
        
        # Clear the accumulated actions
        self.ui.accumulated_paint_actions = []
        
        # Update the mask visualization
        self.update_mask_overlay()
        
        # Reset the pending flag
        self.ui.update_pending = False
    
    def update_paint_cursor(self, x, y):
        """Update the paint cursor position and size with caching for better performance"""
        # Cache the current paint cursor size to avoid unnecessary redraws
        current_radius = getattr(self, '_last_cursor_radius', None)
        current_x = getattr(self, '_last_cursor_x', None)
        current_y = getattr(self, '_last_cursor_y', None)
        
        # Scale the radius by the zoom factor
        screen_radius = int(self.ui.paint_radius * self.ui.zoom_factor)
        
        # Only update if position or size has changed significantly
        if (current_radius != screen_radius or 
            current_x is None or abs(current_x - x) > 2 or 
            current_y is None or abs(current_y - y) > 2):
            
            if hasattr(self.ui, 'paint_cursor') and self.ui.paint_cursor:
                self.ui.canvas.delete(self.ui.paint_cursor)
            
            self.ui.paint_cursor = self.ui.canvas.create_oval(
                x - screen_radius, y - screen_radius,
                x + screen_radius, y + screen_radius,
                outline="white", width=2, dash=(3, 3), tags="paint_cursor"
            )
            
            # Cache the values
            self._last_cursor_radius = screen_radius
            self._last_cursor_x = x
            self._last_cursor_y = y
            
            # Make sure cursor is on top
            self.ui.canvas.tag_raise(self.ui.paint_cursor)
    
    def update_mask_overlay(self):
        """Update the mask overlay with the edited mask"""
        # Only update if we have a mask
        if self.ui.edited_mask is None and self.ui.generated_mask is None:
            return
            
        # First, hide any existing overlay
        if self.ui.overlay_visible:
            self.hide_mask_overlay()
        
        # Update overlay with edited mask
        self.show_mask_overlay()
        
        # Make sure paint cursor stays on top of the overlay
        if hasattr(self.ui, 'paint_cursor') and self.ui.paint_cursor:
            self.ui.canvas.tag_raise(self.ui.paint_cursor)
    
    def add_to_command_stack(self, command_type, command_data):
        """
        Add a command to the undo stack
        
        Args:
            command_type: Type of command (add_point, delete_point, paint, etc.)
            command_data: Data for the command
        """
        # Create command dictionary
        command = {
            "type": command_type,
            "data": command_data
        }
        
        # Add to command stack
        self.ui.command_stack.append(command)
        
        # Clear redo stack
        self.ui.redo_stack = []
        
        # Trim command stack if it gets too large
        if len(self.ui.command_stack) > self.ui.max_undo_steps:
            self.ui.command_stack.pop(0)
    
    def undo(self, event=None):
        """Undo the last action"""
        if not self.ui.command_stack:
            self.ui.status_var.set("Nothing to undo")
            return
            
        # Get the last command
        command = self.ui.command_stack.pop()
        
        # Add to redo stack
        self.ui.redo_stack.append(command)
        
        # Handle different command types
        command_type = command["type"]
        command_data = command["data"]
        
        if command_type == "add_point":
            # Undo adding a point
            point_index = command_data["point_index"]
            if point_index < len(self.ui.points):
                # Find and remove the marker
                marker_to_delete = None
                marker_idx_to_delete = None
                
                for i, (marker, idx) in enumerate(self.ui.point_markers):
                    if idx == point_index:
                        marker_to_delete = marker
                        marker_idx_to_delete = i
                        break
                
                if marker_to_delete is not None:
                    # Remove from canvas
                    self.ui.canvas.delete(marker_to_delete)
                    
                    # Remove from our lists
                    self.ui.point_markers.pop(marker_idx_to_delete)
                    self.ui.points.pop(point_index)
                    
                    # Update indices for remaining markers
                    for i in range(len(self.ui.point_markers)):
                        marker, idx = self.ui.point_markers[i]
                        if idx > point_index:
                            self.ui.point_markers[i] = (marker, idx - 1)
            
            # Update mask after undoing a point
            if self.ui.points or self.ui.box_coords:
                self.generate_and_preview_mask()
            else:
                # No points or box left
                self.ui.generated_mask = None
                self.ui.edited_mask = None
                self.hide_mask_overlay()
                self.ui.overlay_button.config(state=tk.DISABLED)
                self.ui.confirm_button.config(state=tk.DISABLED)
                self.ui.overlay_visible = False
                
            self.ui.status_var.set("Undid adding a point")
            
        elif command_type == "delete_point":
            # Undo deleting a point
            point_index = command_data["point_index"]
            point_data = command_data["point_data"]
            
            # Add the point back
            self.ui.points.insert(point_index, point_data)
            
            # Update indices for existing markers
            for i in range(len(self.ui.point_markers)):
                marker, idx = self.ui.point_markers[i]
                if idx >= point_index:
                    self.ui.point_markers[i] = (marker, idx + 1)
            
            # Create a new marker
            x, y, label = point_data
            
            # Scale original coordinates to display coordinates
            if self.ui.scale < 1:
                display_x = x * self.ui.scale
                display_y = y * self.ui.scale
            else:
                display_x = x
                display_y = y
                
            # Apply zoom and pan
            screen_x = self.ui.interactions.canvas_to_screen_x(display_x)
            screen_y = self.ui.interactions.canvas_to_screen_y(display_y)
            
            # Scale radius by zoom factor
            radius = 5 * self.ui.zoom_factor
            
            # Create appropriate marker based on label
            color = "green" if label == 1 else "red"
            marker = self.ui.canvas.create_oval(
                screen_x - radius, screen_y - radius, 
                screen_x + radius, screen_y + radius, 
                fill=color, outline=color, tags="point_marker"
            )
            
            # Add to point markers
            self.ui.point_markers.append((marker, point_index))
            
            # Update mask after undoing point deletion
            self.generate_and_preview_mask()
            
            self.ui.status_var.set("Undid deleting a point")
            
        elif command_type == "add_box":
            # Undo adding a box
            previous_box = command_data["box_coords"]
            
            # Clear existing box markers
            for marker in self.ui.box_markers:
                self.ui.canvas.delete(marker)
            
            # Clear box data
            self.ui.box_coords = []
            self.ui.box_markers = []
            
            # Update mask
            if self.ui.points:
                self.generate_and_preview_mask()
            else:
                # No points or box left
                self.ui.generated_mask = None
                self.ui.edited_mask = None
                self.hide_mask_overlay()
                self.ui.overlay_button.config(state=tk.DISABLED)
                self.ui.confirm_button.config(state=tk.DISABLED)
                self.ui.overlay_visible = False
            
            self.ui.status_var.set("Undid adding a box")
            
        elif command_type == "clear_box":
            # Undo clearing a box
            previous_box = command_data["previous_box"]
            
            # Restore the box
            self.ui.box_coords = previous_box
            
            # Create new box markers
            if len(self.ui.box_coords) == 2:
                x1, y1 = self.ui.box_coords[0][0], self.ui.box_coords[0][1]
                x2, y2 = self.ui.box_coords[1][0], self.ui.box_coords[1][1]
                
                # Apply zoom and pan
                screen_x1 = self.ui.interactions.canvas_to_screen_x(x1)
                screen_y1 = self.ui.interactions.canvas_to_screen_y(y1)
                screen_x2 = self.ui.interactions.canvas_to_screen_x(x2)
                screen_y2 = self.ui.interactions.canvas_to_screen_y(y2)
                
                # Scale radius by zoom factor
                radius = 5 * self.ui.zoom_factor
                
                # Top-left corner
                tl_marker = self.ui.canvas.create_oval(
                    screen_x1 - radius, screen_y1 - radius, 
                    screen_x1 + radius, screen_y1 + radius,
                    fill="blue", outline="blue", tags="box_marker"
                )
                
                # Bottom-right corner
                br_marker = self.ui.canvas.create_oval(
                    screen_x2 - radius, screen_y2 - radius, 
                    screen_x2 + radius, screen_y2 + radius,
                    fill="blue", outline="blue", tags="box_marker"
                )
                
                # Box outline
                box_marker = self.ui.canvas.create_rectangle(
                    screen_x1, screen_y1, screen_x2, screen_y2,
                    outline="blue", width=2, tags="box_marker"
                )
                
                self.ui.box_markers = [tl_marker, br_marker, box_marker]
            
            # Update mask
            self.generate_and_preview_mask()
            
            self.ui.status_var.set("Undid clearing the box")
            
        elif command_type == "paint":
            # Undo paint action
            before_mask = command_data["before"]
            
            # Restore mask to before state
            self.ui.edited_mask = before_mask.copy()
            
            # Update overlay
            self.update_mask_overlay()
            
            self.ui.status_var.set("Undid paint action")
            
        elif command_type == "clear_all":
            # Undo clearing all selections
            previous_points = command_data["points"]
            previous_box_coords = command_data["box_coords"]
            previous_edited_mask = command_data["edited_mask"]
            previous_generated_mask = command_data["generated_mask"]
            
            # Restore masks
            if previous_generated_mask is not None:
                self.ui.generated_mask = previous_generated_mask.copy()
            if previous_edited_mask is not None:
                self.ui.edited_mask = previous_edited_mask.copy()
            
            # Restore points
            for i, point in enumerate(previous_points):
                x, y, label = point
                
                # Scale original coordinates to display coordinates
                if self.ui.scale < 1:
                    display_x = x * self.ui.scale
                    display_y = y * self.ui.scale
                else:
                    display_x = x
                    display_y = y
                
                # Apply zoom and pan
                screen_x = self.ui.interactions.canvas_to_screen_x(display_x)
                screen_y = self.ui.interactions.canvas_to_screen_y(display_y)
                
                # Scale radius by zoom factor
                radius = 5 * self.ui.zoom_factor
                
                # Create marker
                color = "green" if label == 1 else "red"
                marker = self.ui.canvas.create_oval(
                    screen_x - radius, screen_y - radius, 
                    screen_x + radius, screen_y + radius, 
                    fill=color, outline=color, tags="point_marker"
                )
                
                # Add to points list
                self.ui.points.append(point)
                self.ui.point_markers.append((marker, i))
            
            # Restore box
            if previous_box_coords and len(previous_box_coords) == 2:
                x1, y1 = previous_box_coords[0][0], previous_box_coords[0][1]
                x2, y2 = previous_box_coords[1][0], previous_box_coords[1][1]
                
                # Apply zoom and pan
                screen_x1 = self.ui.interactions.canvas_to_screen_x(x1)
                screen_y1 = self.ui.interactions.canvas_to_screen_y(y1)
                screen_x2 = self.ui.interactions.canvas_to_screen_x(x2)
                screen_y2 = self.ui.interactions.canvas_to_screen_y(y2)
                
                # Scale radius by zoom factor
                radius = 5 * self.ui.zoom_factor
                
                # Top-left corner
                tl_marker = self.ui.canvas.create_oval(
                    screen_x1 - radius, screen_y1 - radius, 
                    screen_x1 + radius, screen_y1 + radius,
                    fill="blue", outline="blue", tags="box_marker"
                )
                
                # Bottom-right corner
                br_marker = self.ui.canvas.create_oval(
                    screen_x2 - radius, screen_y2 - radius, 
                    screen_x2 + radius, screen_y2 + radius,
                    fill="blue", outline="blue", tags="box_marker"
                )
                
                # Box outline
                box_marker = self.ui.canvas.create_rectangle(
                    screen_x1, screen_y1, screen_x2, screen_y2,
                    outline="blue", width=2, tags="box_marker"
                )
                
                self.ui.box_coords = previous_box_coords
                self.ui.box_markers = [tl_marker, br_marker, box_marker]
            
            # Update overlay
            if self.ui.generated_mask is not None or self.ui.edited_mask is not None:
                self.show_mask_overlay()
                self.ui.overlay_visible = True
                self.ui.overlay_button.config(state=tk.NORMAL)
                self.ui.confirm_button.config(state=tk.NORMAL)
                self.ui.generate_button.config(text="Update Mask")
            
            self.ui.status_var.set("Undid clearing all selections")
    
    def redo(self, event=None):
        """Redo the last undone action"""
        if not self.ui.redo_stack:
            self.ui.status_var.set("Nothing to redo")
            return
            
        # Get the last undone command
        command = self.ui.redo_stack.pop()
        
        # Add back to command stack
        self.ui.command_stack.append(command)
        
        # Handle different command types
        command_type = command["type"]
        command_data = command["data"]
        
        if command_type == "add_point":
            # Redo adding a point
            point_index = command_data["point_index"]
            point_data = command_data["point_data"]
            
            # Add the point
            x, y, label = point_data
            
            # Scale original coordinates to display coordinates
            if self.ui.scale < 1:
                display_x = x * self.ui.scale
                display_y = y * self.ui.scale
            else:
                display_x = x
                display_y = y
            
            # Apply zoom and pan
            screen_x = self.ui.interactions.canvas_to_screen_x(display_x)
            screen_y = self.ui.interactions.canvas_to_screen_y(display_y)
            
            # Scale radius by zoom factor
            radius = 5 * self.ui.zoom_factor
            
            # Add point to list
            self.ui.points.append(point_data)
            
            # Create marker
            color = "green" if label == 1 else "red"
            marker = self.ui.canvas.create_oval(
                screen_x - radius, screen_y - radius, 
                screen_x + radius, screen_y + radius, 
                fill=color, outline=color, tags="point_marker"
            )
            
            # Add to point markers
            self.ui.point_markers.append((marker, len(self.ui.points) - 1))
            
            # Update mask
            self.generate_and_preview_mask()
            
            self.ui.status_var.set("Redid adding a point")
            
        elif command_type == "delete_point":
            # Redo deleting a point
            point_index = command_data["point_index"]
            
            # Find and remove the marker
            marker_to_delete = None
            marker_idx_to_delete = None
            
            for i, (marker, idx) in enumerate(self.ui.point_markers):
                if idx == point_index:
                    marker_to_delete = marker
                    marker_idx_to_delete = i
                    break
                    
            if marker_to_delete is not None:
                # Remove from canvas
                self.ui.canvas.delete(marker_to_delete)
                
                # Remove from our lists
                self.ui.point_markers.pop(marker_idx_to_delete)
                self.ui.points.pop(point_index)
                
                # Update indices for remaining markers
                for i in range(len(self.ui.point_markers)):
                    marker, idx = self.ui.point_markers[i]
                    if idx > point_index:
                        self.ui.point_markers[i] = (marker, idx - 1)
            
            # Update mask
            if self.ui.points or self.ui.box_coords:
                self.generate_and_preview_mask()
            else:
                # No points or box left
                self.ui.generated_mask = None
                self.ui.edited_mask = None
                self.hide_mask_overlay()
                self.ui.overlay_button.config(state=tk.DISABLED)
                self.ui.confirm_button.config(state=tk.DISABLED)
                self.ui.overlay_visible = False
            
            self.ui.status_var.set("Redid deleting a point")
            
        elif command_type == "add_box":
            # Redo adding a box
            box_coords = command_data["box_coords"]
            
            # Clear existing box markers
            for marker in self.ui.box_markers:
                self.ui.canvas.delete(marker)
            
            # Add the box
            self.ui.box_coords = box_coords
            
            # Create markers
            if len(box_coords) == 2:
                x1, y1 = box_coords[0][0], box_coords[0][1]
                x2, y2 = box_coords[1][0], box_coords[1][1]
                
                # Apply zoom and pan
                screen_x1 = self.ui.interactions.canvas_to_screen_x(x1)
                screen_y1 = self.ui.interactions.canvas_to_screen_y(y1)
                screen_x2 = self.ui.interactions.canvas_to_screen_x(x2)
                screen_y2 = self.ui.interactions.canvas_to_screen_y(y2)
                
                # Scale radius by zoom factor
                radius = 5 * self.ui.zoom_factor
                
                # Top-left corner
                tl_marker = self.ui.canvas.create_oval(
                    screen_x1 - radius, screen_y1 - radius, 
                    screen_x1 + radius, screen_y1 + radius,
                    fill="blue", outline="blue", tags="box_marker"
                )
                
                # Bottom-right corner
                br_marker = self.ui.canvas.create_oval(
                    screen_x2 - radius, screen_y2 - radius, 
                    screen_x2 + radius, screen_y2 + radius,
                    fill="blue", outline="blue", tags="box_marker"
                )
                
                # Box outline
                box_marker = self.ui.canvas.create_rectangle(
                    screen_x1, screen_y1, screen_x2, screen_y2,
                    outline="blue", width=2, tags="box_marker"
                )
                
                self.ui.box_markers = [tl_marker, br_marker, box_marker]
            
            # Update mask
            self.generate_and_preview_mask()
            
            self.ui.status_var.set("Redid adding a box")
            
        elif command_type == "clear_box":
            # Redo clearing a box
            # Clear existing box markers
            for marker in self.ui.box_markers:
                self.ui.canvas.delete(marker)
            
            # Clear box data
            self.ui.box_coords = []
            self.ui.box_markers = []
            
            # Update mask
            if self.ui.points:
                self.generate_and_preview_mask()
            else:
                # No points or box left
                self.ui.generated_mask = None
                self.ui.edited_mask = None
                self.hide_mask_overlay()
                self.ui.overlay_button.config(state=tk.DISABLED)
                self.ui.confirm_button.config(state=tk.DISABLED)
                self.ui.overlay_visible = False
            
            self.ui.status_var.set("Redid clearing the box")
            
        elif command_type == "paint":
            # Redo paint action
            after_mask = command_data["after"]
            
            # Restore mask to after state
            self.ui.edited_mask = after_mask.copy()
            
            # Update overlay
            self.update_mask_overlay()
            
            self.ui.status_var.set("Redid paint action")
            
        elif command_type == "clear_all":
            # Redo clearing all selections
            # Remove all point markers from canvas
            for marker, _ in self.ui.point_markers:
                self.ui.canvas.delete(marker)
            
            # Remove all box markers from canvas
            for marker in self.ui.box_markers:
                self.ui.canvas.delete(marker)
            
            # Clear data
            self.ui.points = []
            self.ui.point_markers = []
            self.ui.box_coords = []
            self.ui.box_markers = []
            
            # Clear masks
            self.ui.generated_mask = None
            self.ui.edited_mask = None
            
            # Hide overlay
            self.hide_mask_overlay()
            self.ui.overlay_button.config(state=tk.DISABLED)
            self.ui.confirm_button.config(state=tk.DISABLED)
            self.ui.overlay_visible = False
            
            self.ui.status_var.set("Redid clearing all selections")
    
    def apply_loaded_mask(self, mask_input):
        """Apply a loaded mask to the current frame"""
        try:
            # Handle both file path and numpy array inputs
            if isinstance(mask_input, str):
                # Load from file path
                from PIL import Image
                mask_image = Image.open(mask_input).convert('L')
                mask_array = np.array(mask_image)
            else:
                # Use provided numpy array
                mask_array = mask_input
            
            # Resize mask to match current image dimensions if needed
            from PIL import Image
            current_height, current_width = self.ui.image.shape[:2]
            mask_height, mask_width = mask_array.shape
            
            if mask_height != current_height or mask_width != current_width:
                # Resize mask to match image dimensions
                mask_pil = Image.fromarray(mask_array)
                mask_pil = mask_pil.resize((current_width, current_height), Image.NEAREST)
                mask_array = np.array(mask_pil)
            
            # Convert to boolean mask (assuming mask is grayscale where >0 means foreground)
            mask_bool = mask_array > 127  # Threshold at middle gray
            
            # Store as both generated and edited mask
            self.ui.generated_mask = mask_bool.copy()
            self.ui.edited_mask = mask_bool.copy()
            
            # Clear any existing selections since we're loading a complete mask
            self.ui.points = []
            self.ui.box_coords = None
            
            # Create checkpoint for this frame (like after generating a mask)
            self.ui.checkpoints[self.ui.current_frame_index] = {
                'mask': mask_bool.copy(),
                'points': [],
                'box_coords': [],
                'edited_mask': mask_bool.copy()
            }
            
            # Update checkpoint markers
            self.ui.frame_manager.update_checkpoint_markers()
            
            # Update the display
            self.show_mask_overlay()
            
            # Enable overlay button and show overlay
            if hasattr(self.ui, 'overlay_button'):
                self.ui.overlay_button.config(state=tk.NORMAL, text="Hide Overlay")
            self.ui.overlay_visible = True
            
            # Enable confirm button
            if hasattr(self.ui, 'confirm_button'):
                self.ui.confirm_button.config(state=tk.NORMAL)
            
            # Mark this frame as having a generated mask (for red highlighting)
            if hasattr(self.ui, 'mark_frame_with_mask'):
                self.ui.mark_frame_with_mask(self.ui.current_frame_index)
            
            self.ui.status_var.set("Mask loaded successfully")
            
        except Exception as e:
            self.ui.status_var.set(f"Error loading mask: {str(e)}")
            import traceback
            traceback.print_exc()