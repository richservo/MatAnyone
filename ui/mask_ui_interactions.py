"""
# mask_ui_interactions.py - v1.1747845070
# Updated: Wednesday, May 21, 2025 at 09:29:52 PST
# Changes in this version:
# - Optimized coordinate transformations for new initialization approach
# - Cleaned up code for increased reliability
# - Works with the frame navigation workaround to ensure correct coordinates
# - Removed unnecessary debugging and extra checks
# - Compatible with new fit-to-window initialization

UI interaction handlers for the SAM-based mask generator.
Handles mouse events, keyboard shortcuts, and other user interactions.
"""

import tkinter as tk
import numpy as np
import cv2
import time
import platform
from PIL import Image, ImageTk


class MaskUIInteractions:
    """Handles user interactions with the mask generator UI"""
    
    def __init__(self, ui):
        """
        Initialize the interaction handler
        
        Args:
            ui: Reference to the MaskUIBase object
        """
        self.ui = ui
        self.original_cursor = None
    
    def switch_mode(self):
        """Switch between interaction modes without clearing selections"""
        # Force recalculation of point markers in their correct positions
        self.update_point_markers()
        
        mode = self.ui.interaction_mode.get()
        
        # Update the UI based on the selected mode
        if mode == "point":
            self.ui.status_var.set("Click to select foreground points (green), right-click for background (red)")
            # Hide paint settings
            self.ui.paint_frame.pack_forget()
            # Reset cursor
            self.ui.canvas.config(cursor=self.ui.default_cursor)
            # Clear any paint cursor
            if hasattr(self.ui, 'paint_cursor') and self.ui.paint_cursor:
                self.ui.canvas.delete(self.ui.paint_cursor)
                self.ui.paint_cursor = None
            
        elif mode == "box":
            self.ui.status_var.set("Click and drag to define a bounding box")
            # Hide paint settings
            self.ui.paint_frame.pack_forget()
            # Reset cursor
            self.ui.canvas.config(cursor=self.ui.default_cursor)
            # Clear any paint cursor
            if hasattr(self.ui, 'paint_cursor') and self.ui.paint_cursor:
                self.ui.canvas.delete(self.ui.paint_cursor)
                self.ui.paint_cursor = None
            
        elif mode == "paint":
            # Check if we have a generated mask
            if self.ui.generated_mask is None and self.ui.edited_mask is None:
                from tkinter import messagebox
                messagebox.showinfo("Paint Mode", "Please generate a mask first before using paint mode.")
                self.ui.interaction_mode.set("point")  # Switch back to point mode
                return
                
            self.ui.status_var.set("Left-click to add to mask, right-click to remove from mask")
            # Show paint settings
            self.ui.paint_frame.pack(pady=5, fill=tk.X, padx=20, after=self.ui.paint_frame.master.winfo_children()[1])
            
            # Hide system cursor in paint mode
            self.ui.canvas.config(cursor=self.ui.empty_cursor)
            
            # Ensure mask is visible
            if not self.ui.overlay_visible:
                self.ui.editor.show_mask_overlay()
                self.ui.overlay_button.config(text="Hide Overlay")
                
        elif mode == "move":
            self.ui.status_var.set("Click and drag to move the view. Use + and - buttons to zoom.")
            # Hide paint settings
            self.ui.paint_frame.pack_forget()
            # Set move cursor
            self.ui.canvas.config(cursor="fleur")  # Hand/move cursor
            # Clear any paint cursor
            if hasattr(self.ui, 'paint_cursor') and self.ui.paint_cursor:
                self.ui.canvas.delete(self.ui.paint_cursor)
                self.ui.paint_cursor = None
    
    def ensure_coordinate_systems_initialized(self):
        """
        Ensure coordinate systems are properly initialized
        This is important for the first frame where coordinates might not be properly set up
        """
        # The coordinate systems should be fully initialized in initialize_ui now,
        # so this is just a safety check to make sure all required properties exist.
        
        # Check if initialization has already been completed
        if hasattr(self.ui, 'coordinate_systems_initialized') and self.ui.coordinate_systems_initialized:
            return
        
        # If not initialized for some reason, set basic defaults - this should never happen
        # because the navigation fix should have set everything up properly
        if not hasattr(self.ui, 'canvas_offset_x'):
            self.ui.canvas_offset_x = 0
        
        if not hasattr(self.ui, 'canvas_offset_y'):
            self.ui.canvas_offset_y = 0
        
        if not hasattr(self.ui, 'zoom_factor'):
            self.ui.zoom_factor = 1.0
            
        if not hasattr(self.ui, 'fit_to_window'):
            self.ui.fit_to_window = True
            
        if not hasattr(self.ui, 'fit_scale'):
            self.ui.fit_scale = 1.0
        
        # Flag as initialized to prevent any further reset attempts
        self.ui.coordinate_systems_initialized = True
    
    def on_canvas_click(self, event):
        """Handle left-click events on the canvas"""
        # Ensure coordinate systems are properly initialized
        self.ensure_coordinate_systems_initialized()
        
        
        # Handle clicks based on the current interaction mode
        mode = self.ui.interaction_mode.get()
        
        # In move mode, start panning
        if mode == "move":
            self.handle_pan_start(event)
            return
            
        # If we're in spacebar panning mode, handle differently
        if hasattr(self.ui, 'pan_mode_active') and self.ui.pan_mode_active:
            self.handle_pan_start(event)
            return
        
        # In paint mode, directly modify the mask
        if mode == "paint":
            # For paint mode, paint to add to mask (if mask exists)
            if self.ui.edited_mask is not None or self.ui.generated_mask is not None:
                self.ui.is_painting = True
                # Store event coordinates
                self.ui.last_paint_x = event.x
                self.ui.last_paint_y = event.y
                
                # Start accumulating paint actions for batch processing
                self.ui.accumulated_paint_actions = []
                
                # Apply paint to add to mask (foreground)
                self.ui.editor.apply_paint_stroke(event.x, event.y, True)
                
                # Update paint cursor to follow the stroke
                self.ui.editor.update_paint_cursor(event.x, event.y)
            return
        
        # For other modes, check if clicking on an existing point
        clicked_point = self.check_point_click(event.x, event.y)
        if clicked_point is not None:
            # Add to command stack before deleting
            self.ui.editor.add_to_command_stack(
                "delete_point", 
                {"point_index": clicked_point, "point_data": self.ui.points[clicked_point].copy()}
            )
            
            self.ui.editor.delete_point(clicked_point)
            
            # Auto-update mask after deleting a point
            self.ui.editor.generate_and_preview_mask()
            return
            
        if mode == "point":
            # For point mode, add a point (exactly like in v039)
            # Convert to original image coordinates
            orig_x, orig_y = self.scale_coords_to_original(event.x, event.y)
            
            # Add point as foreground (label=1)
            point_index = len(self.ui.points)
            
            # Add to command stack before adding the point
            self.ui.editor.add_to_command_stack(
                "add_point", 
                {"point_index": point_index, "point_data": [orig_x, orig_y, 1]}
            )
            
            # Store the point
            self.ui.points.append([orig_x, orig_y, 1])
            
            # Draw a green circle for foreground point exactly where clicked
            marker = self.ui.canvas.create_oval(
                event.x - 5, event.y - 5, 
                event.x + 5, event.y + 5, 
                fill="green", outline="green", tags="point_marker"
            )
            self.ui.point_markers.append((marker, point_index))
            
            self.ui.status_var.set(f"Added foreground point at ({orig_x}, {orig_y}). "
                                f"Right-click for background points. {len(self.ui.points)} points total.")
            
            # Auto-update mask after adding a point
            self.ui.editor.generate_and_preview_mask()
            
        elif mode == "box":
            # Start box drawing
            self.ui.drawing_box = True
            self.ui.box_start_x = event.x
            self.ui.box_start_y = event.y
            # Convert to original coordinates
            orig_x, orig_y = self.scale_coords_to_original(event.x, event.y)
            self.ui.box_start_orig_x = orig_x
            self.ui.box_start_orig_y = orig_y
            
            
            # Create a placeholder rectangle with raw screen coordinates
            self.ui.temp_box = self.ui.canvas.create_rectangle(
                event.x, 
                event.y, 
                event.x, 
                event.y,
                outline="blue", width=2, tags="temp_box"
            )
    
    def on_canvas_right_click(self, event):
        """Handle right-click events on the canvas (background points or remove from mask)"""
        # Ensure coordinate systems are properly initialized
        self.ensure_coordinate_systems_initialized()
        
        # If we're in panning mode, ignore right-clicks
        if hasattr(self.ui, 'pan_mode_active') and self.ui.pan_mode_active:
            return
            
        # Handle clicks based on the current interaction mode
        mode = self.ui.interaction_mode.get()
        
        # In paint mode, we're removing from the mask
        if mode == "paint":
            # For paint mode with right-click, paint to remove from mask
            if self.ui.edited_mask is not None or self.ui.generated_mask is not None:
                self.ui.is_painting = True
                # Store event coordinates
                self.ui.last_bg_paint_x = event.x
                self.ui.last_bg_paint_y = event.y
                
                # Start accumulating actions
                self.ui.accumulated_paint_actions = []
                
                # Apply paint to remove from mask (background)
                self.ui.editor.apply_paint_stroke(event.x, event.y, False)
                
                # Update cursor
                self.ui.editor.update_paint_cursor(event.x, event.y)
            return
            
        # For other modes, check if clicking on an existing point
        clicked_point = self.check_point_click(event.x, event.y)
        if clicked_point is not None:
            # Add to command stack before deleting
            self.ui.editor.add_to_command_stack(
                "delete_point", 
                {"point_index": clicked_point, "point_data": self.ui.points[clicked_point].copy()}
            )
            
            self.ui.editor.delete_point(clicked_point)
            
            # Auto-update mask after deleting a point
            self.ui.editor.generate_and_preview_mask()
            return
            
        if mode == "point":
            # For point mode, add a background point (exactly like v039)
            # Convert to original image coordinates
            orig_x, orig_y = self.scale_coords_to_original(event.x, event.y)
            
            # Add point as background (label=0)
            point_index = len(self.ui.points)
            
            # Add to command stack before adding the point
            self.ui.editor.add_to_command_stack(
                "add_point", 
                {"point_index": point_index, "point_data": [orig_x, orig_y, 0]}
            )
            
            # Store the point
            self.ui.points.append([orig_x, orig_y, 0])
            
            # Draw a red circle for background point
            marker = self.ui.canvas.create_oval(
                event.x - 5, event.y - 5, 
                event.x + 5, event.y + 5, 
                fill="red", outline="red", tags="point_marker"
            )
            self.ui.point_markers.append((marker, point_index))
            
            self.ui.status_var.set(f"Added background point at ({orig_x}, {orig_y}). "
                               f"{len(self.ui.points)} points total.")
            
            # Auto-update mask after adding a point
            self.ui.editor.generate_and_preview_mask()
    
    def on_mouse_motion(self, event):
        """Handle mouse motion events"""
        # If we're in panning mode (either via spacebar or move mode), handle panning
        if self.ui.is_panning:
            self.handle_pan_motion(event)
            return
            
        # Also check if we're in move mode with mouse down (panning)
        if self.ui.interaction_mode.get() == "move" and self.ui.is_panning:
            self.handle_pan_motion(event)
            return
            
        if self.ui.interaction_mode.get() == "box" and self.ui.drawing_box:
            # Update the box during drag
            if self.ui.temp_box:
                # Get screen coordinates directly
                start_screen_x = self.ui.box_start_x
                start_screen_y = self.ui.box_start_y
                
                # Update the box with raw screen coordinates
                self.ui.canvas.coords(self.ui.temp_box, 
                                   start_screen_x, 
                                   start_screen_y, 
                                   event.x, 
                                   event.y)
                
        elif self.ui.interaction_mode.get() == "paint" and self.ui.is_painting:
            # Update paint cursor to follow mouse
            self.ui.editor.update_paint_cursor(event.x, event.y)
            
            # For paint mode, continue painting to add to mask
            # Calculate points along the line for smoother results
            if hasattr(self.ui, 'last_paint_x'):
                # Calculate spacing based on zoom level and brush size
                # When zoomed in, we need more points for smooth drawing
                spacing = max(1, int(self.ui.paint_radius/(2 * max(1, self.ui.zoom_factor))))
                
                # Use raw screen coordinates for point calculation
                points_on_line = self.get_points_on_line(
                    self.ui.last_paint_x, self.ui.last_paint_y, 
                    event.x, event.y,
                    spacing=spacing  # Adaptive spacing for different zoom levels
                )
                
                # Add each point along the line
                for px, py in points_on_line:
                    self.ui.editor.apply_paint_stroke(px, py, True)  # True for add to mask
            
            # Update last position
            self.ui.last_paint_x = event.x
            self.ui.last_paint_y = event.y
    
    def on_right_paint_motion(self, event):
        """Handle right-click paint motion (for removing from mask)"""
        # If we're in panning mode, ignore right-paint motions
        if hasattr(self.ui, 'pan_mode_active') and self.ui.pan_mode_active:
            return
            
        if self.ui.interaction_mode.get() == "paint":
            # Update paint cursor to follow mouse
            self.ui.editor.update_paint_cursor(event.x, event.y)
            
            # For paint mode with right-click, continue painting to remove from mask
            if not hasattr(self.ui, 'last_bg_paint_x'):
                # Initialize position on first motion
                self.ui.last_bg_paint_x = event.x
                self.ui.last_bg_paint_y = event.y
                
                # Start batch processing
                self.ui.accumulated_paint_actions = []
                
                # Apply first stroke
                self.ui.editor.apply_paint_stroke(event.x, event.y, False)  # False for remove from mask
            
            # Calculate spacing based on zoom level and brush size
            spacing = max(1, int(self.ui.paint_radius/(2 * max(1, self.ui.zoom_factor))))
            
            # Calculate points along the line for smoother results
            points_on_line = self.get_points_on_line(
                self.ui.last_bg_paint_x, self.ui.last_bg_paint_y, 
                event.x, event.y,
                spacing=spacing  # Adaptive spacing for smoother strokes
            )
            
            # Add each point along the line
            for px, py in points_on_line:
                self.ui.editor.apply_paint_stroke(px, py, False)  # False for remove from mask
            
            # Update last position
            self.ui.last_bg_paint_x = event.x
            self.ui.last_bg_paint_y = event.y
    
    def on_button_release(self, event):
        """Handle mouse button release"""
        # End panning if we were panning (either via spacebar or move mode)
        if self.ui.is_panning and event.num == 1:  # Left mouse button
            self.ui.is_panning = False
            # If we're in move mode, keep the move cursor
            if self.ui.interaction_mode.get() == "move":
                self.ui.canvas.config(cursor="fleur")
            return
            
        if self.ui.interaction_mode.get() == "box" and self.ui.drawing_box:
            # Finish box drawing
            self.ui.drawing_box = False
            
            if self.ui.temp_box:
                # Get the final box coordinates
                coords = self.ui.canvas.coords(self.ui.temp_box)
                if len(coords) == 4:
                    # First convert to display coordinates to handle scrolling, zoom and pan
                    # Get the raw screen coordinates first, not the transformed ones
                    screen_x1, screen_y1 = coords[0], coords[1]
                    screen_x2, screen_y2 = coords[2], coords[3]
                    
                    # Then convert screen coordinates to display coordinates
                    display_x1, display_y1 = self.screen_to_display(screen_x1, screen_y1)
                    display_x2, display_y2 = self.screen_to_display(screen_x2, screen_y2)
                    
                    # Convert display coordinates to original image coordinates
                    orig_x1, orig_y1 = self.display_to_original(display_x1, display_y1)
                    orig_x2, orig_y2 = self.display_to_original(display_x2, display_y2)
                    
                    # Convert to integers
                    orig_x1, orig_y1 = int(orig_x1), int(orig_y1)
                    orig_x2, orig_y2 = int(orig_x2), int(orig_y2)
                    
                    # Ensure x1,y1 is the top-left and x2,y2 is the bottom-right
                    if orig_x1 > orig_x2:
                        orig_x1, orig_x2 = orig_x2, orig_x1
                        display_x1, display_x2 = display_x2, display_x1
                    if orig_y1 > orig_y2:
                        orig_y1, orig_y2 = orig_y2, orig_y1
                        display_y1, display_y2 = display_y2, display_y1
                    
                    # Delete the temporary box
                    self.ui.canvas.delete(self.ui.temp_box)
                    self.ui.temp_box = None
                    
                    # Add to command stack before adding the box
                    # First clear any existing box
                    if self.ui.box_coords:
                        self.ui.editor.add_to_command_stack(
                            "clear_box", 
                            {"previous_box": self.ui.box_coords.copy()}
                        )
                        
                        # Clear existing box markers
                        for marker in self.ui.box_markers:
                            self.ui.canvas.delete(marker)
                        self.ui.box_markers = []
                        
                        # Clear existing box coordinates
                        self.ui.box_coords = []
                    
                    # Add the new box
                    self.ui.editor.add_to_command_stack(
                        "add_box", 
                        {"box_coords": [(display_x1, display_y1, orig_x1, orig_y1), (display_x2, display_y2, orig_x2, orig_y2)]}
                    )
                    
                    # Calculate screen coordinates directly from original coordinates
                    # This ensures perfect alignment with the display frame
                    display_x1, display_y1 = self.original_to_display(orig_x1, orig_y1)
                    display_x2, display_y2 = self.original_to_display(orig_x2, orig_y2)
                    
                    # Convert display coordinates to screen coordinates
                    # Convert with display_to_screen which returns a tuple, not individual coordinates
                    screen_coords1 = self.display_to_screen(display_x1, display_y1)
                    screen_coords2 = self.display_to_screen(display_x2, display_y2)
                    
                    # Unpack the screen coordinates
                    screen_x1, screen_y1 = screen_coords1
                    screen_x2, screen_y2 = screen_coords2
                    
                    # Calculate marker size with zoom
                    marker_radius = 5 * self.ui.zoom_factor
                    
                    # Create a permanent box marker
                    # Top-left corner
                    tl_marker = self.ui.canvas.create_oval(
                        screen_x1 - marker_radius, 
                        screen_y1 - marker_radius, 
                        screen_x1 + marker_radius, 
                        screen_y1 + marker_radius,
                        fill="blue", outline="blue", tags="box_marker"
                    )
                    
                    # Bottom-right corner
                    br_marker = self.ui.canvas.create_oval(
                        screen_x2 - marker_radius, 
                        screen_y2 - marker_radius, 
                        screen_x2 + marker_radius, 
                        screen_y2 + marker_radius,
                        fill="blue", outline="blue", tags="box_marker"
                    )
                    
                    # Box outline
                    box_marker = self.ui.canvas.create_rectangle(
                        screen_x1, 
                        screen_y1, 
                        screen_x2, 
                        screen_y2,
                        outline="blue", width=2, tags="box_marker"
                    )
                    
                    # Store the box information
                    self.ui.box_markers.extend([tl_marker, br_marker, box_marker])
                    self.ui.box_coords = [(display_x1, display_y1, orig_x1, orig_y1), (display_x2, display_y2, orig_x2, orig_y2)]
                    
                    self.ui.status_var.set(f"Box defined from ({orig_x1}, {orig_y1}) to ({orig_x2}, {orig_y2})")
                    
                    # Auto-update mask after drawing a box
                    self.ui.editor.generate_and_preview_mask()
                    
                    # Switch to point mode for refinement
                    self.ui.interaction_mode.set("point")
                    self.switch_mode()
        
        elif self.ui.interaction_mode.get() == "paint":
            # End painting
            if self.ui.is_painting:
                # Save current state to command stack before ending the paint action
                if self.ui.edited_mask is not None:
                    self.ui.editor.end_paint_action()
                
            self.ui.is_painting = False
            if hasattr(self.ui, 'last_bg_paint_x'):
                delattr(self.ui, 'last_bg_paint_x')
                delattr(self.ui, 'last_bg_paint_y')
            
            # Process any remaining paint actions
            if self.ui.accumulated_paint_actions:
                self.ui.editor.process_accumulated_paint_actions()
    
    def on_mouse_move(self, event):
        """Handle mouse movement to update the paint cursor"""
        # Store current mouse position for keyboard shortcuts
        self.ui.last_mouse_pos = (event.x, event.y)
        
        if self.ui.interaction_mode.get() == "paint":
            self.ui.editor.update_paint_cursor(event.x, event.y)
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        # [ key decreases brush size
        if event.char == '[':
            self.adjust_brush_size(-1)
        # ] key increases brush size
        elif event.char == ']':
            self.adjust_brush_size(1)
        # Left arrow key for previous frame
        elif event.keysym == 'Left':
            self.ui.frame_manager.navigate_frame(-1)
        # Right arrow key for next frame
        elif event.keysym == 'Right':
            self.ui.frame_manager.navigate_frame(1)
    
    def update_brush_size(self, event=None):
        """Update the paint brush size based on combobox selection"""
        try:
            # Check if there's an active value in the combobox
            combo_value = self.ui.brush_size_combo.get()
            if combo_value:
                new_size = int(combo_value)
                self.ui.paint_radius = min(self.ui.max_brush_size, max(self.ui.min_brush_size, new_size))
            else:
                # Use current paint radius
                self.ui.brush_size_combo.set(str(self.ui.paint_radius))
            
            # Update cursor if in paint mode
            if self.ui.interaction_mode.get() == "paint" and hasattr(self.ui, 'last_mouse_pos'):
                x, y = self.ui.last_mouse_pos
                self.ui.editor.update_paint_cursor(x, y)
                
            # Update status
            self.ui.status_var.set(f"Brush size: {self.ui.paint_radius}")
            
            # Save settings
            self.ui.save_settings()
        except:
            # Default fallback
            if not hasattr(self.ui, 'paint_radius') or not self.ui.paint_radius:
                self.ui.paint_radius = 5
    
    def adjust_brush_size(self, delta):
        """Adjust brush size by the given delta"""
        if self.ui.interaction_mode.get() != "paint":
            return
            
        new_size = max(self.ui.min_brush_size, min(self.ui.max_brush_size, self.ui.paint_radius + delta))
        self.ui.paint_radius = new_size
        
        # Update combobox if possible
        sizes = self.ui.brush_size_combo['values']
        if str(new_size) in sizes:
            self.ui.brush_size_combo.set(str(new_size))
        else:
            # Just update the variable without changing combobox selection
            self.ui.brush_size_combo.set("")
        
        # Update cursor
        if hasattr(self.ui, 'last_mouse_pos'):
            x, y = self.ui.last_mouse_pos
            self.ui.editor.update_paint_cursor(x, y)
        
        # Update status
        self.ui.status_var.set(f"Brush size: {new_size}")
        
        # Save settings
        self.ui.save_settings()
    
    def check_point_click(self, x, y):
        """
        Check if a click was near an existing point
        
        Args:
            x, y: Canvas coordinates of click
            
        Returns:
            Point index if click was on a point, None otherwise
        """
        for marker, point_index in self.ui.point_markers:
            # Get the coordinates of the marker
            x1, y1, x2, y2 = self.ui.canvas.coords(marker)
            
            # Calculate center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Check if click is within radius of point using a larger detection radius
            # to make it easier to hit points
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            if distance <= 15:  # Use a slightly larger click radius than visual marker
                return point_index
        
        return None
    
    def get_points_on_line(self, x1, y1, x2, y2, spacing=5):
        """
        Get evenly spaced points along a line
        
        Args:
            x1, y1: Start point
            x2, y2: End point
            spacing: Spacing between points
            
        Returns:
            List of (x, y) points along the line
        """
        # Ensure all inputs are converted to integers
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)
        spacing = max(1, int(spacing))
        
        # Calculate distance and number of points
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        
        # If very close, just return the end point
        if distance < spacing:
            return [(x2, y2)]
            
        # Calculate number of segments
        num_segments = max(1, int(distance / spacing))
        
        # Generate points
        points = []
        for i in range(num_segments + 1):
            # Calculate point at given fraction of the way
            t = i / num_segments  # 0 to 1
            x = int(x1 + (x2 - x1) * t)
            y = int(y1 + (y2 - y1) * t)
            points.append((x, y))
            
        return points
    
    # =====================================================================
    # COORDINATE TRANSFORMATION FUNCTIONS
    # 
    # There are three coordinate systems in use:
    # 1. Original Image: Coordinates in the original image (e.g., a 4K video frame)
    # 2. Display Canvas: Coordinates in the scaled-down display image
    # 3. Screen: Coordinates with zoom and pan applied
    # =====================================================================
    
    def original_to_display(self, x, y):
        """Convert original image coordinates to display canvas coordinates"""
        if self.ui.scale < 1:
            display_x, display_y = x * self.ui.scale, y * self.ui.scale
        else:
            display_x, display_y = x, y
            
        
        return display_x, display_y
    
    def display_to_original(self, x, y):
        """Convert display canvas coordinates to original image coordinates"""
        if self.ui.scale < 1:
            orig_x, orig_y = x / self.ui.scale, y / self.ui.scale
        else:
            orig_x, orig_y = x, y
            
        
        return orig_x, orig_y
    
    def display_to_screen(self, x, y):
        """Convert display canvas coordinates to screen coordinates (with zoom/pan)"""
        screen_x = self.ui.canvas_offset_x + (x * self.ui.zoom_factor)
        screen_y = self.ui.canvas_offset_y + (y * self.ui.zoom_factor)
        
        
        return screen_x, screen_y
    
    def screen_to_display(self, x, y):
        """Convert screen coordinates (with zoom/pan) to display canvas coordinates"""
        display_x = (x - self.ui.canvas_offset_x) / self.ui.zoom_factor
        display_y = (y - self.ui.canvas_offset_y) / self.ui.zoom_factor
        
        
        return display_x, display_y
    
    def original_to_screen(self, x, y):
        """Convert original image coordinates to screen coordinates"""
        display_x, display_y = self.original_to_display(x, y)
        return self.display_to_screen(display_x, display_y)
    
    def screen_to_original(self, x, y):
        """Convert screen coordinates to original image coordinates"""
        display_x, display_y = self.screen_to_display(x, y)
        return self.display_to_original(display_x, display_y)
    
    # Simple coordinate conversion directly from v039
    def scale_coords_to_original(self, x, y):
        """Scale screen coordinates to original image coordinates"""
        # First convert screen coordinates to display coordinates to handle panning and zooming
        display_x, display_y = self.screen_to_display(x, y)
        
        # Then convert display coordinates to original image coordinates
        orig_x, orig_y = self.display_to_original(display_x, display_y)
        
        # Ensure coordinates are integers
        orig_x, orig_y = int(orig_x), int(orig_y)
        
        
        return orig_x, orig_y
    
    def canvas_to_screen_x(self, x):
        """Convert display canvas x-coordinate to screen coordinate"""
        if isinstance(x, (int, float)):
            screen_x, _ = self.display_to_screen(x, 0)
            return screen_x
        return x  # Pass through non-numeric values
    
    def canvas_to_screen_y(self, y):
        """Convert display canvas y-coordinate to screen coordinate"""
        if isinstance(y, (int, float)):
            _, screen_y = self.display_to_screen(0, y)
            return screen_y
        return y  # Pass through non-numeric values
        
    # Debug utility function (disabled)
    def debug_coordinate_transform(self, event_x, event_y, prefix=""):
        """Log coordinate transformation details for debugging (currently disabled)"""
        pass
    
    def screen_to_canvas_x(self, x):
        """Convert screen x-coordinate to display canvas coordinate"""
        if isinstance(x, (int, float)):
            display_x, _ = self.screen_to_display(x, 0)
            return display_x
        return x  # Pass through non-numeric values
    
    def screen_to_canvas_y(self, y):
        """Convert screen y-coordinate to display canvas coordinate"""
        if isinstance(y, (int, float)):
            _, display_y = self.screen_to_display(0, y)
            return display_y
        return y  # Pass through non-numeric values
    
    def start_pan_mode(self, event):
        """Start pan mode when spacebar is pressed (Photoshop-style)"""
        # Store original cursor
        self.original_cursor = self.ui.canvas.cget("cursor")
        # Change cursor to hand
        self.ui.canvas.config(cursor="hand2")
        # Set panning mode flag
        self.ui.pan_mode_active = True
        # Update status
        self.ui.status_var.set("Pan mode active (click and drag to move)")
        
        # Prevent event propagation
        return "break"
    
    def end_pan_mode(self, event):
        """End pan mode when spacebar is released"""
        if hasattr(self.ui, 'pan_mode_active') and self.ui.pan_mode_active:
            if self.original_cursor:
                # Restore original cursor
                self.ui.canvas.config(cursor=self.original_cursor)
            elif self.ui.interaction_mode.get() == "paint":
                # If we're in paint mode, use empty cursor
                self.ui.canvas.config(cursor=self.ui.empty_cursor)
            else:
                # Otherwise use default cursor
                self.ui.canvas.config(cursor=self.ui.default_cursor)
            
            # Clear panning flags
            self.ui.pan_mode_active = False
            self.ui.is_panning = False
            # Reset status
            self.ui.status_var.set("Pan mode deactivated")
            
        # Prevent event propagation
        return "break"
    
    def handle_pan_start(self, event):
        """Handle start of panning"""
        self.ui.is_panning = True
        self.ui.pan_start_x = event.x
        self.ui.pan_start_y = event.y
        
        # Store current canvas offset
        self.pan_orig_offset_x = self.ui.canvas_offset_x
        self.pan_orig_offset_y = self.ui.canvas_offset_y
    
    def handle_pan_motion(self, event):
        """Handle panning motion"""
        if not self.ui.is_panning:
            return
            
        # Calculate delta from start position
        delta_x = event.x - self.ui.pan_start_x
        delta_y = event.y - self.ui.pan_start_y
        
        # Update canvas offset
        self.ui.canvas_offset_x = self.pan_orig_offset_x + delta_x
        self.ui.canvas_offset_y = self.pan_orig_offset_y + delta_y
        
        # Update all canvas items
        self.update_all_canvas_items()
    
    def update_all_canvas_items(self):
        """Update all canvas items with the current zoom and pan"""
        # Get original image dimensions
        img_height, img_width = self.ui.display_frame.shape[:2]
        
        # Calculate the zoomed dimensions
        zoomed_width = int(img_width * self.ui.zoom_factor)
        zoomed_height = int(img_height * self.ui.zoom_factor)
        
        # Resize the image to the zoomed dimensions
        try:
            pil_image = Image.fromarray(self.ui.display_frame)
            # Use LANCZOS for high quality or BILINEAR for speed
            resample_method = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
            zoomed_pil = pil_image.resize((zoomed_width, zoomed_height), resample_method)
            self.ui.tk_image = ImageTk.PhotoImage(image=zoomed_pil)
            
            # Update the image on canvas
            self.ui.canvas.delete("base_image")
            self.ui.image_container = self.ui.canvas.create_image(
                self.ui.canvas_offset_x, self.ui.canvas_offset_y, 
                anchor=tk.NW, image=self.ui.tk_image, tags="base_image"
            )
        except Exception as e:
            print(f"Error updating image: {str(e)}")
        
        # Update the overlay if it exists
        if self.ui.overlay_visible and hasattr(self.ui, 'overlay_image'):
            self.ui.editor.show_mask_overlay()
        
        # Update all point markers
        self.update_point_markers()
        
        # Update all box markers
        self.update_box_markers()
        
        # Update paint cursor if it exists
        if hasattr(self.ui, 'paint_cursor') and self.ui.paint_cursor and hasattr(self.ui, 'last_mouse_pos'):
            x, y = self.ui.last_mouse_pos
            self.ui.editor.update_paint_cursor(x, y)
        
        # Update the scroll region
        self.ui.canvas.config(scrollregion=self.ui.canvas.bbox(tk.ALL))
    
    def update_point_markers(self):
        """Update point markers with the current zoom and pan"""
        # Update all point markers
        for i, (marker, point_index) in enumerate(self.ui.point_markers):
            # Get point data
            point = self.ui.points[point_index]
            orig_x, orig_y, label = point
            
            # Get screen coordinates directly from original coordinates
            screen_x, screen_y = self.original_to_screen(orig_x, orig_y)
            
            # Scale radius by zoom factor
            radius = 5 * self.ui.zoom_factor
            
            # Update marker
            self.ui.canvas.coords(marker, 
                             screen_x - radius, screen_y - radius, 
                             screen_x + radius, screen_y + radius)
    
    def update_box_markers(self):
        """Update box markers with the current zoom and pan"""
        # Update all box markers
        if self.ui.box_coords and len(self.ui.box_coords) == 2 and len(self.ui.box_markers) >= 3:
            # Get box coordinates from the saved data
            # Structure is: [(x1, y1, orig_x1, orig_y1), (x2, y2, orig_x2, orig_y2)]
            canvas_x1, canvas_y1, orig_x1, orig_y1 = self.ui.box_coords[0]
            canvas_x2, canvas_y2, orig_x2, orig_y2 = self.ui.box_coords[1]
            
            # Convert original coordinates to screen coordinates
            screen_x1, screen_y1 = self.original_to_screen(orig_x1, orig_y1)
            screen_x2, screen_y2 = self.original_to_screen(orig_x2, orig_y2)
            
            # Scale radius by zoom factor
            radius = 5 * self.ui.zoom_factor
            
            # Update top-left corner
            self.ui.canvas.coords(self.ui.box_markers[0], 
                             screen_x1 - radius, screen_y1 - radius, 
                             screen_x1 + radius, screen_y1 + radius)
            
            # Update bottom-right corner
            self.ui.canvas.coords(self.ui.box_markers[1], 
                             screen_x2 - radius, screen_y2 - radius, 
                             screen_x2 + radius, screen_y2 + radius)
            
            # Update box outline
            self.ui.canvas.coords(self.ui.box_markers[2], 
                             screen_x1, screen_y1, 
                             screen_x2, screen_y2)
    
    def on_mouse_wheel(self, event):
        """
        Handle mouse wheel events for zooming - simplified version
        
        Note: We're intentionally not doing anything with wheel events now.
        The user will use the + and - buttons for zooming as they're more reliable
        across platforms.
        """
        # Just prevent the event from propagating (to avoid scrolling the window)
        return "break"
    
    def zoom(self, factor, x=None, y=None):
        """
        Zoom the canvas by the given factor
        
        Args:
            factor: Zoom factor (> 1 for zoom in, < 1 for zoom out)
            x, y: Center of zoom (if None, use center of canvas)
        """
        # Get current canvas size
        canvas_width = self.ui.canvas.winfo_width()
        canvas_height = self.ui.canvas.winfo_height()
        
        # Debug info
        print(f"Zoom: factor={factor}, canvas_size=({canvas_width}x{canvas_height})")
        print(f"  Current zoom: {self.ui.zoom_factor}")
        
        # If no center specified, use center of canvas
        if x is None or x <= 0:
            x = canvas_width / 2
        if y is None or y <= 0:
            y = canvas_height / 2
            
        print(f"  Zoom center: ({x}, {y})")
        
        # Calculate new zoom factor
        new_zoom = self.ui.zoom_factor * factor
        
        # Constrain zoom factor
        new_zoom = max(self.ui.zoom_min, min(self.ui.zoom_max, new_zoom))
        
        # If zoom hasn't changed, do nothing
        if new_zoom == self.ui.zoom_factor:
            print("  No zoom change needed")
            return
            
        print(f"  New zoom: {new_zoom}")
        
        # When zooming, we are no longer in "fit to window" mode
        self.ui.fit_to_window = False
        
        # Store the point position before zoom (in image coordinates)
        p_x = (x - self.ui.canvas_offset_x) / self.ui.zoom_factor
        p_y = (y - self.ui.canvas_offset_y) / self.ui.zoom_factor
        
        # Update zoom factor
        self.ui.zoom_factor = new_zoom
        
        # Calculate new offset to keep zoom centered on mouse
        self.ui.canvas_offset_x = x - (p_x * new_zoom)
        self.ui.canvas_offset_y = y - (p_y * new_zoom)
        
        print(f"  New offset: ({self.ui.canvas_offset_x}, {self.ui.canvas_offset_y})")
        
        # Update all canvas items
        self.update_all_canvas_items()
        
        # Update status
        self.ui.status_var.set(f"Zoom: {self.ui.zoom_factor:.2f}x")
    
    def reset_view(self):
        """Reset view to fit the image within the canvas"""
        # Get canvas and image dimensions
        canvas_width = self.ui.canvas.winfo_width()
        canvas_height = self.ui.canvas.winfo_height()
        img_height, img_width = self.ui.display_frame.shape[:2]
        
        # Calculate scaling factor to fit within canvas (with a small margin)
        margin = 20  # Pixels of margin
        width_scale = (canvas_width - margin) / img_width
        height_scale = (canvas_height - margin) / img_height
        
        # Use the smaller scale to ensure entire image fits
        # Don't scale up small images beyond 1.0 (their actual size)
        fit_scale = min(width_scale, height_scale, 1.0)
        
        # Store the fit information for consistent behavior across frames
        self.ui.fit_to_window = True
        self.ui.fit_scale = fit_scale
        self.ui.last_canvas_width = canvas_width
        self.ui.last_canvas_height = canvas_height
        
        # Print debug info
        print(f"Canvas size: {canvas_width}x{canvas_height}")
        print(f"Image size: {img_width}x{img_height}")
        print(f"Fit scale: {fit_scale} (width_scale={width_scale}, height_scale={height_scale})")
        
        # Update zoom factor
        self.ui.zoom_factor = fit_scale
        
        # Center the image in the canvas
        self.ui.canvas_offset_x = max(0, (canvas_width - (img_width * fit_scale)) / 2)
        self.ui.canvas_offset_y = max(0, (canvas_height - (img_height * fit_scale)) / 2)
        
        print(f"New offset: {self.ui.canvas_offset_x}, {self.ui.canvas_offset_y}")
        
        # Update all canvas items
        self.update_all_canvas_items()
        
        # Update status
        self.ui.status_var.set(f"View reset to fit (zoom: {fit_scale:.2f}x)")