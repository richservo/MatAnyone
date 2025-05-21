"""
# chunk_visualization.py - v1.1735054800
# Updated: December 24, 2024
# Changes in this version:
# - Split off from chunk_optimizer.py to improve code organization and maintainability
# - Contains visualization functionality for chunk analysis and debugging
# - Maintains all existing functionality while improving code structure
"""

import os
import cv2
import numpy as np
import traceback


def create_composite_masks(chunk_outputs, width, height, total_frames, temp_dir=None, sample_interval=30):
    """
    Create a composite mask from all chunks to visualize contribution areas
    
    Args:
        chunk_outputs: List of dictionaries with chunk outputs
        width: Output video width
        height: Output video height
        total_frames: Total number of frames
        temp_dir: Directory to save debug images
        sample_interval: Interval for sampling frames for analysis
    
    Returns:
        Path to composite mask image
    """
    try:
        if not temp_dir:
            return None
            
        print("Creating composite mask visualization...")
        
        # Create a canvas with different colors for each chunk
        composite = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Generate a color for each chunk
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (128, 128, 0),  # Olive
            (128, 0, 128),  # Purple
            (0, 128, 128),  # Teal
            (255, 165, 0),  # Orange
            (255, 192, 203) # Pink
        ]
        
        # Extend with random colors if needed
        while len(colors) < len(chunk_outputs):
            colors.append(tuple(np.random.randint(0, 256, 3).tolist()))
        
        # Create a dictionary to store video captures for each chunk
        chunk_caps = {}
        for i, chunk in enumerate(chunk_outputs):
            pha_cap = cv2.VideoCapture(chunk['pha_path'])
            if pha_cap.isOpened():
                chunk_caps[i] = {
                    'cap': pha_cap,
                    'x_range': chunk['x_range'],
                    'y_range': chunk['y_range'],
                    'color': colors[i]
                }
            else:
                print(f"Warning: Could not open alpha video for chunk {i}")
        
        # Determine number of frames to analyze
        frames_to_analyze = min(total_frames, 10 * sample_interval)
        
        # Create contribution mask for each chunk
        contribution_masks = {}
        for i in chunk_caps:
            contribution_masks[i] = np.zeros((height, width), dtype=np.uint8)
        
        # For every Nth frame (using sample_interval)
        for frame_idx in range(0, frames_to_analyze, sample_interval):
            print(f"Analyzing frame {frame_idx}/{frames_to_analyze}...")
            
            # For each chunk, extract the current frame
            for i, chunk_data in chunk_caps.items():
                cap = chunk_data['cap']
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Get chunk position
                start_x, end_x = chunk_data['x_range']
                start_y, end_y = chunk_data['y_range']
                
                # Extract alpha channel (convert to grayscale if RGB)
                if len(frame.shape) == 3:
                    if np.all(frame[:,:,0] == frame[:,:,1]) and np.all(frame[:,:,1] == frame[:,:,2]):
                        alpha = frame[:,:,0]
                    else:
                        alpha = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    alpha = frame
                
                # Find areas with significant alpha
                mask = alpha > 10
                
                # Update contribution mask for this chunk
                h, w = mask.shape[:2]
                safe_end_x = min(start_x + w, width)
                safe_end_y = min(start_y + h, height)
                width_to_use = safe_end_x - start_x
                height_to_use = safe_end_y - start_y
                
                # Only update if dimensions are valid
                if width_to_use > 0 and height_to_use > 0:
                    # Make sure the region and mask dimensions match
                    mask_region = mask[:height_to_use, :width_to_use]
                    
                    # Update contribution region for significant alpha areas
                    region_slice_y = slice(start_y, start_y + height_to_use)
                    region_slice_x = slice(start_x, start_x + width_to_use)
                    
                    # Only update if region is within bounds
                    if (region_slice_y.start < height and region_slice_x.start < width and
                        region_slice_y.stop > 0 and region_slice_x.stop > 0):
                        
                        # Apply the mask to update the contribution mask
                        contribution_masks[i][region_slice_y, region_slice_x][mask_region] = 255
        
        # Create the composite image
        # For each pixel, show the chunk that contributes the most
        for i, mask in contribution_masks.items():
            # Add this chunk's contribution to the composite image
            non_zero_indices = np.where(mask > 0)
            if len(non_zero_indices[0]) > 0:
                color = chunk_caps[i]['color']
                for c in range(3):
                    composite[non_zero_indices[0], non_zero_indices[1], c] = color[c]
        
        # Create an overall visualization of all chunks combined
        # This shows where chunks overlap and how coverage is distributed
        chunk_counts = np.zeros((height, width), dtype=np.uint8)
        all_contributions = np.zeros((height, width), dtype=np.uint8)
        
        for mask in contribution_masks.values():
            # Increment chunk count where this chunk contributes
            chunk_counts[mask > 0] += 1
            # Update overall contribution
            all_contributions = np.maximum(all_contributions, mask)
        
        # Save chunk count visualization (heat map of overlap)
        # Scale to 0-255 for better visibility
        if np.max(chunk_counts) > 0:
            normalized_counts = (chunk_counts.astype(np.float32) / np.max(chunk_counts) * 255).astype(np.uint8)
        else:
            normalized_counts = np.zeros((height, width), dtype=np.uint8)
            
        # Create a colormap visualization
        heatmap = cv2.applyColorMap(normalized_counts, cv2.COLORMAP_JET)
        heatmap_path = os.path.join(temp_dir, "chunk_overlap_heatmap.png")
        cv2.imwrite(heatmap_path, heatmap)
        
        # Save all contributions (union of all chunk masks)
        all_contrib_path = os.path.join(temp_dir, "all_chunk_contributions.png")
        cv2.imwrite(all_contrib_path, all_contributions)
        
        # Close all video captures
        for i in chunk_caps:
            chunk_caps[i]['cap'].release()
        
        # Save the composite image
        composite_path = os.path.join(temp_dir, "chunk_composite_mask.png")
        cv2.imwrite(composite_path, composite)
        
        print(f"Composite mask visualization created at {composite_path}")
        return composite_path
        
    except Exception as e:
        print(f"Error creating composite mask: {str(e)}")
        traceback.print_exc()
        return None


def create_maximized_alpha_mask(chunk_outputs, width, height, frame_index=0, temp_dir=None):
    """
    Generate a maximized alpha mask from all chunks for a specific frame.
    This represents the full potential mask if we used maximum blending everywhere.
    
    Args:
        chunk_outputs: List of dictionaries with chunk outputs
        width: Output video width
        height: Output video height
        frame_index: Frame index to process
        temp_dir: Directory to save debug images
    
    Returns:
        Maximum combined alpha mask
    """
    try:
        # Create canvas for combined mask
        max_mask = np.zeros((height, width), dtype=np.float32)
        
        # Process each chunk
        for i, chunk in enumerate(chunk_outputs):
            # Open alpha video
            pha_cap = cv2.VideoCapture(chunk['pha_path'])
            if not pha_cap.isOpened():
                print(f"Warning: Could not open alpha video for chunk {i}")
                continue
            
            # Seek to frame
            pha_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = pha_cap.read()
            pha_cap.release()
            
            if not ret:
                continue
            
            # Get chunk position
            start_x, end_x = chunk['x_range']
            start_y, end_y = chunk['y_range']
            
            # Extract alpha channel (convert to grayscale if RGB)
            if len(frame.shape) == 3:
                if np.all(frame[:,:,0] == frame[:,:,1]) and np.all(frame[:,:,1] == frame[:,:,2]):
                    alpha = frame[:,:,0].astype(np.float32)
                else:
                    alpha = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                alpha = frame.astype(np.float32)
            
            # Extract region boundaries
            region_height, region_width = alpha.shape[:2]
            safe_end_x = min(start_x + region_width, width)
            safe_end_y = min(start_y + region_height, height)
            width_to_use = safe_end_x - start_x
            height_to_use = safe_end_y - start_y
            
            # Skip if invalid dimensions
            if width_to_use <= 0 or height_to_use <= 0:
                continue
                
            # Extract the portion of alpha to use
            alpha_region = alpha[:height_to_use, :width_to_use]
            
            # Create region slices
            region_slice_y = slice(start_y, start_y + height_to_use)
            region_slice_x = slice(start_x, start_x + width_to_use)
            
            # Update max mask with maximum alpha values
            current_max = max_mask[region_slice_y, region_slice_x]
            max_mask[region_slice_y, region_slice_x] = np.maximum(current_max, alpha_region)
        
        # Save debug image if requested
        if temp_dir:
            debug_path = os.path.join(temp_dir, f"max_combined_alpha_frame_{frame_index}.png")
            cv2.imwrite(debug_path, max_mask.astype(np.uint8))
        
        return max_mask
        
    except Exception as e:
        print(f"Error creating maximized alpha mask: {str(e)}")
        traceback.print_exc()
        return None
