"""
# chunk_mask_analysis.py - v1.1735054800
# Updated: December 24, 2024
# Changes in this version:
# - Split off from chunk_optimizer.py to improve code organization and maintainability
# - Contains the optimize_chunk_masks function and related mask analysis functionality
# - Maintains all existing functionality while improving code structure
"""

import os
import cv2
import numpy as np
import time
import traceback


def optimize_chunk_masks(chunk_outputs, frame_index, temp_dir=None):
    """
    Analyze overlapping areas between chunks and create an optimized 
    priority map to ensure mask information is preserved
    
    Args:
        chunk_outputs: List of dictionaries with chunk data
        frame_index: Current frame index being processed
        temp_dir: Directory to save debug images
    
    Returns:
        Dictionary with optimized mask data for each chunk
    """
    try:
        # Extract current frame from each chunk's alpha video
        chunk_frames = []
        
        for i, chunk in enumerate(chunk_outputs):
            pha_cap = cv2.VideoCapture(chunk['pha_path'])
            
            # Seek to the current frame
            pha_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            
            # Read the frame
            ret, pha_frame = pha_cap.read()
            pha_cap.release()
            
            if not ret:
                print(f"Warning: Could not read frame {frame_index} from chunk {i}")
                chunk_frames.append(None)
                continue
            
            # Get chunk position and dimensions
            start_x, end_x = chunk['x_range']
            start_y, end_y = chunk['y_range']
            
            # Store frame with position information
            chunk_frames.append({
                'frame': pha_frame,
                'x_range': (start_x, end_x),
                'y_range': (start_y, end_y),
                'index': i
            })
        
        # If only one chunk or no valid frames, no optimization needed
        if len(chunk_frames) <= 1 or all(f is None for f in chunk_frames):
            return None
            
        # Determine the full output dimensions (max of all chunks)
        max_x = max([f['x_range'][1] for f in chunk_frames if f is not None])
        max_y = max([f['y_range'][1] for f in chunk_frames if f is not None])
        
        # Create empty canvas for analysis
        combined_mask = np.zeros((max_y, max_x), dtype=np.float32)
        priority_map = -np.ones((max_y, max_x), dtype=np.int32)  # -1 means no chunk
        
        # Create canvas for maximum possible mask content
        max_possible_mask = np.zeros((max_y, max_x), dtype=np.float32)
        
        # First pass: Build overlap topology map
        chunk_topology = {}  # Maps chunk index -> list of overlapping chunk indices
        overlap_regions = []  # List of overlap region information
        
        # Initialize topology map for all chunks
        for i in range(len(chunk_outputs)):
            chunk_topology[i] = []
        
        # Build the chunk topology map
        print(f"Building chunk topology map for {len(chunk_frames)} chunks")
        for i, chunk1 in enumerate(chunk_frames):
            if chunk1 is None:
                continue
                
            for j in range(i+1, len(chunk_frames)):
                chunk2 = chunk_frames[j]
                if chunk2 is None:
                    continue
                
                # Get chunk regions
                start_x1, end_x1 = chunk1['x_range']
                start_y1, end_y1 = chunk1['y_range']
                start_x2, end_x2 = chunk2['x_range']
                start_y2, end_y2 = chunk2['y_range']
                
                # Check for overlap
                if (start_x1 < end_x2 and end_x1 > start_x2 and 
                    start_y1 < end_y2 and end_y1 > start_y2):
                    
                    # Calculate overlap region
                    overlap_x1 = max(start_x1, start_x2)
                    overlap_y1 = max(start_y1, start_y2)
                    overlap_x2 = min(end_x1, end_x2)
                    overlap_y2 = min(end_y1, end_y2)
                    
                    # Store overlap information
                    overlap_info = {
                        'chunks': (chunk1['index'], chunk2['index']),
                        'region': ((overlap_x1, overlap_y1), (overlap_x2, overlap_y2))
                    }
                    overlap_regions.append(overlap_info)
                    
                    # Update topology map
                    chunk_topology[chunk1['index']].append(chunk2['index'])
                    chunk_topology[chunk2['index']].append(chunk1['index'])
        
        # Display the topology map
        if frame_index == 0:  # Only show for first frame to reduce spam
            print("Chunk overlap topology:")
            for chunk_idx, neighbors in chunk_topology.items():
                if chunk_frames[chunk_idx] is not None:
                    print(f"  Chunk {chunk_idx} overlaps with chunks: {neighbors}")
        
        # Second pass: Extract mask data from all chunks
        chunk_masks = {}  # Dictionary of original chunk masks
        chunk_positions = {}  # Dictionary of chunk positions
        
        for chunk_data in chunk_frames:
            if chunk_data is None:
                continue
                
            chunk_idx = chunk_data['index']
            start_x, end_x = chunk_data['x_range']
            start_y, end_y = chunk_data['y_range']
            
            # Store positions
            chunk_positions[chunk_idx] = {
                'x_range': (start_x, end_x),
                'y_range': (start_y, end_y)
            }
            
            # Get original mask
            frame = chunk_data['frame']
            
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                if np.all(frame[:,:,0] == frame[:,:,1]) and np.all(frame[:,:,1] == frame[:,:,2]):
                    mask = frame[:,:,0].astype(np.float32)
                else:
                    mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                mask = frame.astype(np.float32)
            
            # Store the original mask
            chunk_masks[chunk_idx] = mask.copy()
            
            # Extract region for max possible mask
            region_height, region_width = mask.shape[:2]
            safe_end_x = min(start_x + region_width, max_x)
            safe_end_y = min(start_y + region_height, max_y)
            width_to_use = safe_end_x - start_x
            height_to_use = safe_end_y - start_y
            
            # Set up region slices
            region_slice_y = slice(start_y, start_y + height_to_use)
            region_slice_x = slice(start_x, start_x + width_to_use)
            
            # Get current max mask in this region
            current_max = max_possible_mask[region_slice_y, region_slice_x]
            
            # Update with maximum values
            max_region = mask[:height_to_use, :width_to_use]
            max_possible_mask[region_slice_y, region_slice_x] = np.maximum(current_max, max_region)
            
            # Define threshold for considering a pixel as "masked"
            mask_threshold = 10
            
            # Create a mask for pixels above threshold
            masked_pixels = max_region > mask_threshold
            
            # Update priority map for this chunk's significant content
            # Only override if this chunk has content and existing doesn't or has less
            current_priority = priority_map[region_slice_y, region_slice_x]
            current_mask_value = combined_mask[region_slice_y, region_slice_x]
            
            # Update condition: either new mask value is higher or no previous assignment (-1)
            update_condition = (masked_pixels & 
                              ((max_region > current_mask_value) | (current_priority == -1)))
            
            # Apply updates where the condition is met
            priority_map[region_slice_y, region_slice_x][update_condition] = chunk_idx
            combined_mask[region_slice_y, region_slice_x][update_condition] = max_region[update_condition]
        
        # Create a global maximum mask combining all chunks' mask information
        # This allows us to get the absolute maximum mask content at every pixel
        global_max_mask = np.zeros((max_y, max_x), dtype=np.float32)
        
        for chunk_idx, mask in chunk_masks.items():
            start_x, end_x = chunk_positions[chunk_idx]['x_range']
            start_y, end_y = chunk_positions[chunk_idx]['y_range']
            
            mask_height, mask_width = mask.shape
            height_to_use = min(mask_height, end_y - start_y)
            width_to_use = min(mask_width, end_x - start_x)
            
            # Set up region slices
            region_slice_y = slice(start_y, start_y + height_to_use)
            region_slice_x = slice(start_x, start_x + width_to_use)
            
            # Update global maximum mask
            mask_region = mask[:height_to_use, :width_to_use]
            global_max_mask[region_slice_y, region_slice_x] = np.maximum(
                global_max_mask[region_slice_y, region_slice_x],
                mask_region
            )
        
        # Create enhanced masks for each chunk with propagated mask information
        enhanced_masks = {}
        
        # Start with the original masks
        for chunk_idx, original_mask in chunk_masks.items():
            enhanced_masks[chunk_idx] = original_mask.copy()
        
        # Propagate maximum content to all chunks in overlapping regions
        for overlap_info in overlap_regions:
            chunk_indices = overlap_info['chunks']
            ((overlap_x1, overlap_y1), (overlap_x2, overlap_y2)) = overlap_info['region']
            
            # Get global maximum for this overlap region
            global_overlap_slice_y = slice(overlap_y1, overlap_y2)
            global_overlap_slice_x = slice(overlap_x1, overlap_x2)
            global_max_region = global_max_mask[global_overlap_slice_y, global_overlap_slice_x]
            
            # Only process if there's significant content
            mask_threshold = 10
            if np.max(global_max_region) > mask_threshold:
                # Share this maximum content with all chunks that overlap this region
                for chunk_idx in chunk_indices:
                    if chunk_idx not in chunk_positions:
                        continue
                        
                    # Calculate local coordinates for this chunk
                    start_x, end_x = chunk_positions[chunk_idx]['x_range']
                    start_y, end_y = chunk_positions[chunk_idx]['y_range']
                    
                    # Convert global overlap coordinates to local chunk coordinates
                    local_y1 = overlap_y1 - start_y
                    local_x1 = overlap_x1 - start_x
                    
                    # Calculate dimensions ensuring they're valid for this chunk
                    overlap_height = overlap_y2 - overlap_y1
                    overlap_width = overlap_x2 - overlap_x1
                    
                    # Ensure we don't exceed boundaries
                    if (local_y1 >= 0 and local_x1 >= 0 and 
                        local_y1 + overlap_height <= enhanced_masks[chunk_idx].shape[0] and
                        local_x1 + overlap_width <= enhanced_masks[chunk_idx].shape[1]):
                        
                        # Create local slices
                        local_slice_y = slice(local_y1, local_y1 + overlap_height)
                        local_slice_x = slice(local_x1, local_x1 + overlap_width)
                        
                        # Update with global maximum (propagates all mask content)
                        # Use maximum blending to ensure we never lose mask information
                        enhanced_masks[chunk_idx][local_slice_y, local_slice_x] = np.maximum(
                            enhanced_masks[chunk_idx][local_slice_y, local_slice_x],
                            global_max_region
                        )
        
        # For debugging, save visualizations of original and enhanced masks
        if temp_dir and frame_index == 0:  # Only for first frame
            # Create a colorful visualization of the priority map
            debug_priority_image = np.zeros((max_y, max_x, 3), dtype=np.uint8)
            
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
            
            for i in range(len(chunk_outputs)):
                mask = (priority_map == i)
                if i < len(colors):
                    color = colors[i]
                else:
                    # Generate a random color for additional chunks
                    color = tuple(np.random.randint(0, 256, 3).tolist())
                
                # Make sure the mask is properly shaped for broadcasting
                if mask.shape[0] > 0 and mask.shape[1] > 0:
                    # Apply color where this chunk is prioritized
                    for c in range(3):
                        debug_priority_image[:,:,c][mask] = color[c]
            
            debug_path = os.path.join(temp_dir, f"priority_map_frame_{frame_index}.png")
            cv2.imwrite(debug_path, debug_priority_image)
            
            # Save the combined mask
            mask_debug_path = os.path.join(temp_dir, f"combined_mask_frame_{frame_index}.png")
            cv2.imwrite(mask_debug_path, combined_mask.astype(np.uint8))
            
            # Save the maximum possible mask
            max_mask_path = os.path.join(temp_dir, f"max_possible_mask_frame_{frame_index}.png")
            cv2.imwrite(max_mask_path, max_possible_mask.astype(np.uint8))
            
            # Save the global maximum mask
            global_max_path = os.path.join(temp_dir, f"global_max_mask_frame_{frame_index}.png")
            cv2.imwrite(global_max_path, global_max_mask.astype(np.uint8))
            
            # Save before/after for each chunk
            for chunk_idx in enhanced_masks:
                original_mask = chunk_masks[chunk_idx]
                enhanced_mask = enhanced_masks[chunk_idx]
                
                # Save original mask
                orig_path = os.path.join(temp_dir, f"chunk_{chunk_idx}_original_mask.png")
                cv2.imwrite(orig_path, original_mask.astype(np.uint8))
                
                # Save enhanced mask
                enhanced_path = os.path.join(temp_dir, f"chunk_{chunk_idx}_enhanced_mask.png")
                cv2.imwrite(enhanced_path, enhanced_mask.astype(np.uint8))
                
                # Create a diff image to highlight changes
                diff = enhanced_mask - original_mask
                diff = np.clip(diff, 0, 255).astype(np.uint8)
                diff_path = os.path.join(temp_dir, f"chunk_{chunk_idx}_mask_diff.png")
                cv2.imwrite(diff_path, diff)
                
                # Create a colorized visualization of the differences
                diff_viz = np.zeros((original_mask.shape[0], original_mask.shape[1], 3), dtype=np.uint8)
                diff_viz[:,:,1] = diff  # Green channel for added content
                diff_viz[:,:,2] = original_mask.astype(np.uint8)  # Blue channel for original content
                
                diff_viz_path = os.path.join(temp_dir, f"chunk_{chunk_idx}_enhancement_viz.png")
                cv2.imwrite(diff_viz_path, diff_viz)
                
                # Count pixels enhanced
                pixels_enhanced = np.sum(diff > 0)
                if pixels_enhanced > 0:
                    print(f"Enhanced chunk {chunk_idx} with {pixels_enhanced} pixels from overlapping chunks")
        
        # Create a grid visualization showing all chunks and their overlaps
        if temp_dir and frame_index == 0:
            # Create a canvas with unique colors for each chunk
            grid_viz = np.zeros((max_y, max_x, 3), dtype=np.uint8)
            
            # Draw each chunk with semi-transparent color
            for i, chunk_data in enumerate(chunk_frames):
                if chunk_data is None:
                    continue
                    
                start_x, end_x = chunk_data['x_range']
                start_y, end_y = chunk_data['y_range']
                
                # Choose color
                if i < len(colors):
                    color = colors[i]
                else:
                    color = tuple(np.random.randint(0, 256, 3).tolist())
                
                # Create semi-transparent overlay
                alpha = 0.3
                grid_viz[start_y:end_y, start_x:end_x, 0] = int(grid_viz[start_y:end_y, start_x:end_x, 0] * (1-alpha) + color[0] * alpha)
                grid_viz[start_y:end_y, start_x:end_x, 1] = int(grid_viz[start_y:end_y, start_x:end_x, 1] * (1-alpha) + color[1] * alpha)
                grid_viz[start_y:end_y, start_x:end_x, 2] = int(grid_viz[start_y:end_y, start_x:end_x, 2] * (1-alpha) + color[2] * alpha)
                
                # Draw boundary
                cv2.rectangle(grid_viz, (start_x, start_y), (end_x, end_y), color, 2)
                
                # Add label
                cv2.putText(grid_viz, str(i), (start_x + 10, start_y + 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Highlight overlap regions
            for overlap_info in overlap_regions:
                ((x1, y1), (x2, y2)) = overlap_info['region']
                # Draw semi-transparent white overlay for overlap
                alpha = 0.2
                overlay = grid_viz[y1:y2, x1:x2].copy()
                white = np.ones_like(overlay) * 255
                grid_viz[y1:y2, x1:x2] = cv2.addWeighted(overlay, 1-alpha, white, alpha, 0)
                
                # Draw boundary for overlap
                cv2.rectangle(grid_viz, (x1, y1), (x2, y2), (255, 255, 255), 1)
            
            # Save grid visualization
            grid_path = os.path.join(temp_dir, "chunk_grid_visualization.png")
            cv2.imwrite(grid_path, grid_viz)
        
        # Return the optimized data
        return {
            'priority_map': priority_map,
            'combined_mask': combined_mask,
            'enhanced_masks': enhanced_masks,
            'max_possible_mask': max_possible_mask,
            'global_max_mask': global_max_mask,
            'total_chunks': len([c for c in chunk_frames if c is not None]),
            'chunk_topology': chunk_topology,
            'overlap_regions': overlap_regions
        }
        
    except Exception as e:
        print(f"Error in optimize_chunk_masks: {str(e)}")
        traceback.print_exc()
        return None
