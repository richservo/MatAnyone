"""
# chunk_mask_propagation.py - v1.1735054800
# Updated: December 24, 2024
# Changes in this version:
# - Split off from chunk_optimizer.py to improve code organization and maintainability
# - Contains mask propagation functionality between overlapping chunks
# - Maintains all existing functionality while improving code structure
"""

import os
import cv2
import numpy as np
import traceback


def propagate_mask_data(chunk_outputs, frame_index, temp_dir=None):
    """
    Explicitly propagates mask data between overlapping chunks to ensure
    all chunks have the best possible mask data across all overlapping regions.
    
    Args:
        chunk_outputs: List of dictionaries with chunk outputs
        frame_index: Current frame index
        temp_dir: Directory to save debug visualizations
    
    Returns:
        Dictionary of enhanced chunk data with updated mask frames
    """
    try:
        print(f"Propagating mask data between all overlapping chunks for frame {frame_index}...")
        
        # Step 1: Extract frame and position data from each chunk
        chunk_data = []
        chunk_positions = {}
        original_masks = {}
        
        for i, chunk in enumerate(chunk_outputs):
            # Open the alpha video to extract mask
            pha_cap = cv2.VideoCapture(chunk['pha_path'])
            if not pha_cap.isOpened():
                print(f"Warning: Could not open alpha video for chunk {i}")
                continue
                
            # Get the specified frame
            pha_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = pha_cap.read()
            pha_cap.release()
            
            if not ret:
                print(f"Warning: Could not read frame {frame_index} from chunk {i}")
                continue
                
            # Get chunk position
            start_x, end_x = chunk['x_range']
            start_y, end_y = chunk['y_range']
            
            # Store position data
            chunk_positions[i] = {
                'x_range': (start_x, end_x),
                'y_range': (start_y, end_y)
            }
            
            # Extract grayscale mask
            if len(frame.shape) == 3:
                if np.all(frame[:,:,0] == frame[:,:,1]) and np.all(frame[:,:,1] == frame[:,:,2]):
                    mask = frame[:,:,0].astype(np.float32)
                else:
                    mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                mask = frame.astype(np.float32)
                
            # Store original mask
            original_masks[i] = mask.copy()
            
            # Store chunk information
            chunk_data.append({
                'index': i,
                'mask': mask,
                'x_range': (start_x, end_x),
                'y_range': (start_y, end_y)
            })
        
        # Step 2: Find all overlapping pairs
        overlap_pairs = []
        
        for i in range(len(chunk_data)):
            for j in range(i + 1, len(chunk_data)):
                chunk1 = chunk_data[i]
                chunk2 = chunk_data[j]
                
                # Get positions
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
                    overlap_pairs.append({
                        'chunks': (chunk1['index'], chunk2['index']),
                        'region': ((overlap_x1, overlap_y1), (overlap_x2, overlap_y2))
                    })
        
        # Step 3: Create a full grid overlap topology map
        if frame_index == 0:  # Only show for first frame to reduce clutter
            print(f"Found {len(overlap_pairs)} overlapping chunk pairs")
            
            # Count how many chunks each chunk overlaps with
            overlap_counts = {}
            for pair in overlap_pairs:
                c1, c2 = pair['chunks']
                
                if c1 not in overlap_counts:
                    overlap_counts[c1] = set()
                if c2 not in overlap_counts:
                    overlap_counts[c2] = set()
                    
                overlap_counts[c1].add(c2)
                overlap_counts[c2].add(c1)
            
            # Print chunk overlap topology
            print("Chunk overlap topology:")
            for chunk_idx, neighbors in overlap_counts.items():
                print(f"  Chunk {chunk_idx} overlaps with {len(neighbors)} chunks: {sorted(neighbors)}")
        
        # Step 4: Create enhanced masks for each chunk (starting with original)
        enhanced_masks = {}
        for chunk_idx, original_mask in original_masks.items():
            enhanced_masks[chunk_idx] = original_mask.copy()
        
        # Step 5: Create a global max mask combining all chunks
        # This allows us to ensure maximum propagation of all data
        width = max([chunk['x_range'][1] for chunk in chunk_data])
        height = max([chunk['y_range'][1] for chunk in chunk_data])
        global_max_mask = np.zeros((height, width), dtype=np.float32)
        
        # Fill global max mask with data from all chunks
        for chunk_idx, mask in enhanced_masks.items():
            start_x, end_x = chunk_positions[chunk_idx]['x_range']
            start_y, end_y = chunk_positions[chunk_idx]['y_range']
            
            # Ensure dimensions are compatible
            mask_height, mask_width = mask.shape
            if mask_height > (end_y - start_y) or mask_width > (end_x - start_x):
                # Too big - need to crop
                safe_height = min(mask_height, end_y - start_y)
                safe_width = min(mask_width, end_x - start_x)
                mask_to_use = mask[:safe_height, :safe_width]
            else:
                mask_to_use = mask
            
            # Calculate region slices
            region_height, region_width = mask_to_use.shape
            region_slice_y = slice(start_y, start_y + region_height)
            region_slice_x = slice(start_x, start_x + region_width)
            
            # Update global max mask - use maximum to capture all data
            global_max_mask[region_slice_y, region_slice_x] = np.maximum(
                global_max_mask[region_slice_y, region_slice_x], mask_to_use
            )
            
        # Step 6: Propagate the global max data back to each chunk in overlapping regions
        # This is the key to ensuring all chunks have access to all mask data
        for chunk_idx, mask in enhanced_masks.items():
            start_x, end_x = chunk_positions[chunk_idx]['x_range']
            start_y, end_y = chunk_positions[chunk_idx]['y_range']
            
            # Figure out which region of the global mask applies to this chunk
            region_slice_y = slice(start_y, min(start_y + mask.shape[0], height))
            region_slice_x = slice(start_x, min(start_x + mask.shape[1], width))
            
            # Extract the region from the global max mask
            global_region = global_max_mask[region_slice_y, region_slice_x]
            
            # Calculate size differences to handle different shapes
            global_height, global_width = global_region.shape
            mask_height, mask_width = mask.shape
            
            # Only proceed if the mask dimensions can accommodate the global region
            if global_height <= mask_height and global_width <= mask_width:
                # Local portion of chunk mask
                local_y_slice = slice(0, global_height)
                local_x_slice = slice(0, global_width)
                
                # Get local portion of chunk mask
                local_mask = mask[local_y_slice, local_x_slice]
                
                # Find pixels where global max is higher
                update_mask = global_region > local_mask
                
                # Update those pixels
                if np.any(update_mask):
                    mask[local_y_slice, local_x_slice][update_mask] = global_region[update_mask]
        
        # Step 7: Create debug visualizations
        if temp_dir and frame_index == 0:  # Only for first frame
            # Save the global max mask
            global_max_path = os.path.join(temp_dir, "global_max_mask_propagated.png")
            cv2.imwrite(global_max_path, global_max_mask.astype(np.uint8))
            
            # For each chunk, save before/after comparison
            for chunk_idx in enhanced_masks:
                # Skip if original mask missing
                if chunk_idx not in original_masks:
                    continue
                    
                original_mask = original_masks[chunk_idx]
                enhanced_mask = enhanced_masks[chunk_idx]
                
                # Save original mask
                orig_path = os.path.join(temp_dir, f"chunk_{chunk_idx}_original_mask_propagated.png")
                cv2.imwrite(orig_path, original_mask.astype(np.uint8))
                
                # Save enhanced mask
                enhanced_path = os.path.join(temp_dir, f"chunk_{chunk_idx}_enhanced_mask_propagated.png")
                cv2.imwrite(enhanced_path, enhanced_mask.astype(np.uint8))
                
                # Create difference image
                diff = enhanced_mask - original_mask
                diff = np.clip(diff, 0, 255).astype(np.uint8)
                diff_path = os.path.join(temp_dir, f"chunk_{chunk_idx}_mask_diff_propagated.png")
                cv2.imwrite(diff_path, diff)
                
                # Create a colorized visualization
                diff_viz = np.zeros((original_mask.shape[0], original_mask.shape[1], 3), dtype=np.uint8)
                diff_viz[:,:,1] = diff  # Green channel for added content
                diff_viz[:,:,2] = original_mask.astype(np.uint8)  # Blue channel for original
                viz_path = os.path.join(temp_dir, f"chunk_{chunk_idx}_propagation_viz.png")
                cv2.imwrite(viz_path, diff_viz)
                
                # Calculate enhancement statistics
                pixels_enhanced = np.sum(diff > 0)
                if pixels_enhanced > 0:
                    print(f"Enhanced chunk {chunk_idx} with {pixels_enhanced} pixels across all overlaps")
            
            # Create a visualization of the chunk layout
            if chunk_positions:
                # Find overall canvas size
                max_x = max([chunk_positions[i]['x_range'][1] for i in chunk_positions])
                max_y = max([chunk_positions[i]['y_range'][1] for i in chunk_positions])
                
                # Create canvas for visualization
                grid_viz = np.zeros((max_y, max_x, 3), dtype=np.uint8)
                
                # Define colors
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
                
                # Draw each chunk
                for chunk_idx, position in chunk_positions.items():
                    start_x, end_x = position['x_range']
                    start_y, end_y = position['y_range']
                    
                    # Choose color
                    if chunk_idx < len(colors):
                        color = colors[chunk_idx]
                    else:
                        color = tuple(np.random.randint(0, 256, 3).tolist())
                    
                    # Fill region with semi-transparent color
                    alpha = 0.3
                    region = grid_viz[start_y:end_y, start_x:end_x].copy()
                    colored = np.ones_like(region) * np.array(color, dtype=np.uint8)
                    grid_viz[start_y:end_y, start_x:end_x] = cv2.addWeighted(region, 1-alpha, colored, alpha, 0)
                    
                    # Draw boundary
                    cv2.rectangle(grid_viz, (start_x, start_y), (end_x, end_y), color, 2)
                    
                    # Add label
                    cv2.putText(grid_viz, str(chunk_idx), (start_x + 10, start_y + 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Draw overlap regions
                for overlap_info in overlap_pairs:
                    # Get overlap region
                    ((x1, y1), (x2, y2)) = overlap_info['region']
                    
                    # Draw semi-transparent white overlay
                    alpha = 0.2
                    region = grid_viz[y1:y2, x1:x2].copy()
                    white = np.ones_like(region) * 255
                    grid_viz[y1:y2, x1:x2] = cv2.addWeighted(region, 1-alpha, white, alpha, 0)
                    
                    # Draw boundary
                    cv2.rectangle(grid_viz, (x1, y1), (x2, y2), (255, 255, 255), 1)
                
                # Save grid visualization
                grid_path = os.path.join(temp_dir, "chunk_grid_propagation.png")
                cv2.imwrite(grid_path, grid_viz)
        
        # Return the enhanced masks
        return {
            'enhanced_masks': enhanced_masks,
            'original_masks': original_masks,
            'chunk_positions': chunk_positions,
            'overlap_pairs': overlap_pairs,
            'global_max_mask': global_max_mask
        }
    
    except Exception as e:
        print(f"Error propagating mask data: {str(e)}")
        traceback.print_exc()
        return None
