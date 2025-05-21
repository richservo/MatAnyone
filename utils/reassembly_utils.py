# reassembly_utils.py - v1.1716987123
# Updated: Tuesday, May 21, 2025 at 13:58:43 PST
# Changes in this version:
# - Fixed critical issue with max_alpha blend method not working correctly during chunk reassembly
# - Added enhanced support for different blend methods in the chunk blending process
# - Improved edge handling for max_alpha blending to prevent gray artifacts
# - Added special optimization for max_alpha that prioritizes higher alpha values at every pixel
# - Enhanced debugging visualization for different blend methods

"""
Reassembly utilities for MatAnyone video processing.
Contains functions for reassembling processed video chunks.
"""

import os
import cv2
import numpy as np
import time
import traceback

# Import chunk optimizer for maximizing mask information
from chunking.chunk_optimizer import (
    optimize_chunk_masks,
    apply_mask_optimization,
    create_composite_masks,
    create_maximized_alpha_mask,
    propagate_mask_data
)

# Import video utilities
from utils.video_utils import create_high_quality_writer


def reassemble_strip_chunks(chunk_outputs, width, height, fps, frame_count, fgr_output_path, pha_output_path, 
                           blend_method='weighted', temp_dir=None, apply_expanded_mask=False, full_res_mask_dir=None, 
                           maximize_mask=True, video_codec='Auto', video_quality='High', custom_bitrate=None):
    """
    Reassemble horizontal strip chunks into a complete video with mask optimization
    
    Args:
        chunk_outputs: List of dictionaries with chunk outputs
        width: Output video width
        height: Output video height
        fps: Frames per second
        frame_count: Total frame count
        fgr_output_path: Path to save foreground output
        pha_output_path: Path to save alpha output
        blend_method: Method for blending overlapping regions
        temp_dir: Directory to save temporary files (debug images)
        apply_expanded_mask: Whether to multiply output with expanded mask to prevent spill
        full_res_mask_dir: Directory containing full resolution mask frames
        maximize_mask: Whether to use mask optimization to maximize mask content
        video_codec: Video codec to use
        video_quality: Quality preset
        custom_bitrate: Custom bitrate in kbps
        
    Returns:
        Tuple of (fgr_output_path, pha_output_path)
    """
    # Print video quality settings
    if custom_bitrate:
        print(f"Reassembling with video codec: {video_codec}, quality: {video_quality}, custom bitrate: {custom_bitrate} kbps")
    else:
        print(f"Reassembling with video codec: {video_codec}, quality: {video_quality}")
    
    # Check if all chunks have the same dimensions (for auto-chunk mode)
    chunk_dimensions = [(chunk['width'], chunk['height']) for chunk in chunk_outputs]
    all_same_size = len(set(chunk_dimensions)) == 1
    
    if all_same_size:
        print(f"All chunks have identical dimensions: {chunk_dimensions[0][0]}x{chunk_dimensions[0][1]}")
    else:
        print("Warning: Chunks have different dimensions. This may cause inconsistent results.")
        for i, (w, h) in enumerate(chunk_dimensions):
            print(f"  Chunk {i}: {w}x{h}")
    
    # Sort chunks by start_y first, then by start_x
    chunk_outputs.sort(key=lambda x: (x['y_range'][0], x['x_range'][0]))
    
    # Create high-quality output video writers
    fgr_writer = create_high_quality_writer(
        fgr_output_path, fps, width, height,
        video_codec, video_quality, custom_bitrate
    )
    pha_writer = create_high_quality_writer(
        pha_output_path, fps, width, height,
        video_codec, video_quality, custom_bitrate
    )
    
    # Open all chunk videos
    chunk_videos = []
    for chunk in chunk_outputs:
        fgr_cap = cv2.VideoCapture(chunk['fgr_path'])
        pha_cap = cv2.VideoCapture(chunk['pha_path'])
        
        if not fgr_cap.isOpened() or not pha_cap.isOpened():
            print(f"Warning: Could not open output videos for chunk at {chunk['x_range']}, {chunk['y_range']}. Skipping.")
            continue
        
        start_x, end_x = chunk['x_range']
        start_y, end_y = chunk['y_range']
        
        # Verify that chunk dimensions match what we expect
        chunk_width = end_x - start_x
        chunk_height = end_y - start_y
        
        # Get actual video dimensions
        actual_width = int(fgr_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(fgr_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_width != chunk_width or actual_height != chunk_height:
            print(f"Warning: Chunk video dimensions {actual_width}x{actual_height} don't match expected {chunk_width}x{chunk_height}")
            print("This may cause blending issues. Will attempt to resize during reassembly.")
        
        chunk_videos.append({
            'fgr_cap': fgr_cap,
            'pha_cap': pha_cap,
            'x_range': (start_x, end_x),
            'y_range': (start_y, end_y),
            'width': chunk_width,
            'height': chunk_height
        })
    
    # Create blending weights for each chunk with enhanced 2D feathering (even for strips)
    print(f"Creating blending weights with full 2D feathering (blend method: {blend_method})...")
    blend_weights = []
    
    # Special weight function for 2-chunk case
    is_two_chunks = len(chunk_videos) == 2
    
    for i, chunk in enumerate(chunk_videos):
        # Create weight mask (1.0 in center, fade at edges)
        weight = np.ones((chunk['height'], chunk['width']), dtype=np.float32)
        start_x, end_x = chunk['x_range']
        start_y, end_y = chunk['y_range']
        chunk_width = chunk['width']
        chunk_height = chunk['height']
        
        # Calculate feather width for all four edges
        left_feather = right_feather = top_feather = bottom_feather = 0
        
        # Even for strips, check overlaps with ALL other chunks in all directions
        for j, other_chunk in enumerate(chunk_videos):
            if i == j:
                continue
                
            other_start_x, other_end_x = other_chunk['x_range']
            other_start_y, other_end_y = other_chunk['y_range']
            
            # Check if there's any horizontal overlap for left/right edges
            if ((other_start_y < end_y and other_end_y > start_y) or  # Partial vertical overlap
                (other_start_y <= start_y and other_end_y >= end_y) or  # Full vertical overlap
                (other_start_y >= start_y and other_end_y <= end_y)):  # Other fully inside vertically
                
                # Left edge overlap
                if other_end_x > start_x and other_end_x <= end_x:
                    left_overlap = other_end_x - start_x
                    # Make feather wider to ensure smoother transition
                    left_feather = max(left_feather, min(int(left_overlap * 0.9), chunk_width // 3))
                    print(f"Chunk {i}: Left overlap with chunk {j}: {left_overlap}px, feather: {left_feather}px")
                
                # Right edge overlap
                if other_start_x < end_x and other_start_x >= start_x:
                    right_overlap = end_x - other_start_x
                    # Make feather wider to ensure smoother transition
                    right_feather = max(right_feather, min(int(right_overlap * 0.9), chunk_width // 3))
                    print(f"Chunk {i}: Right overlap with chunk {j}: {right_overlap}px, feather: {right_feather}px")
            
            # Check if there's any vertical overlap for top/bottom edges
            if ((other_start_x < end_x and other_end_x > start_x) or  # Partial horizontal overlap
                (other_start_x <= start_x and other_end_x >= end_x) or  # Full horizontal overlap
                (other_start_x >= start_x and other_end_x <= end_x)):  # Other fully inside horizontally
                
                # Top edge overlap
                if other_end_y > start_y and other_end_y <= end_y:
                    top_overlap = other_end_y - start_y
                    # Make feather wider to ensure smoother transition
                    top_feather = max(top_feather, min(int(top_overlap * 0.9), chunk_height // 3))
                    print(f"Chunk {i}: Top overlap with chunk {j}: {top_overlap}px, feather: {top_feather}px")
                
                # Bottom edge overlap
                if other_start_y < end_y and other_start_y >= start_y:
                    bottom_overlap = end_y - other_start_y
                    # Make feather wider to ensure smoother transition
                    bottom_feather = max(bottom_feather, min(int(bottom_overlap * 0.9), chunk_height // 3))
                    print(f"Chunk {i}: Bottom overlap with chunk {j}: {bottom_overlap}px, feather: {bottom_feather}px")
        
        # For max_alpha blend method, we use modified weighting to prioritize higher alpha values
        if blend_method == 'max_alpha':
            # For max_alpha, we use minimal feathering in overlap areas but keep edges smooth
            # We'll modify the weights later, these are just to ensure smooth transitions at edges
            print(f"Using specialized max_alpha weighting for chunk {i}")
            # Reduce the feathering for max_alpha to allow more content through
            left_feather = max(1, left_feather // 3)
            right_feather = max(1, right_feather // 3)
            top_feather = max(1, top_feather // 3)
            bottom_feather = max(1, bottom_feather // 3)
        
        # Apply left feathering if needed with more aggressive curve
        if left_feather > 0:
            for x in range(left_feather):
                # Use a modified cubic ease that provides more gradual transition
                t = x / left_feather
                # More gradual transition near the edge
                alpha = t * t * (3 - 2 * t) if t < 0.3 else 0.3 + (t - 0.3) * 1.4
                alpha = min(1.0, max(0.0, alpha))  # Clamp to [0,1]
                weight[:, x] *= alpha
        
        # Apply right feathering if needed with more aggressive curve
        if right_feather > 0:
            for x in range(right_feather):
                idx = chunk_width - x - 1  # Right edge
                if idx >= 0:  # Sanity check
                    # Use a modified cubic ease that provides more gradual transition
                    t = x / right_feather
                    # More gradual transition near the edge
                    alpha = t * t * (3 - 2 * t) if t < 0.3 else 0.3 + (t - 0.3) * 1.4
                    alpha = min(1.0, max(0.0, alpha))  # Clamp to [0,1]
                    weight[:, idx] *= alpha
        
        # Apply top feathering if needed with more aggressive curve
        if top_feather > 0:
            for y in range(top_feather):
                # Use a modified cubic ease that provides more gradual transition
                t = y / top_feather
                # More gradual transition near the edge
                alpha = t * t * (3 - 2 * t) if t < 0.3 else 0.3 + (t - 0.3) * 1.4
                alpha = min(1.0, max(0.0, alpha))  # Clamp to [0,1]
                weight[y, :] *= alpha
        
        # Apply bottom feathering if needed with more aggressive curve
        if bottom_feather > 0:
            for y in range(bottom_feather):
                idx = chunk_height - y - 1  # Bottom edge
                if idx >= 0:  # Sanity check
                    # Use a modified cubic ease that provides more gradual transition
                    t = y / bottom_feather
                    # More gradual transition near the edge
                    alpha = t * t * (3 - 2 * t) if t < 0.3 else 0.3 + (t - 0.3) * 1.4
                    alpha = min(1.0, max(0.0, alpha))  # Clamp to [0,1]
                    weight[idx, :] *= alpha
        
        # Apply additional smoothing to corners where feathering from two directions overlaps
        if (left_feather > 0 and top_feather > 0) or (left_feather > 0 and bottom_feather > 0) or \
           (right_feather > 0 and top_feather > 0) or (right_feather > 0 and bottom_feather > 0):
            print(f"Applying corner smoothing for chunk {i}")
            
            # Top-left corner
            if left_feather > 0 and top_feather > 0:
                for y in range(top_feather):
                    for x in range(left_feather):
                        t_x = x / left_feather
                        t_y = y / top_feather
                        # Use radial distance from corner for smoother corner blending
                        r = np.sqrt(t_x**2 + t_y**2) / np.sqrt(2)
                        # Smooth transition based on radial distance
                        alpha = r * r * (3 - 2 * r)
                        weight[y, x] = min(weight[y, x], alpha)
            
            # Top-right corner
            if right_feather > 0 and top_feather > 0:
                for y in range(top_feather):
                    for x in range(right_feather):
                        idx_x = chunk_width - x - 1
                        if idx_x >= 0:
                            t_x = x / right_feather
                            t_y = y / top_feather
                            # Use radial distance from corner for smoother corner blending
                            r = np.sqrt(t_x**2 + t_y**2) / np.sqrt(2)
                            # Smooth transition based on radial distance
                            alpha = r * r * (3 - 2 * r)
                            weight[y, idx_x] = min(weight[y, idx_x], alpha)
            
            # Bottom-left corner
            if left_feather > 0 and bottom_feather > 0:
                for y in range(bottom_feather):
                    idx_y = chunk_height - y - 1
                    if idx_y >= 0:
                        for x in range(left_feather):
                            t_x = x / left_feather
                            t_y = y / bottom_feather
                            # Use radial distance from corner for smoother corner blending
                            r = np.sqrt(t_x**2 + t_y**2) / np.sqrt(2)
                            # Smooth transition based on radial distance
                            alpha = r * r * (3 - 2 * r)
                            weight[idx_y, x] = min(weight[idx_y, x], alpha)
            
            # Bottom-right corner
            if right_feather > 0 and bottom_feather > 0:
                for y in range(bottom_feather):
                    idx_y = chunk_height - y - 1
                    if idx_y >= 0:
                        for x in range(right_feather):
                            idx_x = chunk_width - x - 1
                            if idx_x >= 0:
                                t_x = x / right_feather
                                t_y = y / bottom_feather
                                # Use radial distance from corner for smoother corner blending
                                r = np.sqrt(t_x**2 + t_y**2) / np.sqrt(2)
                                # Smooth transition based on radial distance
                                alpha = r * r * (3 - 2 * r)
                                weight[idx_y, idx_x] = min(weight[idx_y, idx_x], alpha)
        
        # Expand to 3 channels for RGB blending
        weight_3ch = np.repeat(weight[:, :, np.newaxis], 3, axis=2)
        blend_weights.append(weight_3ch)
        
        # Save a debug image of the weight mask for visualization (in temp dir if provided)
        if temp_dir:
            debug_path = os.path.join(temp_dir, f"weight_mask_strip_{i}.png")
        else:
            debug_path = f"weight_mask_strip_{i}.png"
            
        cv2.imwrite(debug_path, (weight * 255).astype(np.uint8))
        
        # Report feathering summary for this chunk
        print(f"Chunk {i} feathering summary:")
        print(f"  Left: {left_feather}px, Right: {right_feather}px, Top: {top_feather}px, Bottom: {bottom_feather}px")
    
    # Optimize weights if mask maximization is enabled
    if maximize_mask:
        print("Using mask optimization to maximize all mask content across chunks")
        # Create a composite visualization of all chunk contributions
        if temp_dir:
            composite_mask_path = create_composite_masks(
                chunk_outputs, width, height, frame_count, temp_dir
            )
            if composite_mask_path:
                print(f"Created composite mask visualization at {composite_mask_path}")
        
        # Get the maximum possible alpha from the first frame by combining all chunks
        max_alpha = create_maximized_alpha_mask(
            chunk_outputs, width, height, frame_index=0, temp_dir=temp_dir
        )
        
        if max_alpha is not None:
            print("Created reference maximum alpha mask")
            
            # Save visualization of max possible alpha
            if temp_dir:
                max_path = os.path.join(temp_dir, "max_possible_alpha.png")
                cv2.imwrite(max_path, max_alpha.astype(np.uint8))
        
        # Get propagated mask data before reassembly
        # This is key to ensuring all chunks have all available mask information
        propagated_data = propagate_mask_data(
            chunk_outputs, frame_index=0, temp_dir=temp_dir
        )
        
        if propagated_data and 'global_max_mask' in propagated_data:
            print("Successfully created global maximum mask with all available mask data")
            
            # Save the global max mask
            if temp_dir:
                global_max_path = os.path.join(temp_dir, "global_max_mask_reassembly.png")
                cv2.imwrite(global_max_path, propagated_data['global_max_mask'].astype(np.uint8))
        
        # Optimize the blending weights based on chunk analysis
        from chunking.chunk_optimizer import optimize_reassembly_weights
        optimized_weights = optimize_reassembly_weights(
            blend_weights, chunk_outputs, width, height, temp_dir
        )
        blend_weights = optimized_weights
    
    # Process all frames
    min_frames = frame_count
    for chunk in chunk_videos:
        chunk_frames = int(chunk['fgr_cap'].get(cv2.CAP_PROP_FRAME_COUNT))
        min_frames = min(min_frames, chunk_frames)
    
    if min_frames < frame_count:
        print(f"Warning: Using {min_frames} frames instead of {frame_count} (mismatch detected)")
        frame_count = min_frames
    
    # Create frame counter for each chunk to ensure synchronization
    chunk_frame_positions = [0] * len(chunk_videos)
    
    # Process frame by frame
    print("Reassembling frames...")
    start_time = time.time()
    
    # Cache a copy of the global maximum mask for each frame
    global_max_mask_cache = None
    
    # Process frames
    for frame_idx in range(frame_count):
        try:
            # Get optimized mask data if enabled
            optimized_data = None
            if maximize_mask:
                optimized_data = optimize_chunk_masks(
                    chunk_outputs, frame_idx, temp_dir if frame_idx == 0 else None
                )
                
                if optimized_data and frame_idx == 0:
                    print(f"Frame {frame_idx}: Found optimal mask data from {optimized_data['total_chunks']} chunks")
                    
                    # Cache the global max mask if available
                    if 'global_max_mask' in optimized_data:
                        global_max_mask_cache = optimized_data['global_max_mask'].copy()
            
            # Create output canvas
            fgr_canvas = np.zeros((height, width, 3), dtype=np.float32)
            pha_canvas = np.zeros((height, width, 3), dtype=np.float32)
            
            # For max_alpha blending, create a mask to track filled alpha areas
            if blend_method == 'max_alpha':
                # Create alpha value tracking array to store the maximum alpha at each pixel
                # Initialize with zeros (no alpha content)
                max_alpha_values = np.zeros((height, width, 3), dtype=np.float32)
                
                # Create a binary mask for regions where we have any alpha content
                has_alpha_content = np.zeros((height, width, 3), dtype=bool)
                
                # For debug purposes, create a contribution mask to see which chunk contributes to each pixel
                if frame_idx == 0 and temp_dir:
                    # We'll use a unique color for each chunk
                    contribution_mask = np.zeros((height, width, 3), dtype=np.uint8)
                    colors = [
                        (255, 0, 0),    # Red
                        (0, 255, 0),    # Green
                        (0, 0, 255),    # Blue
                        (255, 255, 0),  # Yellow
                        (0, 255, 255),  # Cyan
                        (255, 0, 255),  # Magenta
                        (128, 128, 0),  # Olive
                        (128, 0, 128),  # Purple
                        (0, 128, 128)   # Teal
                    ]
                    # Add more colors if needed
                    while len(colors) < len(chunk_videos):
                        colors.append(tuple(np.random.randint(0, 256, 3).tolist()))
            
            # Track accumulated weights separately for FGR and PHA
            fgr_weight_sum = np.zeros((height, width, 3), dtype=np.float32)
            pha_weight_sum = np.zeros((height, width, 3), dtype=np.float32)
            
            # Track maximum alpha mask for this frame
            max_alpha_mask = np.zeros((height, width), dtype=np.float32)
            
            # Create a canvas to store all mask data from all chunks at full resolution
            # This will be used to identify mask regions that should be filled even if a particular chunk doesn't have them
            all_chunks_mask = np.zeros((height, width), dtype=np.float32)
            
            # First, collect maximum alpha info across all chunks to identify edge regions
            all_alpha_masks = []  # Store individual chunk alpha masks
            
            for i, chunk in enumerate(chunk_videos):
                # Get the region in the output canvas
                start_x, end_x = chunk['x_range']
                start_y, end_y = chunk['y_range']
                
                # Skip invalid regions
                if start_x >= end_x or start_y >= end_y or start_x >= width or start_y >= height:
                    continue
                
                # Ensure we're at the right frame position
                current_pos = int(chunk['pha_cap'].get(cv2.CAP_PROP_POS_FRAMES))
                if current_pos != chunk_frame_positions[i]:
                    # Seek to correct position if needed
                    chunk['pha_cap'].set(cv2.CAP_PROP_POS_FRAMES, chunk_frame_positions[i])
                    chunk['fgr_cap'].set(cv2.CAP_PROP_POS_FRAMES, chunk_frame_positions[i]) 
                
                # Read alpha frame for max mask calculation
                ret_pha, pha_frame = chunk['pha_cap'].read()
                
                # If read fails, skip this chunk
                if not ret_pha:
                    chunk_frame_positions[i] += 1  # Still increment position
                    continue
                
                # Extract alpha channel
                if len(pha_frame.shape) == 3:
                    if np.all(pha_frame[:,:,0] == pha_frame[:,:,1]) and np.all(pha_frame[:,:,1] == pha_frame[:,:,2]):
                        alpha = pha_frame[:,:,0].astype(np.float32)
                    else:
                        alpha = cv2.cvtColor(pha_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                else:
                    alpha = pha_frame.astype(np.float32)
                
                # Calculate region dimensions
                width_to_use = min(end_x - start_x, pha_frame.shape[1])
                height_to_use = min(end_y - start_y, pha_frame.shape[0])
                
                # Set up region slices
                region_slice_y = slice(start_y, start_y + height_to_use)
                region_slice_x = slice(start_x, start_x + width_to_use)
                
                # Update max alpha mask
                current_max = max_alpha_mask[region_slice_y, region_slice_x]
                alpha_region = alpha[:height_to_use, :width_to_use]
                max_alpha_mask[region_slice_y, region_slice_x] = np.maximum(current_max, alpha_region)
                
                # Update all chunks mask
                all_chunks_mask[region_slice_y, region_slice_x] = np.maximum(
                    all_chunks_mask[region_slice_y, region_slice_x],
                    alpha_region
                )
                
                # Store alpha mask and position information for edge detection
                all_alpha_masks.append({
                    'alpha': alpha,
                    'x_range': (start_x, end_x),
                    'y_range': (start_y, end_y),
                    'chunk_idx': i
                })
            
            # IMPROVED: Use more advanced edge detection
            # This will help us identify mask boundaries that need to be preserved exactly
            
            # First, apply threshold to get binary mask for edge detection
            _, binary_mask = cv2.threshold(all_chunks_mask.astype(np.uint8), 20, 255, cv2.THRESH_BINARY)
            
            # Apply dilation to fill small holes before edge detection
            dilated_mask = cv2.dilate(binary_mask, np.ones((3, 3), np.uint8), iterations=1)
            
            # Apply Canny edge detection with appropriate parameters
            # Lower threshold, higher threshold, and aperture size
            edge_map = cv2.Canny(dilated_mask, 30, 100, apertureSize=3)
            
            # If we have global_max_mask_cache, use it to improve the edge map
            if global_max_mask_cache is not None:
                # Create a binary version of the global max mask
                _, global_binary = cv2.threshold(
                    global_max_mask_cache.astype(np.uint8), 20, 255, cv2.THRESH_BINARY
                )
                
                # Detect edges in the global max mask
                global_edges = cv2.Canny(global_binary, 30, 100, apertureSize=3)
                
                # Combine edge maps, taking the maximum at each pixel
                edge_map = np.maximum(edge_map, global_edges)
            
            # Dilate the edge map slightly to ensure edges are fully covered
            edge_map = cv2.dilate(edge_map, np.ones((3, 3), np.uint8), iterations=1)
            
            # Save edge map for debugging if first frame
            if frame_idx == 0 and temp_dir:
                edge_path = os.path.join(temp_dir, "alpha_edge_map.png")
                cv2.imwrite(edge_path, edge_map)
                
                # Save dilated mask for debugging
                dilated_path = os.path.join(temp_dir, "dilated_mask.png")
                cv2.imwrite(dilated_path, dilated_mask)
                
                # Save all chunks mask for debugging
                all_chunks_path = os.path.join(temp_dir, "all_chunks_mask.png")
                cv2.imwrite(all_chunks_path, all_chunks_mask.astype(np.uint8))
            
            # ENHANCED: Also identify interior regions of the mask that should always be filled
            # Dilate the binary mask substantially and then subtract the edge regions
            mask_interior = cv2.dilate(binary_mask, np.ones((7, 7), np.uint8), iterations=1)
            # Subtract the edge regions
            mask_interior[edge_map > 0] = 0
            
            # Save interior mask for debugging if first frame
            if frame_idx == 0 and temp_dir:
                interior_path = os.path.join(temp_dir, "mask_interior.png")
                cv2.imwrite(interior_path, mask_interior)
            
            # Now process each chunk with a hybrid approach for FGR and PHA
            for i, chunk in enumerate(chunk_videos):
                # Use the same position from previous pass (don't rewind)
                current_pos = int(chunk['pha_cap'].get(cv2.CAP_PROP_POS_FRAMES))
                # Adjust position if needed
                if current_pos != chunk_frame_positions[i]:
                    chunk['pha_cap'].set(cv2.CAP_PROP_POS_FRAMES, chunk_frame_positions[i])
                    chunk['fgr_cap'].set(cv2.CAP_PROP_POS_FRAMES, chunk_frame_positions[i])
                
                # Read both frames now (foreground and alpha)
                chunk['pha_cap'].set(cv2.CAP_PROP_POS_FRAMES, chunk_frame_positions[i])
                chunk['fgr_cap'].set(cv2.CAP_PROP_POS_FRAMES, chunk_frame_positions[i])
                
                ret_fgr, fgr_frame = chunk['fgr_cap'].read()
                ret_pha, pha_frame = chunk['pha_cap'].read()
                
                # Update frame position 
                chunk_frame_positions[i] += 1
                
                # Skip if reading fails
                if not ret_fgr or not ret_pha:
                    continue
                
                # Get the region in the output canvas
                start_x, end_x = chunk['x_range']
                start_y, end_y = chunk['y_range']
                
                if start_x >= end_x or start_y >= end_y or start_x >= width or start_y >= height:
                    continue  # Skip invalid regions
                
                # Ensure end coordinates don't exceed output dimensions
                end_x = min(end_x, width)
                end_y = min(end_y, height)
                
                # Make sure frames match expected dimensions
                expected_width = chunk['width']
                expected_height = chunk['height']
                actual_width = fgr_frame.shape[1]
                actual_height = fgr_frame.shape[0]
                
                # If dimensions don't match, resize the frames
                if actual_width != expected_width or actual_height != expected_height:
                    try:
                        fgr_frame = cv2.resize(fgr_frame, (expected_width, expected_height), interpolation=cv2.INTER_LINEAR)
                        pha_frame = cv2.resize(pha_frame, (expected_width, expected_height), interpolation=cv2.INTER_LINEAR)
                    except Exception as e:
                        print(f"Error resizing frame for chunk {i}: {str(e)}")
                        continue
                
                # Convert to float32 for calculations
                fgr_float = fgr_frame.astype(np.float32)
                pha_float = pha_frame.astype(np.float32)
                
                # Extract alpha channel for analysis
                if len(pha_float.shape) == 3:
                    if np.all(pha_float[:,:,0] == pha_float[:,:,1]) and np.all(pha_float[:,:,1] == pha_float[:,:,2]):
                        alpha = pha_float[:,:,0]
                    else:
                        alpha = cv2.cvtColor(pha_float, cv2.COLOR_BGR2GRAY)
                else:
                    alpha = pha_float
                
                # Make sure blend weights match frame dimensions
                weight = blend_weights[i]
                if weight.shape[1] != expected_width or weight.shape[0] != expected_height:
                    try:
                        weight = cv2.resize(weight, (expected_width, expected_height), interpolation=cv2.INTER_LINEAR)
                    except Exception as e:
                        print(f"Error resizing weight mask for chunk {i}: {str(e)}")
                        continue
                
                # Calculate region dimensions accounting for potential cropping
                width_to_use = min(end_x - start_x, expected_width)
                height_to_use = min(end_y - start_y, expected_height)
                
                # Set up region slices
                region_slice_y = slice(start_y, start_y + height_to_use)
                region_slice_x = slice(start_x, start_x + width_to_use)
                
                try:
                    # Extract relevant portion of the edge map for this chunk
                    chunk_edge_map = edge_map[region_slice_y, region_slice_x]
                    
                    # Extract interior mask for this chunk
                    chunk_interior = mask_interior[region_slice_y, region_slice_x]
                    
                    # Extract alpha region for analysis
                    alpha_region = alpha[:height_to_use, :width_to_use]
                    
                    # HANDLE DIFFERENT BLEND METHODS
                    if blend_method == 'max_alpha':
                        # For max_alpha method, use a completely different approach that prioritizes 
                        # higher alpha values for every pixel rather than weighted blending
                        
                        # Extract regions from canvas
                        current_fgr = fgr_canvas[region_slice_y, region_slice_x]
                        current_pha = pha_canvas[region_slice_y, region_slice_x]
                        
                        # Extract mask values array for this region
                        region_max_alpha = max_alpha_values[region_slice_y, region_slice_x]
                        region_has_alpha = has_alpha_content[region_slice_y, region_slice_x]
                        
                        # Extract chunk data
                        chunk_fgr = fgr_float[:height_to_use, :width_to_use]
                        chunk_pha = pha_float[:height_to_use, :width_to_use]
                        
                        # Use alpha threshold to identify content
                        mask_threshold = 10  # For identifying content
                        
                        # Check all 3 channels for whether this chunk has higher alpha than current max
                        has_higher_alpha = np.zeros((height_to_use, width_to_use, 3), dtype=bool)
                        is_first_content = np.zeros((height_to_use, width_to_use, 3), dtype=bool)
                        
                        for c in range(3):
                            # Check where this chunk has higher alpha than the current maximum
                            has_higher_alpha[:,:,c] = (chunk_pha[:,:,c] > region_max_alpha[:,:,c]) & (chunk_pha[:,:,c] > mask_threshold)
                            
                            # Also identify pixels where we don't yet have any alpha content but this chunk has some
                            is_first_content[:,:,c] = (chunk_pha[:,:,c] > mask_threshold) & ~region_has_alpha[:,:,c]
                        
                        # Combine conditions: either higher alpha or first content
                        should_update = has_higher_alpha | is_first_content
                        
                        # Update the pixels where this chunk has higher alpha values
                        if np.any(should_update):
                            for c in range(3):
                                # Where this chunk has higher alpha or is first content, update both foreground and alpha
                                current_fgr[:,:,c][should_update[:,:,c]] = chunk_fgr[:,:,c][should_update[:,:,c]]
                                current_pha[:,:,c][should_update[:,:,c]] = chunk_pha[:,:,c][should_update[:,:,c]]
                                
                                # Update max alpha values for tracking
                                region_max_alpha[:,:,c][should_update[:,:,c]] = chunk_pha[:,:,c][should_update[:,:,c]]
                                
                                # Mark that this pixel now has content
                                region_has_alpha[:,:,c][should_update[:,:,c]] = True
                                
                                # For first frame debugging, update contribution mask
                                if frame_idx == 0 and temp_dir:
                                    contribution_mask[region_slice_y, region_slice_x][should_update[:,:,0]] = colors[i % len(colors)]
                        
                        # Update canvas with modified regions
                        fgr_canvas[region_slice_y, region_slice_x] = current_fgr
                        pha_canvas[region_slice_y, region_slice_x] = current_pha
                        
                        # Update tracking arrays
                        max_alpha_values[region_slice_y, region_slice_x] = region_max_alpha
                        has_alpha_content[region_slice_y, region_slice_x] = region_has_alpha
                        
                    else:
                        # For weighted/average/min_alpha blend methods, use standard weighted approach
                        
                        # Identify edge pixels, interior mask pixels, and everything else
                        edge_pixels = chunk_edge_map > 0
                        mask_threshold = 20
                        
                        # IMPROVED: Identify interior mask pixels based on both:
                        # 1. This chunk's mask content
                        # 2. The overall interior mask that combines all chunks
                        interior_from_chunk = (alpha_region > mask_threshold) & ~edge_pixels
                        interior_from_all = (chunk_interior > 0)
                        
                        # ENHANCED: Combine interiors - any pixel that should be interior based on either source
                        interior_pixels = interior_from_chunk | interior_from_all
                        
                        # Create masks for processing different regions differently
                        edge_mask = np.zeros((height_to_use, width_to_use, 3), dtype=bool)
                        interior_mask = np.zeros((height_to_use, width_to_use, 3), dtype=bool)
                        
                        for c in range(3):
                            edge_mask[:,:,c] = edge_pixels
                            interior_mask[:,:,c] = interior_pixels
                        
                        # Extract relevant regions from current canvas
                        current_fgr = fgr_canvas[region_slice_y, region_slice_x]
                        current_pha = pha_canvas[region_slice_y, region_slice_x]
                        current_fgr_weight = fgr_weight_sum[region_slice_y, region_slice_x]
                        current_pha_weight = pha_weight_sum[region_slice_y, region_slice_x]
                        
                        # Extract chunk data
                        chunk_fgr = fgr_float[:height_to_use, :width_to_use]
                        chunk_pha = pha_float[:height_to_use, :width_to_use]
                        chunk_weight = weight[:height_to_use, :width_to_use]
                        
                        # Apply weight to chunk data
                        weighted_fgr = chunk_fgr * chunk_weight
                        weighted_pha = chunk_pha * chunk_weight
                        
                        # ENHANCED HYBRID BLENDING APPROACH
                        
                        # For edge pixels - use maximum blending for alpha to preserve edge detail
                        if np.any(edge_pixels):
                            # For edges in alpha channel, identify where this chunk has higher values than current canvas
                            alpha_edges = edge_pixels & (alpha_region > 5)  # Only consider significant alpha content
                            
                            if np.any(alpha_edges):
                                # Convert to 3-channel mask
                                alpha_edges_3ch = np.zeros((height_to_use, width_to_use, 3), dtype=bool)
                                for c in range(3):
                                    alpha_edges_3ch[:,:,c] = alpha_edges
                                
                                # Special handling for min_alpha blend method at edges
                                if blend_method == 'min_alpha':
                                    # For min_alpha, we want to use the minimum value at edges
                                    # But only where both chunks have content (to avoid black edges)
                                    has_content_both = (chunk_pha > mask_threshold) & (current_pha > mask_threshold)
                                    use_edges = alpha_edges_3ch & has_content_both
                                    
                                    if np.any(use_edges):
                                        for c in range(3):
                                            # Use minimum value where both have content
                                            current_pha[:,:,c][use_edges[:,:,c]] = np.minimum(
                                                chunk_pha[:,:,c][use_edges[:,:,c]], 
                                                current_pha[:,:,c][use_edges[:,:,c]]
                                            )
                                            
                                            # Also update foreground with the same pixels for consistency
                                            current_fgr[:,:,c][use_edges[:,:,c]] = chunk_fgr[:,:,c][use_edges[:,:,c]]
                                            
                                            # Set weights to 1.0 to indicate this is the final value
                                            current_pha_weight[:,:,c][use_edges[:,:,c]] = 1.0
                                            current_fgr_weight[:,:,c][use_edges[:,:,c]] = 1.0
                                else:
                                    # For weighted/average blend methods, identify where this chunk has higher alpha at edges
                                    # This ensures we preserve the best edge detail from any chunk
                                    higher_alpha = np.zeros((height_to_use, width_to_use, 3), dtype=bool)
                                    for c in range(3):
                                        higher_alpha[:,:,c] = chunk_pha[:,:,c] > current_pha[:,:,c]
                                    
                                    # Where this chunk has higher alpha at edges, use it directly
                                    use_edges = alpha_edges_3ch & higher_alpha
                                    if np.any(use_edges):
                                        for c in range(3):
                                            # Update alpha values at edges using direct replacement for higher values
                                            current_pha[:,:,c][use_edges[:,:,c]] = chunk_pha[:,:,c][use_edges[:,:,c]]
                                            
                                            # Also update foreground with weighted value at these pixels
                                            # This ensures the foreground color matches the alpha
                                            current_fgr[:,:,c][use_edges[:,:,c]] = chunk_fgr[:,:,c][use_edges[:,:,c]]
                                            
                                            # Set weights to 1.0 to indicate this is the final value
                                            current_pha_weight[:,:,c][use_edges[:,:,c]] = 1.0
                                            current_fgr_weight[:,:,c][use_edges[:,:,c]] = 1.0
                            
                            # For other edge pixels, use weighted addition with reduced contribution
                            # This focuses on preserving edges from the chunks that define them best
                            use_normal_edges = edge_pixels & ~alpha_edges
                            if np.any(use_normal_edges):
                                use_normal_edges_3ch = np.zeros((height_to_use, width_to_use, 3), dtype=bool)
                                for c in range(3):
                                    use_normal_edges_3ch[:,:,c] = use_normal_edges
                                
                                for c in range(3):
                                    # Add weighted contribution with standard blending
                                    current_fgr[:,:,c][use_normal_edges] += weighted_fgr[:,:,c][use_normal_edges]
                                    current_pha[:,:,c][use_normal_edges] += weighted_pha[:,:,c][use_normal_edges]
                                    
                                    # Accumulate weights for normalization
                                    current_fgr_weight[:,:,c][use_normal_edges] += chunk_weight[:,:,c][use_normal_edges]
                                    current_pha_weight[:,:,c][use_normal_edges] += chunk_weight[:,:,c][use_normal_edges]
                        
                        # For interior mask pixels - use enhanced filling to avoid gaps
                        if np.any(interior_pixels):
                            # Identify interior pixels where this chunk has significant content
                            has_content = alpha_region > mask_threshold
                            interior_with_content = interior_pixels & has_content
                            interior_without_content = interior_pixels & ~has_content
                            
                            # Interior with content - standard weighted addition but with boosted weights
                            if np.any(interior_with_content):
                                interior_with_content_3ch = np.zeros((height_to_use, width_to_use, 3), dtype=bool)
                                for c in range(3):
                                    interior_with_content_3ch[:,:,c] = interior_with_content
                                
                                # Boost weights for interior with content by 1.5x
                                boosted_weight = chunk_weight.copy()
                                for c in range(3):
                                    boosted_weight[:,:,c][interior_with_content] *= 1.5
                                
                                # Apply boosted weighted addition
                                for c in range(3):
                                    # Add boosted weighted contribution
                                    current_fgr[:,:,c][interior_with_content] += chunk_fgr[:,:,c][interior_with_content] * boosted_weight[:,:,c][interior_with_content]
                                    current_pha[:,:,c][interior_with_content] += chunk_pha[:,:,c][interior_with_content] * boosted_weight[:,:,c][interior_with_content]
                                    
                                    # Accumulate boosted weights
                                    current_fgr_weight[:,:,c][interior_with_content] += boosted_weight[:,:,c][interior_with_content]
                                    current_pha_weight[:,:,c][interior_with_content] += boosted_weight[:,:,c][interior_with_content]
                            
                            # Interior without content - use background neutral color with minimum alpha
                            # This prevents black artifacts in regions that should be filled but this chunk doesn't have content
                            if np.any(interior_without_content):
                                for c in range(3):
                                    # For alpha, we'll use a small value to ensure it's not completely transparent
                                    # but still allows other chunks to contribute more substantial content
                                    current_pha[:,:,c][interior_without_content] += 5 * chunk_weight[:,:,c][interior_without_content]
                                    
                                    # For foreground, use a neutral gray value to avoid black artifacts
                                    current_fgr[:,:,c][interior_without_content] += 128 * chunk_weight[:,:,c][interior_without_content]
                                    
                                    # Add to weight sum for normalization
                                    current_fgr_weight[:,:,c][interior_without_content] += chunk_weight[:,:,c][interior_without_content]
                                    current_pha_weight[:,:,c][interior_without_content] += chunk_weight[:,:,c][interior_without_content]
                        
                        # For non-edge, non-interior areas - use standard weighted addition
                        non_special = ~(edge_pixels | interior_pixels)
                        if np.any(non_special):
                            non_special_3ch = np.zeros((height_to_use, width_to_use, 3), dtype=bool)
                            for c in range(3):
                                non_special_3ch[:,:,c] = non_special
                            
                            for c in range(3):
                                # Standard weighted addition
                                current_fgr[:,:,c][non_special] += weighted_fgr[:,:,c][non_special]
                                current_pha[:,:,c][non_special] += weighted_pha[:,:,c][non_special]
                                
                                # Accumulate weights
                                current_fgr_weight[:,:,c][non_special] += chunk_weight[:,:,c][non_special]
                                current_pha_weight[:,:,c][non_special] += chunk_weight[:,:,c][non_special]
                        
                        # Update canvas with the processed regions
                        fgr_canvas[region_slice_y, region_slice_x] = current_fgr
                        pha_canvas[region_slice_y, region_slice_x] = current_pha
                        fgr_weight_sum[region_slice_y, region_slice_x] = current_fgr_weight
                        pha_weight_sum[region_slice_y, region_slice_x] = current_pha_weight
                    
                except Exception as e:
                    print(f"Error blending chunk {i} at position {(start_x, start_y)}-({end_x, end_y}): {str(e)}")
                    traceback.print_exc()
                    continue
            
            # Save the max_alpha contribution visualization if needed
            if frame_idx == 0 and temp_dir and blend_method == 'max_alpha':
                contribution_path = os.path.join(temp_dir, "max_alpha_contribution_map.png")
                cv2.imwrite(contribution_path, contribution_mask)
                print(f"Saved max_alpha contribution visualization to {contribution_path}")
            
            # For max_alpha method, we already have the final output - no normalization needed
            if blend_method == 'max_alpha':
                # Assign outputs directly from canvas
                fgr_output = fgr_canvas
                pha_output = pha_canvas
            else:
                # ENHANCED NORMALIZATION for other methods
                # Use different normalization strategies for different regions
                
                # Get binary mask for edge and interior regions
                edge_binary = (edge_map > 0).astype(np.uint8)
                interior_binary = mask_interior.astype(np.uint8)
                
                # Create 3-channel masks
                edge_mask_3ch = np.zeros((height, width, 3), dtype=bool)
                interior_mask_3ch = np.zeros((height, width, 3), dtype=bool)
                
                for c in range(3):
                    edge_mask_3ch[:,:,c] = edge_binary > 0
                    interior_mask_3ch[:,:,c] = interior_binary > 0
                
                # For foreground - use different thresholds for different regions
                fgr_edge_threshold = 0.05
                fgr_interior_threshold = 0.01
                fgr_other_threshold = 0.01
                
                # Apply minimum threshold to weights to avoid division by zero
                # But use different thresholds for different regions
                fgr_weight_sum_safe = fgr_weight_sum.copy()
                fgr_weight_sum_safe[edge_mask_3ch] = np.maximum(fgr_weight_sum_safe[edge_mask_3ch], fgr_edge_threshold)
                fgr_weight_sum_safe[interior_mask_3ch & ~edge_mask_3ch] = np.maximum(
                    fgr_weight_sum_safe[interior_mask_3ch & ~edge_mask_3ch], fgr_interior_threshold
                )
                fgr_weight_sum_safe[~edge_mask_3ch & ~interior_mask_3ch] = np.maximum(
                    fgr_weight_sum_safe[~edge_mask_3ch & ~interior_mask_3ch], fgr_other_threshold
                )
                
                # Normalize foreground
                fgr_output = fgr_canvas / fgr_weight_sum_safe
                
                # For alpha - use similar approach with different thresholds
                pha_edge_threshold = 0.05
                pha_interior_threshold = 0.01
                pha_other_threshold = 0.01
                
                # Apply minimum threshold to weights to avoid division by zero
                pha_weight_sum_safe = pha_weight_sum.copy()
                pha_weight_sum_safe[edge_mask_3ch] = np.maximum(pha_weight_sum_safe[edge_mask_3ch], pha_edge_threshold)
                pha_weight_sum_safe[interior_mask_3ch & ~edge_mask_3ch] = np.maximum(
                    pha_weight_sum_safe[interior_mask_3ch & ~edge_mask_3ch], pha_interior_threshold
                )
                pha_weight_sum_safe[~edge_mask_3ch & ~interior_mask_3ch] = np.maximum(
                    pha_weight_sum_safe[~edge_mask_3ch & ~interior_mask_3ch], pha_other_threshold
                )
                
                # Normalize alpha
                pha_output = pha_canvas / pha_weight_sum_safe
                
                # For min_alpha blend method, apply additional reduction at the end
                if blend_method == 'min_alpha':
                    # Create a binary mask of regions with actual content
                    content_threshold = 10  # For identifying content regions
                    has_content_mask = (pha_output > content_threshold).astype(np.float32)
                    
                    # Apply a small reduction to the alpha values where we have content
                    # This helps prevent fringing around subject edges
                    reduction_factor = 0.9  # Reduce by 10%
                    pha_output = pha_output * (has_content_mask * reduction_factor + (1 - has_content_mask))
            
            # ENHANCED HIGH-RES MASK APPLICATION
            # Apply high-res mask more carefully if requested
            if apply_expanded_mask and full_res_mask_dir:
                # Load the full-res mask for this frame
                mask_path = os.path.join(full_res_mask_dir, f"{frame_idx:08d}.png")
                
                if os.path.exists(mask_path):
                    # Load the mask
                    frame_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    if frame_mask is not None:
                        # Expand the mask slightly to include edge area (prevent edge artifacts)
                        kernel_size = 7  # Slightly larger dilation to better fill interior
                        kernel = np.ones((kernel_size, kernel_size), np.uint8)
                        expanded_mask = cv2.dilate(frame_mask, kernel, iterations=1)
                        
                        # Convert to float32 and normalize to 0-1
                        mask_float = expanded_mask.astype(np.float32) / 255.0
                        
                        # Apply slight blur to prevent hard mask edges
                        # This helps preserve feathered edges from chunk blending
                        mask_float = cv2.GaussianBlur(mask_float, (7, 7), 0)
                        
                        # ENHANCED: Create an edge map for the expanded mask
                        # This lets us avoid modifying the precise alpha edges
                        _, mask_binary = cv2.threshold(expanded_mask, 20, 255, cv2.THRESH_BINARY)
                        mask_edges = cv2.Canny(mask_binary, 30, 100)
                        
                        # Dilate edges slightly to ensure we preserve all edge details
                        mask_edges = cv2.dilate(mask_edges, np.ones((3, 3), np.uint8), iterations=1)
                        
                        # Create a mask that excludes edges (we don't want to modify edges)
                        non_edge_mask = (mask_edges == 0).astype(np.float32)
                        
                        # Expand to 3 channels for RGB multiplication
                        mask_3ch = np.repeat(mask_float[:, :, np.newaxis], 3, axis=2)
                        non_edge_mask_3ch = np.repeat(non_edge_mask[:, :, np.newaxis], 3, axis=2)
                        
                        # Apply mask to foreground and alpha
                        # For foreground, apply only to non-edge areas
                        # This preserves the precise edge colors from the matting process
                        fgr_output = fgr_output * (mask_3ch * non_edge_mask_3ch + (1 - non_edge_mask_3ch))
                        
                        # For alpha, apply maximum blending to ensure we get the best edges
                        # from both the matting process and the high-res mask
                        max_blend_mask = np.maximum(pha_output, mask_3ch)
                        
                        # Apply maximum blend mask to alpha output but preserve edges from matting
                        edge_preserve_weight = 0.9  # How much to preserve edges (0.0-1.0)
                        pha_output = (pha_output * edge_preserve_weight + max_blend_mask * (1 - edge_preserve_weight)) * mask_3ch
                        
                        # Debug: Add verification of mask application
                        if frame_idx == 0 and temp_dir:
                            # Save the mask for debugging
                            debug_mask_path = os.path.join(temp_dir, "expanded_mask_debug.png")
                            cv2.imwrite(debug_mask_path, expanded_mask)
                            
                            # Save the mask edges for debugging
                            debug_edges_path = os.path.join(temp_dir, "mask_edges_debug.png")
                            cv2.imwrite(debug_edges_path, mask_edges)
                            
                            # Report statistics for debugging
                            print(f"Applied expanded mask to frame {frame_idx}")
                            print(f"  Mask non-zero pixels: {np.count_nonzero(expanded_mask)}")
                            print(f"  Mask dimensions: {expanded_mask.shape}")
                            
                            # Also save a sample of the alpha output for the first frame
                            debug_alpha_path = os.path.join(temp_dir, "alpha_after_mask_debug.png")
                            alpha_debug = np.clip(pha_output, 0, 255).astype(np.uint8)
                            cv2.imwrite(debug_alpha_path, alpha_debug)
                            
                            # Report alpha statistics
                            print(f"  Alpha non-zero pixels after masking: {np.count_nonzero(alpha_debug)}")
                            print(f"  Alpha min/max values: {np.min(alpha_debug)}/{np.max(alpha_debug)}")
                    else:
                        if frame_idx == 0:
                            print(f"Warning: Could not read mask frame from {mask_path}")
                else:
                    if frame_idx == 0:
                        print(f"Warning: Mask frame not found at {mask_path}")
            
            # Save debug images for the first few frames
            if frame_idx < 3 and temp_dir:
                # Save foreground and alpha debug images
                fgr_debug = np.clip(fgr_output, 0, 255).astype(np.uint8)
                pha_debug = np.clip(pha_output, 0, 255).astype(np.uint8)
                
                fgr_debug_path = os.path.join(temp_dir, f"fgr_frame_{frame_idx}_debug.png")
                pha_debug_path = os.path.join(temp_dir, f"pha_frame_{frame_idx}_debug.png")
                
                cv2.imwrite(fgr_debug_path, fgr_debug)
                cv2.imwrite(pha_debug_path, pha_debug)
                
                # Report frame statistics for debugging
                print(f"Frame {frame_idx} debug stats:")
                print(f"  Foreground non-zero pixels: {np.count_nonzero(fgr_debug)}")
                print(f"  Alpha non-zero pixels: {np.count_nonzero(pha_debug)}")
            
            # Convert to uint8
            fgr_output = np.clip(fgr_output, 0, 255).astype(np.uint8)
            pha_output = np.clip(pha_output, 0, 255).astype(np.uint8)
            
            # Write frames
            fgr_writer.write(fgr_output)
            pha_writer.write(pha_output)
            
            # Show progress periodically
            if frame_idx % 10 == 0 or frame_idx == frame_count - 1:
                elapsed = time.time() - start_time
                fps_rate = (frame_idx + 1) / max(0.001, elapsed)
                eta = (frame_count - frame_idx - 1) / max(0.001, fps_rate)
                
                print(f"Reassembled {frame_idx+1}/{frame_count} frames "
                      f"({(frame_idx+1)/frame_count*100:.1f}%) - "
                      f"{fps_rate:.1f} fps, ETA: {eta:.1f}s")
            
        except Exception as e:
            print(f"Error reassembling frame {frame_idx}: {str(e)}")
            traceback.print_exc()
            # Try to continue with next frame
    
    # Release resources
    for chunk in chunk_videos:
        chunk['fgr_cap'].release()
        chunk['pha_cap'].release()
    
    fgr_writer.release()
    pha_writer.release()
    
    return fgr_output_path, pha_output_path


def reassemble_grid_chunks(chunk_outputs, width, height, fps, frame_count, fgr_output_path, pha_output_path, 
                          blend_method='weighted', temp_dir=None, apply_expanded_mask=False, full_res_mask_dir=None,
                          maximize_mask=True, video_codec='Auto', video_quality='High', custom_bitrate=None):
    """
    Reassemble grid chunks into a complete video with mask optimization
    
    Args:
        chunk_outputs: List of dictionaries with chunk outputs
        width: Output video width
        height: Output video height
        fps: Frames per second
        frame_count: Total frame count
        fgr_output_path: Path to save foreground output
        pha_output_path: Path to save alpha output
        blend_method: Method for blending overlapping regions
        temp_dir: Directory to save temporary files (debug images)
        apply_expanded_mask: Whether to multiply output with expanded mask to prevent spill
        full_res_mask_dir: Directory containing full resolution mask frames
        maximize_mask: Whether to use mask optimization to maximize mask content
        video_codec: Video codec to use
        video_quality: Quality preset
        custom_bitrate: Custom bitrate in kbps
        
    Returns:
        Tuple of (fgr_output_path, pha_output_path)
    """
    # Grid reassembly uses the same logic as strip reassembly
    # We reuse the code but indicate it's for grid chunks
    print(f"Reassembling grid chunks with mask propagation (blend method: {blend_method})...")
    
    # Create grid visualization if possible
    if temp_dir:
        # Create a grid visualization showing all chunks
        grid_viz = np.zeros((height, width, 3), dtype=np.uint8)
        
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
        
        # Draw each chunk position
        for i, chunk in enumerate(chunk_outputs):
            start_x, end_x = chunk['x_range']
            start_y, end_y = chunk['y_range']
            
            # Choose color
            if i < len(colors):
                color = colors[i]
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
            cv2.putText(grid_viz, str(i), (start_x + 10, start_y + 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save grid visualization
        grid_path = os.path.join(temp_dir, "grid_chunks_layout.png")
        cv2.imwrite(grid_path, grid_viz)
    
    # The rest of the function is identical to strip reassembly
    return reassemble_strip_chunks(
        chunk_outputs, width, height, fps, frame_count, 
        fgr_output_path, pha_output_path, blend_method, 
        temp_dir, apply_expanded_mask, full_res_mask_dir, maximize_mask,
        video_codec, video_quality, custom_bitrate
    )
