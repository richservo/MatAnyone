# apply_mask_optimization.py - v1.1737790500
# Updated: Friday, January 24, 2025 at 21:08:20 PST
# Changes in this version:
# - Significantly reduced aggressive mask optimization that was causing bright fringes
# - Implemented more conservative blending strategies to preserve original chunk quality
# - Added weight normalization improvements to prevent color artifacts
# - Reduced boost factors that were over-emphasizing certain regions
# - Improved edge handling to prevent fringe artifacts around foreground objects
# - Added better boundary condition handling for chunk overlap regions

"""
# apply_mask_optimization.py - v1.1717244555
# Updated: Friday, May 31, 2025
# Changes in this version:
# - Extracted apply_mask_optimization into a separate file to reduce chunk_optimizer.py size
# - Optimized pixel processing by using vectorized operations instead of pixel-by-pixel loops
# - Enhanced blending logic for better handling of edges vs interior areas
# - Added more efficient mask application for regions with missing content
# - Improved documentation for better code maintainability
"""

import numpy as np
import traceback


def apply_mask_optimization(fgr_canvas, pha_canvas, weight_sum, optimized_data, chunk_frame, chunk_idx, region_slice_y, region_slice_x):
    """
    Apply conservative mask optimization during reassembly to preserve chunk quality while maximizing mask preservation
    
    Args:
        fgr_canvas: Foreground canvas being composed from multiple chunks
        pha_canvas: Alpha/mask canvas being composed from multiple chunks
        weight_sum: Sum of weights for normalization
        optimized_data: Dictionary containing optimized mask data
        chunk_frame: Current chunk frame being processed
        chunk_idx: Index of current chunk
        region_slice_y: Y-axis slice of region being processed
        region_slice_x: X-axis slice of region being processed
        
    Returns:
        None - operates on inputs in-place
    """
    try:
        # If no optimized data, just return
        if optimized_data is None or 'enhanced_masks' not in optimized_data:
            return
        
        # Only apply optimizations if we have data for this chunk
        if chunk_idx not in optimized_data['enhanced_masks']:
            return
            
        # Get enhanced mask for this chunk
        if chunk_frame in optimized_data['enhanced_masks'][chunk_idx]:
            enhanced_mask = optimized_data['enhanced_masks'][chunk_idx][chunk_frame]
        else:
            return
        
        # Get the corresponding canvas regions
        canvas_region_y = slice(region_slice_y.start, region_slice_y.stop)
        canvas_region_x = slice(region_slice_x.start, region_slice_x.stop)
        
        # Ensure dimensions match
        expected_height = region_slice_y.stop - region_slice_y.start
        expected_width = region_slice_x.stop - region_slice_x.start
        
        if enhanced_mask.shape[0] != expected_height or enhanced_mask.shape[1] != expected_width:
            print(f"Warning: Enhanced mask dimensions ({enhanced_mask.shape}) don't match canvas region ({expected_height}, {expected_width})")
            # Try to resize if there's a mismatch
            if expected_height > 0 and expected_width > 0 and enhanced_mask.size > 0:
                try:
                    import cv2
                    enhanced_mask = cv2.resize(enhanced_mask, (expected_width, expected_height), interpolation=cv2.INTER_NEAREST)
                except Exception as e:
                    print(f"Error resizing enhanced mask: {str(e)}")
                    return
            else:
                return
                
        # Create 3-channel mask for processing
        enhanced_mask_3ch = np.repeat(enhanced_mask[:, :, np.newaxis], 3, axis=2)
        
        # Get current alpha values in this region
        current_alpha = pha_canvas[canvas_region_y, canvas_region_x]
        
        # Identify areas where we have mask content in the enhanced mask
        # but not in the current alpha channel
        add_mask = (enhanced_mask_3ch > 50) & (current_alpha < 10)
        
        if np.any(add_mask):
            # Determine how much to boost the weight sum in these areas
            # Use a modest boost (1.2x) to avoid artifacts
            boost_factor = 1.2
            weight_boost = np.zeros_like(weight_sum[canvas_region_y, canvas_region_x])
            weight_boost[add_mask] = weight_sum[canvas_region_y, canvas_region_x][add_mask] * (boost_factor - 1.0)
            
            # Apply the weight boost
            weight_sum[canvas_region_y, canvas_region_x] += weight_boost
            
            # For areas with missing content, increase alpha slightly
            # Use a conservative boost to avoid unnatural transitions
            alpha_boost = np.zeros_like(pha_canvas[canvas_region_y, canvas_region_x])
            alpha_boost[add_mask] = (enhanced_mask_3ch[add_mask] * 0.5).astype(np.float32)
            
            # Apply boost to alpha canvas
            pha_canvas[canvas_region_y, canvas_region_x] += alpha_boost
            
            # Also apply proportional boost to foreground canvas in these areas
            # This ensures that foreground pixels aren't darkened in the normalization step
            fgr_boost = np.zeros_like(fgr_canvas[canvas_region_y, canvas_region_x])
            fgr_boost[add_mask] = chunk_frame[add_mask] * 0.5
            fgr_canvas[canvas_region_y, canvas_region_x] += fgr_boost
        
    except Exception as e:
        print(f"Error in apply_mask_optimization: {str(e)}")
        traceback.print_exc()
        # Continue processing without optimization


def optimize_mask(mask_path, output_path, dilation_radius=5, blur_radius=3):
    """
    Optimize a mask by dilating and smoothing edges
    
    Args:
        mask_path: Path to input mask
        output_path: Path to save optimized mask
        dilation_radius: Radius for dilation operation
        blur_radius: Radius for blur operation
        
    Returns:
        Path to optimized mask
    """
    import cv2
    import os
    
    # Read the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Could not read mask at {mask_path}")
        return None
    
    # Dilate the mask
    if dilation_radius > 0:
        kernel = np.ones((dilation_radius, dilation_radius), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Blur the mask edges
    if blur_radius > 0:
        mask = cv2.GaussianBlur(mask, (blur_radius*2+1, blur_radius*2+1), 0)
    
    # Threshold back to binary
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Save the optimized mask
    cv2.imwrite(output_path, mask)
    
    return output_path