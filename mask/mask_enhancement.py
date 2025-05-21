"""
# mask_enhancement.py - v1.1717244556
# Updated: Friday, May 31, 2025
# Changes in this version:
# - Created dedicated mask enhancement utilities to fix chunk border issues
# - Added functions for applying high-resolution mask during reassembly
# - Enhanced edge preservation during mask application
# - Added overscan capabilities to fill gaps in final output
# - Improved interior/edge differentiation for better mask blending
"""

import os
import cv2
import numpy as np
import traceback


def enhance_mask_edges(mask, kernel_size=3, edge_strength=1.5):
    """
    Enhance edges in a mask to make them more defined and less susceptible to blending artifacts
    
    Args:
        mask: Input mask (grayscale)
        kernel_size: Size of the edge detection kernel
        edge_strength: Strength multiplier for edges
    
    Returns:
        Enhanced mask with stronger edges
    """
    try:
        # Convert mask to uint8 if needed
        if mask.dtype != np.uint8:
            mask_uint8 = mask.astype(np.uint8)
        else:
            mask_uint8 = mask.copy()
        
        # Apply Canny edge detection
        edges = cv2.Canny(mask_uint8, 30, 100)
        
        # Dilate the edges slightly to make them more prominent
        dilated_edges = cv2.dilate(edges, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
        
        # Create a float mask
        mask_float = mask.astype(np.float32)
        
        # Enhance edges by adding them to the mask
        mask_float[dilated_edges > 0] *= edge_strength
        
        # Clip values to valid range
        mask_float = np.clip(mask_float, 0, 255)
        
        return mask_float
        
    except Exception as e:
        print(f"Error enhancing mask edges: {str(e)}")
        traceback.print_exc()
        return mask


def detect_mask_interior(mask, threshold=20, margin=5):
    """
    Detect the interior regions of a mask, away from the edges
    
    Args:
        mask: Input mask (grayscale)
        threshold: Threshold for considering a pixel as masked
        margin: Margin from the edge to consider as interior
    
    Returns:
        Binary mask of interior regions
    """
    try:
        # Convert to binary mask
        _, binary = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
        
        # Detect edges
        edges = cv2.Canny(binary, 30, 100)
        
        # Dilate edges to create margin
        dilated_edges = cv2.dilate(edges, np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8), iterations=1)
        
        # Create interior mask (binary mask minus edges)
        interior = binary.copy()
        interior[dilated_edges > 0] = 0
        
        # Erode interior slightly to ensure it's well away from edges
        interior = cv2.erode(interior, np.ones((margin, margin), np.uint8), iterations=1)
        
        return interior
        
    except Exception as e:
        print(f"Error detecting mask interior: {str(e)}")
        traceback.print_exc()
        return np.zeros_like(mask)


def overscan_mask(mask, overscan_pixels=10):
    """
    Expand a mask's content by the specified number of pixels in all directions
    This helps fill gaps where chunks might be missing data
    
    Args:
        mask: Input mask (grayscale)
        overscan_pixels: Number of pixels to overscan
    
    Returns:
        Overscanned mask
    """
    try:
        # Convert to binary mask
        _, binary = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
        
        # Dilate to create overscan
        kernel = np.ones((overscan_pixels * 2 + 1, overscan_pixels * 2 + 1), np.uint8)
        overscanned = cv2.dilate(binary, kernel, iterations=1)
        
        # Create distance map from original mask edges
        dist_map = cv2.distanceTransform(overscanned, cv2.DIST_L2, 5)
        
        # Normalize distance map to range [0, 1]
        dist_max = np.max(dist_map)
        if dist_max > 0:
            dist_map = dist_map / dist_max
        
        # Create gradient for the overscan area
        gradient = np.ones_like(mask, dtype=np.float32)
        
        # Only in overscanned areas (not in original mask), apply distance-based falloff
        overscan_only = (overscanned > 0) & (binary == 0)
        if np.any(overscan_only):
            # Invert and scale the distance for overscan areas - closer to original = higher value
            gradient[overscan_only] = 1.0 - dist_map[overscan_only]
            # Apply gamma correction to make transition smoother
            gradient[overscan_only] = gradient[overscan_only] ** 0.5  # square root for gentler falloff
        
        # Create final overscanned mask with proper falloff
        result = mask.astype(np.float32).copy()
        result[overscan_only] = gradient[overscan_only] * 255.0
        
        return result
        
    except Exception as e:
        print(f"Error overscanning mask: {str(e)}")
        traceback.print_exc()
        return mask


def apply_high_res_mask(foreground, alpha, mask_path, preserve_edges=True, expand_pixels=7):
    """
    Apply a high-resolution mask to foreground and alpha outputs while preserving edges
    
    Args:
        foreground: Foreground output array (RGB)
        alpha: Alpha output array (RGB)
        mask_path: Path to the high-resolution mask
        preserve_edges: Whether to preserve edges from the original alpha
        expand_pixels: Number of pixels to expand the mask to fill gaps
    
    Returns:
        Tuple of (modified_foreground, modified_alpha)
    """
    try:
        # Check if mask exists
        if not os.path.exists(mask_path):
            print(f"Mask file not found: {mask_path}")
            return foreground, alpha
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to load mask: {mask_path}")
            return foreground, alpha
        
        # Ensure mask has the same dimensions as the inputs
        if mask.shape[:2] != foreground.shape[:2]:
            print(f"Resizing mask from {mask.shape} to {foreground.shape[:2]}")
            mask = cv2.resize(mask, (foreground.shape[1], foreground.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Apply binary thresholding to create a clean mask
        _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        
        # Expand the mask slightly to fill gaps
        if expand_pixels > 0:
            expanded_mask = cv2.dilate(binary_mask, np.ones((expand_pixels, expand_pixels), np.uint8), iterations=1)
        else:
            expanded_mask = binary_mask
        
        # Create a float mask [0.0, 1.0]
        mask_float = expanded_mask.astype(np.float32) / 255.0
        
        # Detect edges in the original alpha matte
        original_alpha = alpha.copy()
        if len(alpha.shape) == 3:
            # Use first channel if RGB
            alpha_channel = alpha[:, :, 0]
        else:
            alpha_channel = alpha
            
        # Create a binary version of alpha for edge detection
        _, alpha_binary = cv2.threshold(alpha_channel.astype(np.uint8), 20, 255, cv2.THRESH_BINARY)
        alpha_edges = cv2.Canny(alpha_binary, 30, 100)
        
        # Dilate alpha edges to ensure we preserve all edge detail
        if preserve_edges:
            alpha_edge_mask = cv2.dilate(alpha_edges, np.ones((5, 5), np.uint8), iterations=1)
            
            # Create an inverse edge mask to allow mask application elsewhere
            edge_mask = (alpha_edge_mask > 0).astype(np.float32)
            non_edge_mask = 1.0 - edge_mask
            
            # Expand to 3 channels if needed
            if len(foreground.shape) == 3:
                non_edge_mask_3ch = np.stack([non_edge_mask] * foreground.shape[2], axis=2)
                mask_float_3ch = np.stack([mask_float] * foreground.shape[2], axis=2)
            else:
                non_edge_mask_3ch = non_edge_mask
                mask_float_3ch = mask_float
            
            # Create a smooth blend between original alpha and high-res mask
            # Edge areas: preserve more of the original alpha
            # Non-edge areas: stronger influence from high-res mask
            edge_preserve_ratio = 0.9  # Higher value preserves more of the original edge detail
            
            # For alpha, ensure mask is applied while preserving edges
            # We use a weighted blend that keeps edges from original alpha
            alpha_blend = alpha * edge_mask + (alpha * edge_preserve_ratio + mask_float_3ch * (1.0 - edge_preserve_ratio)) * non_edge_mask_3ch
            
            # Apply expanded mask as a maximum to ensure all mask regions are filled
            # Only apply to non-edge areas to preserve edge detail
            alpha_result = np.maximum(alpha_blend, mask_float_3ch * non_edge_mask_3ch)
            
            # For foreground, only constrain to the expanded mask
            # This preserves the foreground colors for the edges
            foreground_result = foreground * mask_float_3ch
        else:
            # Simpler approach - just apply mask directly
            if len(foreground.shape) == 3:
                mask_float_3ch = np.stack([mask_float] * foreground.shape[2], axis=2)
            else:
                mask_float_3ch = mask_float
                
            alpha_result = alpha * mask_float_3ch
            foreground_result = foreground * mask_float_3ch
        
        return foreground_result, alpha_result
        
    except Exception as e:
        print(f"Error applying high-resolution mask: {str(e)}")
        traceback.print_exc()
        return foreground, alpha


def detect_and_fill_gaps(alpha, threshold=10, max_gap_size=20):
    """
    Detect and fill small gaps in the alpha matte where chunks might not overlap properly
    
    Args:
        alpha: Input alpha matte (grayscale or RGB)
        threshold: Threshold for considering a pixel as masked
        max_gap_size: Maximum gap size to fill
    
    Returns:
        Alpha matte with gaps filled
    """
    try:
        # Extract alpha channel if RGB
        if len(alpha.shape) == 3:
            alpha_channel = alpha[:, :, 0].copy()
        else:
            alpha_channel = alpha.copy()
        
        # Create binary mask
        _, binary = cv2.threshold(alpha_channel.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a mask for gaps
        gap_mask = np.zeros_like(binary)
        
        # Fill small holes inside contours
        for contour in contours:
            # Create a filled contour mask
            contour_mask = np.zeros_like(binary)
            cv2.drawContours(contour_mask, [contour], 0, 255, -1)
            
            # Find holes (areas inside contour but not in the binary mask)
            holes = contour_mask & ~binary
            
            # Find connected components in holes
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes)
            
            # Check each connected component
            for i in range(1, num_labels):  # Skip background (0)
                # If area is small enough, consider it a gap to fill
                if stats[i, cv2.CC_STAT_AREA] < max_gap_size * max_gap_size:
                    gap_mask[labels == i] = 255
        
        # Apply gap filling using distance transform to create smooth gradient
        # This creates a more natural fill than just binary filling
        
        # Distance transform from mask edges
        dist = cv2.distanceTransform(gap_mask, cv2.DIST_L2, 5)
        
        # Normalize distances to [0, 1]
        max_dist = np.max(dist)
        if max_dist > 0:
            norm_dist = dist / max_dist
        else:
            norm_dist = dist
            
        # Apply distance-based alpha values to gaps
        # Areas closer to edges will have higher alpha
        filled_alpha = alpha_channel.copy()
        filled_alpha[gap_mask > 0] = 255 * (1.0 - norm_dist[gap_mask > 0])
        
        # Create output matching input dimensionality
        if len(alpha.shape) == 3:
            result = alpha.copy()
            for c in range(alpha.shape[2]):
                result[:, :, c] = filled_alpha
            return result
        else:
            return filled_alpha
        
    except Exception as e:
        print(f"Error detecting and filling gaps: {str(e)}")
        traceback.print_exc()
        return alpha
