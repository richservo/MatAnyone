"""
# chunk_weight_optimization.py - v1.1735054800
# Updated: December 24, 2024
# Changes in this version:
# - Split off from chunk_optimizer.py to improve code organization and maintainability
# - Contains weight optimization functionality for chunk reassembly
# - Maintains all existing functionality while improving code structure
"""

import os
import cv2
import numpy as np
import traceback


def optimize_reassembly_weights(blend_weights, chunk_outputs, max_x, max_y, temp_dir=None):
    """
    Optimize blending weights based on analyzing all chunks' mask content
    
    Args:
        blend_weights: Original blend weights list
        chunk_outputs: List of dictionaries with chunk outputs 
        max_x: Maximum x dimension (width)
        max_y: Maximum y dimension (height)
        temp_dir: Directory to save debug images
    
    Returns:
        Optimized blend weights list
    """
    try:
        print("Optimizing reassembly weights to maximize mask information...")
        
        # Create a global mask contribution analysis for all chunks
        chunk_contribution = np.zeros((max_y, max_x, len(chunk_outputs)), dtype=np.float32)
        
        # For each chunk, analyze a sample of frames to determine contribution
        sample_frames = [0]  # Start with just the first frame for quick analysis
        
        for chunk_idx, chunk in enumerate(chunk_outputs):
            # Open alpha video
            pha_cap = cv2.VideoCapture(chunk['pha_path'])
            if not pha_cap.isOpened():
                print(f"Warning: Could not open alpha video for chunk {chunk_idx}")
                continue
            
            start_x, end_x = chunk['x_range']
            start_y, end_y = chunk['y_range']
            
            # Process sample frames
            for frame_idx in sample_frames:
                pha_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = pha_cap.read()
                
                if not ret:
                    continue
                
                # Convert to grayscale if needed
                if len(frame.shape) == 3:
                    if np.all(frame[:,:,0] == frame[:,:,1]) and np.all(frame[:,:,1] == frame[:,:,2]):
                        alpha = frame[:,:,0].astype(np.float32)
                    else:
                        alpha = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                else:
                    alpha = frame.astype(np.float32)
                
                # Extract region boundaries
                region_height, region_width = alpha.shape[:2]
                safe_end_x = min(start_x + region_width, max_x)
                safe_end_y = min(start_y + region_height, max_y)
                width_to_use = safe_end_x - start_x
                height_to_use = safe_end_y - start_y
                
                # Only update if dimensions are valid
                if width_to_use > 0 and height_to_use > 0:
                    # Extract the valid portion of the mask
                    alpha_region = alpha[:height_to_use, :width_to_use]
                    
                    # Create region slices
                    region_slice_y = slice(start_y, start_y + height_to_use)
                    region_slice_x = slice(start_x, start_x + width_to_use)
                    
                    # Update contribution for this chunk
                    chunk_contribution[region_slice_y, region_slice_x, chunk_idx] = alpha_region
            
            pha_cap.release()
        
        # Identify areas where multiple chunks have significant content
        mask_threshold = 10  # Threshold for considering a pixel as "masked"
        has_content = chunk_contribution > mask_threshold
        
        # Count how many chunks have content for each pixel
        content_count = np.sum(has_content, axis=2)
        
        # Find areas with content in multiple chunks (overlap)
        overlap_mask = content_count > 1
        
        # Create optimized weights
        optimized_weights = []
        
        for chunk_idx, original_weight in enumerate(blend_weights):
            # Start with the original weight
            new_weight = original_weight.copy()
            
            # Get this chunk's position
            start_x, end_x = chunk_outputs[chunk_idx]['x_range']
            start_y, end_y = chunk_outputs[chunk_idx]['y_range']
            
            # Ensure boundaries are within the main canvas
            safe_end_x = min(end_x, max_x)
            safe_end_y = min(end_y, max_y)
            width_to_use = safe_end_x - start_x
            height_to_use = safe_end_y - start_y
            
            # Skip if invalid dimensions
            if width_to_use <= 0 or height_to_use <= 0:
                optimized_weights.append(original_weight)
                continue
            
            # Create region slices
            region_slice_y = slice(start_y, start_y + height_to_use)
            region_slice_x = slice(start_x, start_x + width_to_use)
                
            # Extract this chunk's content from the contribution map
            chunk_content = chunk_contribution[region_slice_y, region_slice_x, chunk_idx].copy()
            
            # Extract the overlap mask for this chunk's region
            region_overlap = overlap_mask[region_slice_y, region_slice_x]
            
            # Calculate local maximum across all chunks for this region
            all_contributions = chunk_contribution[region_slice_y, region_slice_x, :]
            
            # Get maximum value for each pixel across all chunks
            max_contributions = np.max(all_contributions, axis=2)
            
            # Calculate which chunks have the ABSOLUTE maximum for each pixel
            is_max_contributor = chunk_content >= max_contributions
            
            # Calculate how close this chunk's content is to the maximum
            relative_strength = np.zeros_like(chunk_content)
            nonzero_mask = max_contributions > 0
            
            # Only update if there are non-zero pixels
            if np.any(nonzero_mask):
                # Avoid division by zero
                safe_max = np.maximum(max_contributions, 1e-6)
                # Calculate relative strength
                relative_strength[nonzero_mask] = chunk_content[nonzero_mask] / safe_max[nonzero_mask]
            
            # IMPROVED BOOSTING STRATEGY:
            # 1. Extreme boost for pixels where this chunk is the absolute maximum contributor
            # 2. High boost for pixels where this chunk has strong relative content 
            # 3. Medium boost for all content areas to ensure mask data is preserved
            # 4. Standard weights for non-content areas
            
            # 1. Extreme boost for max contributor pixels (20x)
            max_boost_factor = 20.0  # Much higher boost for absolute max
            max_mask = is_max_contributor & (chunk_content > mask_threshold) & region_overlap
            
            # 2. High boost for strong relative content (10x)
            strong_boost_factor = 10.0  # Higher boost for strong content
            strong_boost_threshold = 0.9  # Threshold for strong content
            strong_mask = (relative_strength > strong_boost_threshold) & (chunk_content > mask_threshold) & region_overlap & ~max_mask
            
            # 3. Medium boost for all content areas (5x)
            medium_boost_factor = 5.0  # Medium boost for any content
            content_mask = (chunk_content > mask_threshold) & ~max_mask & ~strong_mask
            
            # Apply boosts based on masks
            # Handle 2D and 3D weights appropriately
            if len(new_weight.shape) == 2:
                # Handle 2D weights
                new_weight[max_mask] *= max_boost_factor
                new_weight[strong_mask] *= strong_boost_factor
                new_weight[content_mask] *= medium_boost_factor
                
                # Add minimum weight for areas with significant content
                min_weight = 0.5  # Higher minimum weight to ensure content preservation
                significant_content = chunk_content > mask_threshold * 2
                min_weight_areas = (new_weight < min_weight) & significant_content
                new_weight[min_weight_areas] = min_weight
                
            elif len(new_weight.shape) == 3:
                # For 3-channel weights
                for c in range(new_weight.shape[2]):
                    if max_mask.shape == new_weight.shape[:2]:
                        new_weight[:,:,c][max_mask] *= max_boost_factor
                        new_weight[:,:,c][strong_mask] *= strong_boost_factor
                        new_weight[:,:,c][content_mask] *= medium_boost_factor
                        
                        # Add minimum weight for areas with significant content
                        min_weight = 0.5  # Higher minimum weight to ensure content preservation
                        significant_content = chunk_content > mask_threshold * 2
                        min_weight_areas = (new_weight[:,:,c] < min_weight) & significant_content
                        new_weight[:,:,c][min_weight_areas] = min_weight
                    elif max_mask.shape[0] <= new_weight.shape[0] and max_mask.shape[1] <= new_weight.shape[1]:
                        # Handle case where masks are smaller than weights
                        new_weight[:max_mask.shape[0], :max_mask.shape[1], c][max_mask] *= max_boost_factor
                        new_weight[:strong_mask.shape[0], :strong_mask.shape[1], c][strong_mask] *= strong_boost_factor
                        new_weight[:content_mask.shape[0], :content_mask.shape[1], c][content_mask] *= medium_boost_factor
                        
                        # Add minimum weight for areas with significant content
                        min_weight = 0.5  # Higher minimum weight to ensure content preservation
                        significant_content = chunk_content > mask_threshold * 2
                        if significant_content.shape[0] <= new_weight.shape[0] and significant_content.shape[1] <= new_weight.shape[1]:
                            min_weight_areas = (new_weight[:significant_content.shape[0], :significant_content.shape[1], c] < min_weight) & significant_content
                            new_weight[:significant_content.shape[0], :significant_content.shape[1], c][min_weight_areas] = min_weight
            
            # Save the optimized weight
            optimized_weights.append(new_weight)
            
            # Save debug images if requested
            if chunk_idx == 0 and temp_dir:
                # Save original weight
                orig_debug = original_weight
                if len(original_weight.shape) == 3:
                    orig_debug = original_weight[:, :, 0]
                debug_original = orig_debug * 255
                cv2.imwrite(os.path.join(temp_dir, f"original_weight_{chunk_idx}.png"), debug_original.astype(np.uint8))
                
                # Save optimized weight
                opt_debug = new_weight
                if len(new_weight.shape) == 3:
                    opt_debug = new_weight[:, :, 0]
                debug_optimized = opt_debug * 255
                cv2.imwrite(os.path.join(temp_dir, f"optimized_weight_{chunk_idx}.png"), debug_optimized.astype(np.uint8))
                
                # Create and save mask visualizations
                if max_mask.shape[0] > 0 and max_mask.shape[1] > 0:
                    max_viz = np.zeros_like(max_mask, dtype=np.uint8)
                    max_viz[max_mask] = 255
                    cv2.imwrite(os.path.join(temp_dir, f"max_boost_mask_{chunk_idx}.png"), max_viz)
                
                if strong_mask.shape[0] > 0 and strong_mask.shape[1] > 0:
                    strong_viz = np.zeros_like(strong_mask, dtype=np.uint8)
                    strong_viz[strong_mask] = 255
                    cv2.imwrite(os.path.join(temp_dir, f"strong_boost_mask_{chunk_idx}.png"), strong_viz)
                
                if content_mask.shape[0] > 0 and content_mask.shape[1] > 0:
                    content_viz = np.zeros_like(content_mask, dtype=np.uint8)
                    content_viz[content_mask] = 255
                    cv2.imwrite(os.path.join(temp_dir, f"content_mask_{chunk_idx}.png"), content_viz)
                
                # Save significant content mask
                significant_content = chunk_content > mask_threshold * 2
                if significant_content.shape[0] > 0 and significant_content.shape[1] > 0:
                    sig_viz = np.zeros_like(significant_content, dtype=np.uint8)
                    sig_viz[significant_content] = 255
                    cv2.imwrite(os.path.join(temp_dir, f"significant_content_{chunk_idx}.png"), sig_viz)
                    
                # Create a colorful visualization showing all mask categories
                if max_mask.shape == strong_mask.shape == content_mask.shape:
                    category_viz = np.zeros((max_mask.shape[0], max_mask.shape[1], 3), dtype=np.uint8)
                    category_viz[:,:,0][max_mask] = 255  # Red for max boost
                    category_viz[:,:,1][strong_mask] = 255  # Green for strong boost
                    category_viz[:,:,2][content_mask] = 255  # Blue for medium boost
                    cv2.imwrite(os.path.join(temp_dir, f"boost_categories_{chunk_idx}.png"), category_viz)
        
        print(f"Created optimized weights for {len(optimized_weights)} chunks with enhanced mask preservation.")
        return optimized_weights
        
    except Exception as e:
        print(f"Error optimizing reassembly weights: {str(e)}")
        traceback.print_exc()
        # Return original weights as fallback
        return blend_weights
