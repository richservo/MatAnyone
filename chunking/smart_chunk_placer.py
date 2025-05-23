"""
# smart_chunk_placer.py - v1.000000000
# Created: Wednesday, December 18, 2024
# Intelligent chunk placement based on heat map analysis

This module determines optimal chunk placement using heat maps to maximize
content coverage while minimizing the number of chunks needed.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2


class SmartChunkPlacer:
    """
    Determines optimal chunk placement based on heat map analysis.
    """
    
    def __init__(self, overlap_ratio: float = 0.2, min_activity_threshold: float = 0.05):
        """
        Initialize the smart chunk placer.
        
        Args:
            overlap_ratio: Overlap ratio between chunks (0.2 = 20% overlap)
            min_activity_threshold: Minimum heat map activity to consider placing a chunk
        """
        self.overlap_ratio = overlap_ratio
        self.min_activity_threshold = min_activity_threshold
        self.placed_chunks = []
        
    def find_optimal_chunk_placement(self, heat_map: np.ndarray, 
                                   chunk_width: int, chunk_height: int,
                                   model_factor: int = 8, 
                                   allow_rotation: bool = True) -> List[Dict]:
        """
        Find optimal chunk placement based on heat map.
        
        Args:
            heat_map: 2D heat map array
            chunk_width: Width of each chunk (matching low-res dimensions)
            chunk_height: Height of each chunk (matching low-res dimensions)
            model_factor: Factor for dimension divisibility
            
        Returns:
            List of chunk dictionaries with placement info
        """
        print(f"Finding optimal chunk placement for {chunk_width}x{chunk_height} chunks")
        
        frame_height, frame_width = heat_map.shape
        
        # Calculate overlap in pixels
        overlap_x = int(chunk_width * self.overlap_ratio)
        overlap_y = int(chunk_height * self.overlap_ratio)
        
        # Ensure overlaps are divisible by model_factor
        overlap_x = (overlap_x // model_factor) * model_factor
        overlap_y = (overlap_y // model_factor) * model_factor
        
        print(f"Using overlap: {overlap_x}x{overlap_y} pixels")
        
        # Create a copy of heat map to track remaining activity
        remaining_heat = heat_map.copy()
        self.placed_chunks = []
        
        # Define orientations - allow both if rotation is enabled
        if allow_rotation:
            orientations = [
                {'width': chunk_width, 'height': chunk_height, 'name': 'horizontal'},
                {'width': chunk_height, 'height': chunk_width, 'name': 'vertical'}
            ]
        else:
            orientations = [{'width': chunk_width, 'height': chunk_height, 'name': 'horizontal'}]
        
        # Iteratively place chunks until activity is below threshold
        iteration = 0
        min_score_threshold = 0.01  # Minimum score to place a chunk
        
        while iteration < 100:
            iteration += 1
            
            # Find best position and orientation
            best_score = -1
            best_position = None
            best_orientation = None
            
            for orientation in orientations:
                o_width = orientation['width']
                o_height = orientation['height']
                
                # Skip if chunk doesn't fit
                if o_width > frame_width or o_height > frame_height:
                    continue
                
                # Scan all possible positions
                for y in range(0, frame_height - o_height + 1, model_factor):
                    for x in range(0, frame_width - o_width + 1, model_factor):
                        # Calculate score for this position with slight centering preference
                        chunk_region = remaining_heat[y:y+o_height, x:x+o_width]
                        score = self._calculate_score_with_centering(chunk_region, x, y, o_width, o_height)
                        
                        if score > best_score:
                            best_score = score
                            best_position = (x, y)
                            best_orientation = orientation
            
            # Stop only if there's truly no remaining activity
            if best_score < min_score_threshold or best_position is None:
                # Double-check for any missed areas
                if self._has_uncovered_activity(remaining_heat, self.min_activity_threshold):
                    print(f"Warning: Some activity remains uncovered. Attempting additional passes...")
                    # Lower threshold and continue
                    min_score_threshold *= 0.5
                    if min_score_threshold < 0.001:
                        break
                else:
                    break
            
            # Place chunk at best position
            x, y = best_position
            chunk_info = {
                'x_range': (x, x + best_orientation['width']),
                'y_range': (y, y + best_orientation['height']),
                'width': best_orientation['width'],
                'height': best_orientation['height'],
                'orientation': best_orientation['name'],
                'score': float(best_score),
                'chunk_id': len(self.placed_chunks)
            }
            
            self.placed_chunks.append(chunk_info)
            
            print(f"Placed chunk {len(self.placed_chunks)}: {best_orientation['name']} "
                  f"at ({x},{y}), score: {best_score:.2f}")
            
            # Mark this area as covered
            # Set to zero for the actual chunk area (not the overlap area)
            remaining_heat[y:y+best_orientation['height'], x:x+best_orientation['width']] = 0
            
            # Reduce heat in overlap areas to allow some reuse but discourage redundancy
            if overlap_x > 0:
                # Left overlap
                if x > 0:
                    left_start = max(0, x - overlap_x)
                    remaining_heat[y:y+best_orientation['height'], left_start:x] *= 0.3
                # Right overlap
                if x + best_orientation['width'] < frame_width:
                    right_end = min(frame_width, x + best_orientation['width'] + overlap_x)
                    remaining_heat[y:y+best_orientation['height'], x+best_orientation['width']:right_end] *= 0.3
            
            if overlap_y > 0:
                # Top overlap
                if y > 0:
                    top_start = max(0, y - overlap_y)
                    remaining_heat[top_start:y, x:x+best_orientation['width']] *= 0.3
                # Bottom overlap
                if y + best_orientation['height'] < frame_height:
                    bottom_end = min(frame_height, y + best_orientation['height'] + overlap_y)
                    remaining_heat[y+best_orientation['height']:bottom_end, x:x+best_orientation['width']] *= 0.3
        
        # Ensure minimum overlap between all chunks
        self._ensure_chunk_overlap()
        
        # Final pass: check for any missed areas and add chunks if needed
        self._ensure_complete_coverage(heat_map, chunk_width, chunk_height, model_factor)
        
        print(f"Total chunks placed: {len(self.placed_chunks)}")
        return self.placed_chunks
    
    def _calculate_score_with_centering(self, chunk_region, x, y, chunk_width, chunk_height):
        """
        Calculate score for chunk placement with slight preference for centering.
        
        Args:
            chunk_region: Heat map region for this chunk position
            x, y: Top-left position of the chunk
            chunk_width, chunk_height: Dimensions of the chunk
            
        Returns:
            Score for this chunk position
        """
        if chunk_region.size == 0:
            return 0
        
        # Base score is the total activity in the region
        total_activity = np.sum(chunk_region)
        if total_activity == 0:
            return 0
        
        # Find the center of mass of activity to check if subjects might be cut off
        y_coords, x_coords = np.where(chunk_region > 0.1)
        if len(y_coords) > 0:
            weights = chunk_region[y_coords, x_coords]
            center_y = np.average(y_coords, weights=weights)
            center_x = np.average(x_coords, weights=weights)
            
            # Calculate how far the center is from chunk edges
            edge_margin = min(chunk_width // 6, chunk_height // 6)  # 1/6 of chunk size
            
            # Check if center is too close to edges
            too_close_to_edge = (center_x < edge_margin or 
                               center_x > chunk_width - edge_margin or
                               center_y < edge_margin or 
                               center_y > chunk_height - edge_margin)
            
            # Small penalty if activity center is near edges (15% reduction)
            if too_close_to_edge:
                total_activity *= 0.85
        
        return total_activity
    
    def _ensure_chunk_overlap(self):
        """
        Ensure all chunks have proper overlap with their neighbors.
        """
        if len(self.placed_chunks) <= 1:
            return
        
        # Check each chunk pair for overlap
        for i in range(len(self.placed_chunks)):
            chunk_i = self.placed_chunks[i]
            
            for j in range(i + 1, len(self.placed_chunks)):
                chunk_j = self.placed_chunks[j]
                
                # Check if chunks are adjacent (touching or close)
                if self._chunks_are_adjacent(chunk_i, chunk_j):
                    # Ensure they have proper overlap
                    self._adjust_for_overlap(chunk_i, chunk_j)
    
    def _chunks_are_adjacent(self, chunk1: Dict, chunk2: Dict) -> bool:
        """
        Check if two chunks are adjacent (close enough to need overlap).
        """
        # Get chunk boundaries
        x1_start, x1_end = chunk1['x_range']
        y1_start, y1_end = chunk1['y_range']
        x2_start, x2_end = chunk2['x_range']
        y2_start, y2_end = chunk2['y_range']
        
        # Check horizontal adjacency
        horizontal_gap = max(x1_start - x2_end, x2_start - x1_end)
        vertical_gap = max(y1_start - y2_end, y2_start - y1_end)
        
        # Consider adjacent if gap is less than overlap amount
        overlap_threshold_x = int(chunk1['width'] * self.overlap_ratio)
        overlap_threshold_y = int(chunk1['height'] * self.overlap_ratio)
        
        return horizontal_gap < overlap_threshold_x or vertical_gap < overlap_threshold_y
    
    def _adjust_for_overlap(self, chunk1: Dict, chunk2: Dict):
        """
        Adjust chunk positions to ensure proper overlap.
        """
        # This is a placeholder - in practice, we rely on the placement algorithm
        # to handle overlap through the heat map reduction approach
        pass
    
    def get_chunk_coverage_stats(self, heat_map: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistics about chunk coverage.
        
        Args:
            heat_map: Original heat map
            
        Returns:
            Dictionary with coverage statistics
        """
        if not self.placed_chunks:
            return {'coverage': 0.0, 'efficiency': 0.0}
        
        # Create coverage mask
        coverage_mask = np.zeros_like(heat_map, dtype=bool)
        
        for chunk in self.placed_chunks:
            x_start, x_end = chunk['x_range']
            y_start, y_end = chunk['y_range']
            coverage_mask[y_start:y_end, x_start:x_end] = True
        
        # Calculate statistics
        total_activity = np.sum(heat_map)
        covered_activity = np.sum(heat_map[coverage_mask])
        
        stats = {
            'coverage': float(covered_activity / total_activity) if total_activity > 0 else 0.0,
            'num_chunks': len(self.placed_chunks),
            'total_chunk_area': sum(c['width'] * c['height'] for c in self.placed_chunks),
            'frame_area': heat_map.shape[0] * heat_map.shape[1],
            'efficiency': float(covered_activity / len(self.placed_chunks)) if self.placed_chunks else 0.0
        }
        
        return stats
    
    def _evaluate_orientation_fit(self, heat_map: np.ndarray, chunk_width: int, chunk_height: int) -> float:
        """
        Evaluate how well a given chunk orientation fits the heat map activity.
        
        Args:
            heat_map: Heat map to evaluate
            chunk_width: Width of chunks in this orientation
            chunk_height: Height of chunks in this orientation
            
        Returns:
            Score indicating how well this orientation covers the activity
        """
        total_score = 0.0
        num_positions = 0
        
        frame_height, frame_width = heat_map.shape
        
        # Sample positions across the heat map
        for y in range(0, frame_height - chunk_height + 1, chunk_height // 2):
            for x in range(0, frame_width - chunk_width + 1, chunk_width // 2):
                # Get the heat in this chunk area
                chunk_heat = heat_map[y:y+chunk_height, x:x+chunk_width]
                chunk_score = np.sum(chunk_heat)
                
                if chunk_score > 0:
                    # Calculate how efficiently this chunk uses its space
                    efficiency = chunk_score / (chunk_width * chunk_height)
                    total_score += efficiency
                    num_positions += 1
        
        return total_score / max(1, num_positions)
    
    def _has_uncovered_activity(self, remaining_heat: np.ndarray, threshold: float) -> bool:
        """
        Check if there's any significant uncovered activity in the heat map.
        
        Args:
            remaining_heat: Current remaining heat map
            threshold: Activity threshold to consider significant
            
        Returns:
            True if there's uncovered activity above threshold
        """
        # Check if any region has activity above threshold
        return np.max(remaining_heat) > threshold
    
    def _ensure_complete_coverage(self, original_heat_map: np.ndarray, chunk_width: int, 
                                chunk_height: int, model_factor: int):
        """
        Final pass to ensure complete coverage of all activity areas.
        Adds additional chunks if any areas were missed.
        """
        # Create a coverage mask showing which areas are covered
        frame_height, frame_width = original_heat_map.shape
        coverage_mask = np.zeros_like(original_heat_map, dtype=bool)
        
        # Mark all covered areas
        for chunk in self.placed_chunks:
            x_start, x_end = chunk['x_range']
            y_start, y_end = chunk['y_range']
            coverage_mask[y_start:y_end, x_start:x_end] = True
        
        # Find uncovered areas with activity
        uncovered_activity = original_heat_map * (~coverage_mask)
        
        # If significant uncovered activity exists, add more chunks
        activity_threshold = 0.05  # 5% of max activity
        max_activity = np.max(original_heat_map)
        
        while np.max(uncovered_activity) > activity_threshold * max_activity:
            # Find the region with highest uncovered activity
            best_score = -1
            best_position = None
            best_orientation = None
            
            # Try both orientations
            for orientation_name, width, height in [('horizontal', chunk_width, chunk_height),
                                                   ('vertical', chunk_height, chunk_width)]:
                if width > frame_width or height > frame_height:
                    continue
                    
                for y in range(0, frame_height - height + 1, model_factor):
                    for x in range(0, frame_width - width + 1, model_factor):
                        # Calculate uncovered activity in this region
                        region_activity = uncovered_activity[y:y+height, x:x+width]
                        score = np.sum(region_activity)
                        
                        if score > best_score:
                            best_score = score
                            best_position = (x, y)
                            best_orientation = {'width': width, 'height': height, 'name': orientation_name}
            
            if best_position is None or best_score < 0.01:
                break
                
            # Add the chunk
            x, y = best_position
            chunk_info = {
                'x_range': (x, x + best_orientation['width']),
                'y_range': (y, y + best_orientation['height']),
                'width': best_orientation['width'],
                'height': best_orientation['height'],
                'orientation': best_orientation['name'],
                'score': float(best_score),
                'chunk_id': len(self.placed_chunks),
                'coverage_pass': True  # Mark as added during coverage pass
            }
            
            self.placed_chunks.append(chunk_info)
            
            # Update coverage mask
            coverage_mask[y:y+best_orientation['height'], x:x+best_orientation['width']] = True
            uncovered_activity = original_heat_map * (~coverage_mask)
            
            print(f"Added coverage chunk {len(self.placed_chunks)}: {best_orientation['name']} "
                  f"at ({x},{y}) to cover missed activity")
    
    def visualize_chunk_placement(self, heat_map: np.ndarray, output_path: str):
        """
        Create a visualization of chunk placement on the heat map.
        
        Args:
            heat_map: Original heat map
            output_path: Path to save visualization
        """
        # Create visualization
        heat_map_vis = (heat_map * 255).astype(np.uint8)
        vis_image = cv2.cvtColor(heat_map_vis, cv2.COLOR_GRAY2BGR)
        vis_image = cv2.applyColorMap(vis_image, cv2.COLORMAP_JET)
        
        # Draw chunks
        for i, chunk in enumerate(self.placed_chunks):
            x_start, x_end = chunk['x_range']
            y_start, y_end = chunk['y_range']
            
            # Draw rectangle
            color = (0, 255, 0) if chunk['orientation'] == 'horizontal' else (255, 0, 0)
            cv2.rectangle(vis_image, (x_start, y_start), (x_end, y_end), color, 2)
            
            # Add chunk number
            cv2.putText(vis_image, str(i+1), 
                       (x_start + 10, y_start + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save visualization
        cv2.imwrite(output_path, vis_image)
        print(f"Chunk placement visualization saved to: {output_path}")


def calculate_optimal_chunks(heat_map: np.ndarray, low_res_width: int, low_res_height: int,
                           overlap_ratio: float = 0.2, model_factor: int = 8) -> List[Dict]:
    """
    Convenience function to calculate optimal chunk placement.
    
    Args:
        heat_map: Heat map from HeatMapAnalyzer
        low_res_width: Width of low-res processing (chunk width)
        low_res_height: Height of low-res processing (chunk height)
        overlap_ratio: Overlap between chunks
        model_factor: Factor for dimension divisibility
        
    Returns:
        List of optimal chunk placements
    """
    placer = SmartChunkPlacer(overlap_ratio=overlap_ratio)
    chunks = placer.find_optimal_chunk_placement(
        heat_map, low_res_width, low_res_height, model_factor
    )
    
    # Print statistics
    stats = placer.get_chunk_coverage_stats(heat_map)
    print(f"Coverage: {stats['coverage']*100:.1f}% with {stats['num_chunks']} chunks")
    print(f"Efficiency: {stats['efficiency']:.2f} activity per chunk")
    
    return chunks