"""
Test script for heat map based chunking functionality
"""

import os
import sys
import numpy as np
import cv2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunking.heat_map_analyzer import HeatMapAnalyzer
from chunking.smart_chunk_placer import SmartChunkPlacer


def test_heat_map_generation():
    """Test heat map generation from synthetic masks"""
    print("Testing heat map generation...")
    
    # Create synthetic mask data
    width, height = 1920, 1080
    num_frames = 30
    
    # Create temporary directory for masks
    temp_mask_dir = "/tmp/test_masks"
    os.makedirs(temp_mask_dir, exist_ok=True)
    
    # Generate synthetic masks with activity in specific regions
    for i in range(num_frames):
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Add a moving object (simulating a person walking)
        x_center = int(width * (0.3 + 0.4 * i / num_frames))
        y_center = height // 2
        cv2.circle(mask, (x_center, y_center), 100, 255, -1)
        
        # Save mask
        mask_path = os.path.join(temp_mask_dir, f"mask_{i:06d}.png")
        cv2.imwrite(mask_path, mask)
    
    # Test heat map generation
    analyzer = HeatMapAnalyzer(face_priority_weight=3.0)
    heat_map = analyzer.analyze_mask_sequence(temp_mask_dir)
    
    # Save visualization
    vis_path = "/tmp/test_heat_map.png"
    analyzer.save_heat_map_visualization(vis_path)
    print(f"Heat map visualization saved to: {vis_path}")
    
    # Get statistics
    stats = analyzer.get_activity_stats()
    print(f"Heat map statistics: {stats}")
    
    # Clean up
    import shutil
    shutil.rmtree(temp_mask_dir)
    
    return heat_map


def test_chunk_placement(heat_map):
    """Test optimal chunk placement"""
    print("\nTesting chunk placement...")
    
    # Test with different chunk sizes (simulating 0.5x scale)
    chunk_width = 960
    chunk_height = 540
    
    placer = SmartChunkPlacer(overlap_ratio=0.2)
    chunks = placer.find_optimal_chunk_placement(heat_map, chunk_width, chunk_height)
    
    print(f"Placed {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk['orientation']} at ({chunk['x_range'][0]},{chunk['y_range'][0]}) "
              f"size: {chunk['width']}x{chunk['height']}, score: {chunk['score']:.2f}")
    
    # Save placement visualization
    vis_path = "/tmp/test_chunk_placement.png"
    placer.visualize_chunk_placement(heat_map, vis_path)
    print(f"Chunk placement visualization saved to: {vis_path}")
    
    # Get coverage statistics
    stats = placer.get_chunk_coverage_stats(heat_map)
    print(f"Coverage statistics: {stats}")


def main():
    """Run all tests"""
    print("Heat Map Chunking Test Suite")
    print("=" * 50)
    
    # Test heat map generation
    heat_map = test_heat_map_generation()
    
    # Test chunk placement
    test_chunk_placement(heat_map)
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    main()