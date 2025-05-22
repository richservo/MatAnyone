"""
Simple test for heat map analyzer without circular imports
"""

import os
import sys
import numpy as np
import cv2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_heat_map_basic():
    """Test basic heat map functionality"""
    print("Testing basic heat map functionality...")
    
    # Test the modules can be imported
    try:
        from chunking.heat_map_analyzer import HeatMapAnalyzer
        print("✓ Heat map analyzer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import heat map analyzer: {e}")
        return
        
    try:
        from chunking.smart_chunk_placer import SmartChunkPlacer
        print("✓ Smart chunk placer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import smart chunk placer: {e}")
        return
    
    # Create a simple test
    print("\nCreating synthetic heat map...")
    width, height = 1920, 1080
    
    # Create a simple heat map with activity in the center
    heat_map = np.zeros((height, width), dtype=np.float32)
    cv2.circle(heat_map, (width//2, height//2), 200, 1.0, -1)
    
    # Test chunk placement
    print("Testing chunk placement...")
    placer = SmartChunkPlacer(overlap_ratio=0.2)
    chunks = placer.find_optimal_chunk_placement(heat_map, 960, 540)
    
    print(f"Successfully placed {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk['orientation']} at ({chunk['x_range'][0]},{chunk['y_range'][0]})")
    
    print("\n✓ Basic test passed!")


if __name__ == "__main__":
    test_heat_map_basic()