"""
# test_autochunk.py - v1.1715884120
# Created: Thursday, May 16, 2025
# Test script for the fixed auto-chunk mode in MatAnyone.
# This script helps verify that all chunks have identical dimensions matching the low-res mask.

This script tests the auto-chunk mode to ensure it creates chunks with identical
dimensions that match the low-resolution mask dimensions.
"""

import os
import argparse
import time
import numpy as np
import cv2
from core.inference_core import InterruptibleInferenceCore

def verify_chunk_dimensions(chunks, low_res_width, low_res_height):
    """
    Verify that all chunks have the same dimensions, matching the low-res dimensions
    
    Args:
        chunks: List of chunk dictionaries
        low_res_width: Low-resolution width that chunks should match
        low_res_height: Low-resolution height that chunks should match
        
    Returns:
        True if all chunks have correct dimensions, False otherwise
    """
    valid = True
    
    print("\nVerifying chunk dimensions:")
    print(f"Expected dimensions: {low_res_width}x{low_res_height}")
    print("-" * 50)
    
    for i, chunk in enumerate(chunks):
        x_range = chunk['x_range']
        y_range = chunk['y_range']
        width = chunk['width']
        height = chunk['height']
        
        # Calculate dimensions from ranges
        calculated_width = x_range[1] - x_range[0]
        calculated_height = y_range[1] - y_range[0]
        
        # Check if dimensions are consistent
        width_consistent = (width == calculated_width)
        height_consistent = (height == calculated_height)
        
        # Check if dimensions match low-res dimensions
        width_match = (width == low_res_width)
        height_match = (height == low_res_height)
        
        # Report results
        chunk_valid = width_consistent and height_consistent and width_match and height_match
        valid = valid and chunk_valid
        
        status = "✓ Valid" if chunk_valid else "✗ Invalid"
        print(f"Chunk {i+1}: {width}x{height} at ({x_range[0]},{y_range[0]}) - {status}")
        
        if not width_consistent or not height_consistent:
            print(f"  - Internal inconsistency: Stored {width}x{height}, calculated {calculated_width}x{calculated_height}")
        
        if not width_match or not height_match:
            print(f"  - Dimension mismatch: Expected {low_res_width}x{low_res_height}, got {width}x{height}")
            
    print("-" * 50)
    if valid:
        print("✓ All chunks have correct dimensions!")
    else:
        print("✗ Some chunks have incorrect dimensions!")
    
    return valid

def test_autochunk(input_path, mask_path, output_path, low_res_scale=0.25):
    """
    Test the auto-chunk mode with the specified parameters
    
    Args:
        input_path: Path to input video
        mask_path: Path to mask image
        output_path: Path to output directory
        low_res_scale: Scale factor for low-resolution preprocessing
    """
    # Print test parameters
    print("=" * 80)
    print("Testing Auto-Chunk Mode in MatAnyone")
    print("=" * 80)
    print(f"Input video: {input_path}")
    print(f"Mask: {mask_path}")
    print(f"Output directory: {output_path}")
    print(f"Low-res scale: {low_res_scale}")
    print("-" * 80)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize the processor
    processor = InterruptibleInferenceCore()
    print("Processor initialized")
    
    # Get video properties to calculate expected low-res dimensions
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Calculate adjusted dimensions (divisible by 8)
    MODEL_FACTOR = 8
    adjusted_width = (original_width // MODEL_FACTOR) * MODEL_FACTOR
    adjusted_height = (original_height // MODEL_FACTOR) * MODEL_FACTOR
    
    print(f"Video dimensions: {original_width}x{original_height}")
    print(f"Adjusted dimensions: {adjusted_width}x{adjusted_height}")
    
    # Calculate expected low-res dimensions
    low_res_width = int(adjusted_width * low_res_scale)
    low_res_height = int(adjusted_height * low_res_scale)
    
    # Make sure low-res dimensions are divisible by MODEL_FACTOR
    low_res_width = (low_res_width // MODEL_FACTOR) * MODEL_FACTOR
    low_res_height = (low_res_height // MODEL_FACTOR) * MODEL_FACTOR
    
    print(f"Expected low-res dimensions: {low_res_width}x{low_res_height}")
    print("-" * 80)
    
    # Import required functions for testing auto-chunking
    from chunking.chunking_utils import get_autochunk_segments
    
    # First, just test the chunking without processing
    print("Testing chunk creation...")
    chunks = get_autochunk_segments(
        adjusted_width,
        adjusted_height,
        low_res_width,
        low_res_height,
        MODEL_FACTOR
    )
    
    # Verify all chunks have the correct dimensions
    verify_result = verify_chunk_dimensions(chunks, low_res_width, low_res_height)
    
    if not verify_result:
        print("Chunk verification failed! Auto-chunk mode is not creating chunks with correct dimensions.")
        return
    
    # Process parameters for testing actual video processing
    process_test = input("Do you want to test actual video processing with auto-chunk mode? (y/n): ")
    
    if process_test.lower() == 'y':
        params = {
            'max_size': 512,  # Reasonable default size
            'save_image': False,  # Don't save individual frames for testing
            'bidirectional': True,  # Always use bidirectional
            'blend_method': 'weighted',
            'reverse_dilate': 15,
            'cleanup_temp': False,  # Keep temp files for debugging
            'mask_skip_threshold': 5,
            'low_res_scale': low_res_scale,
            'use_autochunk': True  # Enable auto-chunk mode
        }
        
        # Start timing
        start_time = time.time()
        
        # Process the video using the enhanced chunk processor with auto-chunk mode
        try:
            print("Starting video processing with auto-chunk mode...")
            processor.process_video_with_enhanced_chunking(
                input_path=input_path,
                mask_path=mask_path,
                output_path=output_path,
                **params
            )
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            print("-" * 80)
            print(f"Processing completed in {elapsed_time:.2f} seconds")
            print(f"Results saved to {output_path}")
            print("=" * 80)
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\nTest complete!")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test Auto-Chunk Mode in MatAnyone")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--mask", required=True, help="Path to mask image")
    parser.add_argument("--output", default="./output", help="Path to output directory")
    parser.add_argument("--scale", type=float, default=0.25, help="Low-res scale factor")
    
    args = parser.parse_args()
    
    # Run the test
    test_autochunk(
        args.input,
        args.mask,
        args.output,
        args.scale
    )
