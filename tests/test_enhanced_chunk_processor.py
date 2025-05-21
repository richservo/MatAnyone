"""
# Created: Sunday, May 11, 2025
Test script for the enhanced chunk processor for MatAnyone.
This script allows testing the enhanced chunk processor with different settings.
"""

import os
import argparse
import time
from core.inference_core import InterruptibleInferenceCore

def test_enhanced_chunk_processor(input_path, mask_path, output_path, num_chunks, bidirectional, scale):
    """
    Test the enhanced chunk processor with specified parameters
    
    Args:
        input_path: Path to input video
        mask_path: Path to mask image
        output_path: Path to output directory
        num_chunks: Number of horizontal chunks
        bidirectional: Whether to use bidirectional processing
        scale: Preprocess scale factor
    """
    # Print test parameters
    print("=" * 80)
    print("Testing Enhanced Chunk Processor")
    print("=" * 80)
    print(f"Input video: {input_path}")
    print(f"Mask: {mask_path}")
    print(f"Output directory: {output_path}")
    print(f"Number of chunks: {num_chunks}")
    print(f"Bidirectional: {bidirectional}")
    print(f"Preprocess scale: {scale}")
    print("-" * 80)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize the processor
    processor = InterruptibleInferenceCore()
    print("Processor initialized")
    
    # Process parameters
    params = {
        'max_size': 512,  # Reasonable default size
        'save_image': True,  # Save individual frames
        'n_warmup': 10,
        'r_erode': 10,
        'r_dilate': 15,
        'reverse_dilate': 15,
        'blend_method': 'weighted'
    }
    
    # Start timing
    start_time = time.time()
    
    # Process the video using the enhanced chunk processor
    try:
        processor.process_video_in_chunks_enhanced(
            input_path=input_path,
            mask_path=mask_path,
            output_path=output_path,
            num_chunks=num_chunks,
            bidirectional=bidirectional,
            preprocess_scale=scale,
            **params
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        print("-" * 80)
        print(f"Processing completed in {elapsed_time:.2f} seconds")
        print(f"Results saved to {output_path}")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test Enhanced Chunk Processor")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--mask", required=True, help="Path to mask image")
    parser.add_argument("--output", default="./output", help="Path to output directory")
    parser.add_argument("--chunks", type=int, default=2, help="Number of horizontal chunks")
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional processing")
    parser.add_argument("--scale", type=float, default=0.25, help="Preprocess scale factor")
    
    args = parser.parse_args()
    
    # Run the test
    test_enhanced_chunk_processor(
        args.input,
        args.mask,
        args.output,
        args.chunks,
        args.bidirectional,
        args.scale
    )
