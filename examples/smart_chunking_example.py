#!/usr/bin/env python3
"""
Smart Chunking Example - Standalone Usage
This example shows how to use the smart chunking system independently
for any video processing task.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from chunking.heat_map_analyzer import HeatMapAnalyzer
from chunking.smart_chunk_placer import SmartChunkPlacer


def generate_motion_heat_map(video_path, output_dir):
    """
    Generate a heat map based on motion detection.
    This is a simple example - you can replace with any activity detection method.
    """
    print(f"Analyzing motion in video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Could not read video")
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    height, width = prev_gray.shape
    
    # Initialize heat map
    heat_map = np.zeros((height, width), dtype=np.float32)
    
    frame_count = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(gray, prev_gray)
        
        # Threshold to get motion mask
        _, motion_mask = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Add to heat map
        heat_map += motion_mask.astype(np.float32) / 255.0
        
        prev_gray = gray
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")
    
    cap.release()
    
    # Normalize heat map
    heat_map = heat_map / frame_count
    
    # Save heat map visualization
    os.makedirs(output_dir, exist_ok=True)
    heat_map_vis = (heat_map * 255).astype(np.uint8)
    heat_map_colored = cv2.applyColorMap(heat_map_vis, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_dir, "motion_heat_map.png"), heat_map_colored)
    
    print(f"Heat map saved to {output_dir}/motion_heat_map.png")
    
    return heat_map


def apply_blur_effect_with_smart_chunks(video_path, output_path, heat_map, chunk_size=512):
    """
    Example: Apply blur effect only to areas with motion using smart chunks.
    """
    # Find optimal chunk placement
    placer = SmartChunkPlacer(overlap_ratio=0.2)
    chunks = placer.find_optimal_chunk_placement(
        heat_map,
        chunk_width=chunk_size,
        chunk_height=chunk_size,
        model_factor=8,
        allow_rotation=True
    )
    
    print(f"Found {len(chunks)} optimal chunks for processing")
    
    # Visualize chunk placement
    output_dir = os.path.dirname(output_path)
    placer.visualize_chunk_placement(heat_map, os.path.join(output_dir, "chunk_placement.png"))
    
    # Process video with chunks
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process each chunk
        output_frame = frame.copy()
        
        for i, chunk in enumerate(chunks):
            x_start, x_end = chunk['x_range']
            y_start, y_end = chunk['y_range']
            
            # Extract chunk
            chunk_region = frame[y_start:y_end, x_start:x_end]
            
            # Apply effect (blur in this example)
            blurred_chunk = cv2.GaussianBlur(chunk_region, (21, 21), 0)
            
            # Create a mask for smooth blending
            mask = np.ones(chunk_region.shape[:2], dtype=np.float32)
            
            # Feather the edges for smooth blending
            feather_size = 20
            mask[:feather_size, :] *= np.linspace(0, 1, feather_size)[:, np.newaxis]
            mask[-feather_size:, :] *= np.linspace(1, 0, feather_size)[:, np.newaxis]
            mask[:, :feather_size] *= np.linspace(0, 1, feather_size)[np.newaxis, :]
            mask[:, -feather_size:] *= np.linspace(1, 0, feather_size)[np.newaxis, :]
            
            # Blend the processed chunk
            mask_3channel = np.stack([mask, mask, mask], axis=2)
            output_frame[y_start:y_end, x_start:x_end] = (
                blurred_chunk * mask_3channel + 
                output_frame[y_start:y_end, x_start:x_end] * (1 - mask_3channel)
            ).astype(np.uint8)
        
        out.write(output_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    out.release()
    
    print(f"Output saved to {output_path}")
    
    # Print statistics
    stats = placer.get_chunk_coverage_stats(heat_map)
    print(f"\nChunking Statistics:")
    print(f"- Coverage: {stats['coverage']*100:.1f}%")
    print(f"- Number of chunks: {stats['num_chunks']}")
    print(f"- Efficiency: {stats['efficiency']:.2f} activity per chunk")


def process_with_custom_detection(video_path, output_path):
    """
    Example using custom detection for activity mapping.
    """
    print("Using custom detection (detecting bright areas)...")
    
    # Create custom analyzer
    analyzer = HeatMapAnalyzer()
    
    # Generate heat map based on brightness
    cap = cv2.VideoCapture(video_path)
    heat_map = None
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect bright areas (e.g., for highlighting effects)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        if heat_map is None:
            heat_map = bright_mask.astype(np.float32) / 255.0
        else:
            heat_map += bright_mask.astype(np.float32) / 255.0
        
        frame_count += 1
    
    cap.release()
    
    # Normalize
    heat_map = heat_map / frame_count
    
    # Apply smart chunking
    apply_blur_effect_with_smart_chunks(video_path, output_path, heat_map)


def main():
    """Main function demonstrating different usage patterns."""
    
    # Example video path (replace with your video)
    video_path = "input_video.mp4"
    output_dir = "smart_chunk_output"
    
    if not os.path.exists(video_path):
        print(f"Please provide a video file at: {video_path}")
        print("Or modify the video_path variable in main()")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Smart Chunking System Demo")
    print("=" * 50)
    
    # Example 1: Motion-based chunking
    print("\n1. Motion-based smart chunking:")
    heat_map = generate_motion_heat_map(video_path, output_dir)
    output_path = os.path.join(output_dir, "motion_blur_output.mp4")
    apply_blur_effect_with_smart_chunks(video_path, output_path, heat_map)
    
    # Example 2: Custom detection
    print("\n2. Custom detection (bright areas):")
    output_path2 = os.path.join(output_dir, "brightness_blur_output.mp4")
    process_with_custom_detection(video_path, output_path2)
    
    print("\n✅ Demo complete! Check the output directory for results.")
    print(f"   Output directory: {os.path.abspath(output_dir)}")
    print("\nYou can adapt this code for:")
    print("- Video effects processing")
    print("- Selective video enhancement")
    print("- Region-based video analysis")
    print("- Memory-efficient video processing")


if __name__ == "__main__":
    main()