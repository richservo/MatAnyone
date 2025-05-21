"""
# chunking_utils.py - v1.1715883650
# Updated: Thursday, May 16, 2025
# Changes in this version:
# - Fixed auto-chunking to ensure all chunks have identical dimensions, matching low-res mask resolution
# - Improved chunk creation to maintain consistent size even at image boundaries
# - Enhanced overlap handling to ensure proper coverage without dimension changes
# - Added better diagnostics and error checking for chunk dimensions

Utility functions for chunk-based video processing in MatAnyone.
Contains functions for managing video chunks and chunk-related preprocessing.
"""

import os
import cv2
import numpy as np
import time
import traceback


def get_strip_chunks(width, height, num_chunks, model_factor):
    """
    Get horizontal strip chunk segments
    
    Args:
        width: Frame width
        height: Frame height
        num_chunks: Number of horizontal chunks
        model_factor: Factor for dimension divisibility
    
    Returns:
        List of chunk dictionaries with x_range and y_range
    """
    # Calculate base width for each chunk, ensuring divisibility by model_factor
    base_chunk_width = width // num_chunks
    base_chunk_width = (base_chunk_width // model_factor) * model_factor
    base_chunk_width = max(model_factor * 4, base_chunk_width)  # Ensure minimum width
    
    # Define overlap for blending with special case for 2-chunk processing
    if num_chunks == 2:
        # Special case for 2 chunks - use larger overlap (33% of base width)
        overlap_pixels = base_chunk_width // 3
        overlap_pixels = (overlap_pixels // model_factor) * model_factor
        overlap_pixels = max(model_factor * 8, min(overlap_pixels, base_chunk_width // 2))
        print(f"Using enhanced overlap of {overlap_pixels} pixels for 2-chunk processing")
    else:
        # Standard overlap for 3+ chunks (20% of base width)
        overlap_pixels = base_chunk_width // 5
        overlap_pixels = (overlap_pixels // model_factor) * model_factor
        overlap_pixels = max(model_factor * 8, min(overlap_pixels, base_chunk_width // 3))
        print(f"Using {overlap_pixels} pixels overlap between strips")
    
    # Define chunk segments with safer calculations
    chunk_segments = []
    for i in range(num_chunks):
        # Calculate start position
        if i == 0:
            start_x = 0
        else:
            start_x = i * base_chunk_width - overlap_pixels
            start_x = max(0, (start_x // model_factor) * model_factor)
        
        # Calculate end position
        if i == num_chunks - 1:
            end_x = width
        else:
            end_x = (i + 1) * base_chunk_width + overlap_pixels
            end_x = min((end_x // model_factor) * model_factor, width)
        
        # Validate chunk dimensions
        if end_x <= start_x or end_x > width or start_x < 0:
            print(f"Warning: Invalid chunk dimensions: start={start_x}, end={end_x}. Skipping this chunk.")
            continue
        
        # Ensure width is divisible by model_factor
        chunk_width = end_x - start_x
        
        # Only add the chunk if width >= model_factor * 4
        if chunk_width >= model_factor * 4:
            chunk_info = {
                'x_range': (start_x, end_x),
                'y_range': (0, height),  # Full height for strips
                'width': chunk_width,
                'height': height
            }
            chunk_segments.append(chunk_info)
            print(f"Strip {len(chunk_segments)}: X={start_x} to {end_x}, Width: {end_x - start_x}")
    
    return chunk_segments


def get_grid_chunks(width, height, num_chunks, model_factor):
    """
    Get grid-based chunks that preserve aspect ratio
    
    Args:
        width: Frame width
        height: Frame height
        num_chunks: Target number of chunks (actual might differ slightly)
        model_factor: Factor for dimension divisibility
    
    Returns:
        List of chunk dictionaries with x_range and y_range
    """
    # Calculate grid dimensions based on aspect ratio
    aspect_ratio = width / height
    
    # Determine number of rows and columns based on aspect ratio and target number of chunks
    if aspect_ratio >= 1:  # Wider than tall
        # For landscape videos, use more columns than rows
        # Try to match the video's aspect ratio in the grid layout
        grid_cols = int(np.ceil(np.sqrt(num_chunks * aspect_ratio)))
        grid_rows = int(np.ceil(num_chunks / grid_cols))
    else:  # Taller than wide
        # For portrait videos, use more rows than columns
        grid_rows = int(np.ceil(np.sqrt(num_chunks / aspect_ratio)))
        grid_cols = int(np.ceil(num_chunks / grid_rows))
    
    # Ensure we have at least 2 rows and 2 columns for a grid
    grid_rows = max(2, grid_rows)
    grid_cols = max(2, grid_cols)
    
    print(f"Creating a {grid_rows}x{grid_cols} grid of chunks (aspect ratio: {aspect_ratio:.2f})")
    
    # Calculate base dimensions
    base_chunk_width = width // grid_cols
    base_chunk_height = height // grid_rows
    
    # Ensure dimensions are divisible by model_factor
    base_chunk_width = (base_chunk_width // model_factor) * model_factor
    base_chunk_height = (base_chunk_height // model_factor) * model_factor
    
    # Calculate overlaps (20% of chunk size)
    overlap_x = max(model_factor * 2, base_chunk_width // 5)
    overlap_y = max(model_factor * 2, base_chunk_height // 5)
    
    # Ensure overlaps are divisible by model_factor
    overlap_x = (overlap_x // model_factor) * model_factor
    overlap_y = (overlap_y // model_factor) * model_factor
    
    print(f"Using overlaps of {overlap_x}x{overlap_y} pixels between grid cells")
    
    # Generate grid chunks
    chunk_segments = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            # Calculate chunk boundaries with overlap
            if col == 0:
                start_x = 0
            else:
                start_x = col * base_chunk_width - overlap_x
                start_x = max(0, (start_x // model_factor) * model_factor)
            
            if row == 0:
                start_y = 0
            else:
                start_y = row * base_chunk_height - overlap_y
                start_y = max(0, (start_y // model_factor) * model_factor)
            
            if col == grid_cols - 1:
                end_x = width
            else:
                end_x = (col + 1) * base_chunk_width + overlap_x
                end_x = min((end_x // model_factor) * model_factor, width)
            
            if row == grid_rows - 1:
                end_y = height
            else:
                end_y = (row + 1) * base_chunk_height + overlap_y
                end_y = min((end_y // model_factor) * model_factor, height)
            
            # Validate chunk dimensions
            if end_x <= start_x or end_y <= start_y or start_x < 0 or start_y < 0 or end_x > width or end_y > height:
                print(f"Warning: Invalid chunk dimensions at grid ({row},{col}): X={start_x}-{end_x}, Y={start_y}-{end_y}. Skipping.")
                continue
            
            # Ensure dimensions are divisible by model_factor
            chunk_width = end_x - start_x
            chunk_height = end_y - start_y
            
            # Only add the chunk if dimensions are sufficient
            if chunk_width >= model_factor * 4 and chunk_height >= model_factor * 4:
                chunk_info = {
                    'x_range': (start_x, end_x),
                    'y_range': (start_y, end_y),
                    'width': chunk_width,
                    'height': chunk_height,
                    'grid_pos': (row, col)
                }
                chunk_segments.append(chunk_info)
                print(f"Grid chunk ({row},{col}): X={start_x}-{end_x}, Y={start_y}-{end_y}, Size: {chunk_width}x{chunk_height}")
    
    return chunk_segments


def get_autochunk_segments(width, height, low_res_width, low_res_height, model_factor):
    """
    Automatically determine optimal chunk segments based on low-resolution dimensions
    
    This creates chunks based on the low-res dimensions, ensuring ALL chunks have 
    identical dimensions matching the low-res mask, with appropriate overlap.
    
    Args:
        width: Full frame width
        height: Full frame height
        low_res_width: Low resolution width
        low_res_height: Low resolution height
        model_factor: Factor for dimension divisibility
    
    Returns:
        List of chunk dictionaries with x_range and y_range
    """
    print(f"Auto-chunking using low-res dimensions as chunk size: {low_res_width}x{low_res_height}")
    
    # Ensure low-res dimensions are at least minimally sized and divisible by model_factor
    low_res_width = max(model_factor * 4, (low_res_width // model_factor) * model_factor)
    low_res_height = max(model_factor * 4, (low_res_height // model_factor) * model_factor)
    
    # Calculate overlap (25% of chunk size for better blending)
    # These overlaps will be larger than standard to ensure smooth transitions
    overlap_x = max(model_factor * 4, low_res_width // 4)
    overlap_y = max(model_factor * 4, low_res_height // 4)
    
    # Ensure overlaps are divisible by model_factor
    overlap_x = (overlap_x // model_factor) * model_factor
    overlap_y = (overlap_y // model_factor) * model_factor
    
    # Calculate effective step size (distance between chunk starts)
    # This is the chunk size minus the overlap
    step_x = low_res_width - overlap_x
    step_y = low_res_height - overlap_y
    
    # Make sure steps are at least model_factor * 4
    step_x = max(model_factor * 4, step_x)
    step_y = max(model_factor * 4, step_y)
    
    # Calculate how many chunks we need to cover the entire frame with proper overlap
    # Note: We use ceil to ensure we have enough chunks to cover the entire frame
    num_chunks_x = max(1, int(np.ceil((width - low_res_width) / step_x)) + 1)
    num_chunks_y = max(1, int(np.ceil((height - low_res_height) / step_y)) + 1)
    
    print(f"Auto-chunking will create a {num_chunks_y}x{num_chunks_x} grid")
    print(f"Using consistent chunk size of {low_res_width}x{low_res_height} for all chunks")
    print(f"Using overlaps of {overlap_x}x{overlap_y} pixels between chunks")
    
    # Generate chunks with CONSISTENT size (this is the key fix)
    chunk_segments = []
    for row in range(num_chunks_y):
        for col in range(num_chunks_x):
            # Calculate chunk boundaries
            start_x = col * step_x
            start_y = row * step_y
            
            # Ensure start coordinates are divisible by model_factor
            start_x = (start_x // model_factor) * model_factor
            start_y = (start_y // model_factor) * model_factor
            
            # Calculate end coordinates - CRITICAL: maintain consistent chunk size
            # This is the key change: ALL chunks must be exactly low_res dimensions
            end_x = start_x + low_res_width
            end_y = start_y + low_res_height
            
            # Handle boundary cases - if chunk exceeds frame boundary, adjust start position
            # This ensures the chunk stays the correct size but moves inward
            if end_x > width:
                # Adjust start position to ensure chunk stays within frame while maintaining size
                start_x = width - low_res_width
                start_x = max(0, (start_x // model_factor) * model_factor)
                end_x = start_x + low_res_width
                
                # Skip if this creates a duplicate chunk that's already covered
                if any(s['x_range'][0] == start_x and s['y_range'][0] == start_y for s in chunk_segments):
                    continue
            
            if end_y > height:
                # Adjust start position to ensure chunk stays within frame while maintaining size
                start_y = height - low_res_height
                start_y = max(0, (start_y // model_factor) * model_factor)
                end_y = start_y + low_res_height
                
                # Skip if this creates a duplicate chunk that's already covered
                if any(s['x_range'][0] == start_x and s['y_range'][0] == start_y for s in chunk_segments):
                    continue
            
            # Double-check chunk dimensions are exactly what we expect
            chunk_width = end_x - start_x
            chunk_height = end_y - start_y
            
            # Verify the chunk has correct dimensions
            if chunk_width != low_res_width or chunk_height != low_res_height:
                print(f"Warning: Chunk dimensions {chunk_width}x{chunk_height} don't match expected {low_res_width}x{low_res_height}")
                # Adjust to ensure exact dimensions
                end_x = start_x + low_res_width
                end_y = start_y + low_res_height
            
            # Validate chunk boundaries
            if start_x < 0 or start_y < 0 or end_x > width or end_y > height:
                print(f"Warning: Chunk exceeds frame boundaries at ({row},{col}): X={start_x}-{end_x}, Y={start_y}-{end_y}. Adjusting...")
                
                # Adjust the chunk to ensure it's within boundaries while maintaining size
                if start_x < 0:
                    start_x = 0
                    end_x = low_res_width
                
                if start_y < 0:
                    start_y = 0
                    end_y = low_res_height
                
                if end_x > width:
                    start_x = width - low_res_width
                    start_x = max(0, start_x)
                    end_x = start_x + low_res_width
                
                if end_y > height:
                    start_y = height - low_res_height
                    start_y = max(0, start_y)
                    end_y = start_y + low_res_height
            
            # Final verification of chunk dimensions
            chunk_width = end_x - start_x
            chunk_height = end_y - start_y
            
            # Only add valid chunks with the correct dimensions
            if (chunk_width == low_res_width and chunk_height == low_res_height and
                start_x >= 0 and start_y >= 0 and end_x <= width and end_y <= height):
                
                # Check for duplicates
                if not any(s['x_range'][0] == start_x and s['y_range'][0] == start_y for s in chunk_segments):
                    chunk_info = {
                        'x_range': (start_x, end_x),
                        'y_range': (start_y, end_y),
                        'width': chunk_width,
                        'height': chunk_height,
                        'grid_pos': (row, col)
                    }
                    chunk_segments.append(chunk_info)
                    print(f"Auto chunk ({row},{col}): X={start_x}-{end_x}, Y={start_y}-{end_y}, Size: {chunk_width}x{chunk_height}")
    
    print(f"Auto-chunking created {len(chunk_segments)} chunks, all with identical dimensions of {low_res_width}x{low_res_height}")
    
    # Additional verification step
    for i, chunk in enumerate(chunk_segments):
        chunk_width = chunk['width']
        chunk_height = chunk['height']
        if chunk_width != low_res_width or chunk_height != low_res_height:
            print(f"ERROR: Chunk {i} has incorrect dimensions: {chunk_width}x{chunk_height}, expected {low_res_width}x{low_res_height}")
    
    return chunk_segments


def create_low_res_video(input_path, output_path, width, height, fps):
    """
    Create a low-resolution version of the input video
    
    Args:
        input_path: Path to the input video
        output_path: Path to save the low-res video
        width: Width of the low-res video
        height: Height of the low-res video
        fps: Frames per second
    
    Returns:
        Path to the low-res video
    """
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise ValueError(f"Could not create output video: {output_path}")
        
        # Process frames
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame
            low_res_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            
            # Write to output
            out.write(low_res_frame)
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Resized {frame_idx}/{frame_count} frames to low resolution")
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"Created low-resolution video at {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error creating low-resolution video: {str(e)}")
        traceback.print_exc()
        return None
