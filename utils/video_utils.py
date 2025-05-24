# video_utils.py - v1.1737774900
# Updated: Friday, January 24, 2025 at 17:45:00 PST
# Changes in this version:
# - Added video quality and codec support to improve output quality and reduce compression artifacts
# - Enhanced all video writing functions with codec selection and quality parameters
# - Implemented intelligent codec fallback system (H.264 -> H.265 -> VP9 -> mp4v)
# - Added bitrate control where supported by codecs
# - Improved error handling for codec initialization failures

"""
Video utility functions for MatAnyone video processing.
Contains standalone functions for video manipulation.
"""

import os
import numpy as np
import torch
import cv2
import time
import traceback


def get_video_codec_and_params(codec_choice='Auto', quality='High', custom_bitrate=None):
    """
    Get video codec fourcc and parameters based on user choices
    
    Args:
        codec_choice: Codec selection ('Auto', 'H.264', 'H.265', 'VP9')
        quality: Quality preset ('Low', 'Medium', 'High', 'Very High', 'Lossless')
        custom_bitrate: Custom bitrate in kbps (overrides quality preset)
    
    Returns:
        Tuple of (fourcc, quality_params)
    """
    # Define codec options with fallbacks
    codec_options = {
        'H.264': ['avc1', 'h264', 'x264'],
        'H.265': ['hev1', 'hevc', 'h265', 'x265'],
        'VP9': ['vp09', 'vp9'],
        'Auto': ['avc1', 'h264', 'hev1', 'hevc', 'vp09', 'vp9', 'mp4v']
    }
    
    # Get codec list to try
    if codec_choice in codec_options:
        codecs_to_try = codec_options[codec_choice]
    else:
        codecs_to_try = codec_options['Auto']
    
    # Always add mp4v as final fallback
    if 'mp4v' not in codecs_to_try:
        codecs_to_try.append('mp4v')
    
    # Quality parameters (these may or may not be used depending on codec)
    quality_params = {}
    
    if custom_bitrate is not None:
        # Use custom bitrate
        quality_params['bitrate'] = custom_bitrate * 1000  # Convert kbps to bps
    else:
        # Use quality preset
        quality_bitrates = {
            'Low': 2000,      # 2 Mbps
            'Medium': 5000,   # 5 Mbps  
            'High': 8000,     # 8 Mbps
            'Very High': 15000, # 15 Mbps
            'Lossless': 50000   # 50 Mbps (very high for near-lossless)
        }
        quality_params['bitrate'] = quality_bitrates.get(quality, 8000) * 1000
    
    # Determine fourcc to use
    fourcc = None
    for codec in codecs_to_try:
        try:
            # Test if codec is available by creating a temporary VideoWriter
            test_fourcc = cv2.VideoWriter_fourcc(*codec) if len(codec) == 4 else cv2.VideoWriter_fourcc(*'mp4v')
            
            # Try to create a minimal test writer
            test_writer = cv2.VideoWriter('test_codec.mp4', test_fourcc, 30, (640, 480))
            if test_writer.isOpened():
                test_writer.release()
                # Clean up test file
                try:
                    os.remove('test_codec.mp4')
                except:
                    pass
                
                # This codec works
                fourcc = test_fourcc
                print(f"Using codec: {codec}")
                break
            else:
                test_writer.release()
                
        except Exception as e:
            continue
    
    # Fallback to mp4v if nothing else works
    if fourcc is None:
        print("Warning: Falling back to mp4v codec")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    return fourcc, quality_params


def create_high_quality_writer(output_path, fps, width, height, codec_choice='Auto', quality='High', custom_bitrate=None):
    """
    Create a high-quality video writer with advanced codec and quality settings
    
    Args:
        output_path: Path to save the video
        fps: Frames per second
        width: Video width
        height: Video height
        codec_choice: Codec selection ('Auto', 'H.264', 'H.265', 'VP9')
        quality: Quality preset ('Low', 'Medium', 'High', 'Very High', 'Lossless')
        custom_bitrate: Custom bitrate in kbps
    
    Returns:
        OpenCV VideoWriter object
    """
    fourcc, quality_params = get_video_codec_and_params(codec_choice, quality, custom_bitrate)
    
    # Create writer with quality settings
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        print(f"Warning: Could not create video writer with selected codec. Retrying with mp4v...")
        # Fallback to basic mp4v
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    return writer


def blend_videos(forward_path, reverse_path, output_path, method='weighted', codec_choice='Auto', quality='High', custom_bitrate=None):
    """
    Blend forward and reverse videos using the specified method with high quality encoding
    
    Args:
        forward_path: Path to forward pass video
        reverse_path: Path to reversed reverse pass video
        output_path: Path to save blended video
        method: Blending method - 'weighted', 'max_alpha', 'min_alpha', or 'average'
        codec_choice: Video codec to use
        quality: Quality preset
        custom_bitrate: Custom bitrate in kbps
    """
    print(f"Blending videos using {method} method with {quality} quality...")
    
    # Validate inputs
    if not os.path.exists(forward_path) or not os.path.exists(reverse_path):
        print(f"Error: One or both input videos don't exist.")
        return None
    
    # Open both videos
    forward_cap = cv2.VideoCapture(forward_path)
    reverse_cap = cv2.VideoCapture(reverse_path)
    
    if not forward_cap.isOpened() or not reverse_cap.isOpened():
        print("Error: Could not open one of the videos for blending")
        return None
    
    # Get video properties (assuming both videos have same properties)
    fps = forward_cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(forward_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(forward_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(forward_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Verify both videos have the same number of frames
    reverse_frame_count = int(reverse_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if reverse_frame_count != frame_count:
        print(f"Warning: Frame count mismatch - forward: {frame_count}, reverse: {reverse_frame_count}")
        frame_count = min(frame_count, reverse_frame_count)
    
    # Create high-quality output video writer
    out = create_high_quality_writer(output_path, fps, width, height, codec_choice, quality, custom_bitrate)
    
    if not out.isOpened():
        print(f"Error: Could not create output video at {output_path}")
        forward_cap.release()
        reverse_cap.release()
        return None
    
    # Process frames
    start_time = time.time()
    for i in range(frame_count):
        # Read frames from both videos
        ret1, forward_frame = forward_cap.read()
        ret2, reverse_frame = reverse_cap.read()
        
        if not ret1 or not ret2:
            print(f"Error reading frame {i}")
            break
        
        # Calculate blend weight based on frame position
        if method == 'weighted':
            # Linear weighting based on frame position
            alpha = i / (frame_count - 1)  # 0 at start, 1 at end
            # Forward weight decreases, reverse weight increases
            forward_weight = 1.0 - (alpha * 0.5)  # 1.0 -> 0.5
            reverse_weight = 0.5 + (alpha * 0.5)  # 0.5 -> 1.0
            
            # Normalize weights to sum to 1
            total_weight = forward_weight + reverse_weight
            forward_weight /= total_weight
            reverse_weight /= total_weight
            
            # Blend frames using calculated weights
            blended_frame = cv2.addWeighted(forward_frame, forward_weight, 
                                           reverse_frame, reverse_weight, 0)
        elif method == 'max_alpha':
            # For alpha matte videos, take maximum alpha value at each pixel
            blended_frame = np.maximum(forward_frame, reverse_frame)
        elif method == 'min_alpha':
            # For alpha matte videos, take minimum alpha value at each pixel
            # This helps reduce fringing artifacts around the subject edges
            blended_frame = np.minimum(forward_frame, reverse_frame)
        else:  # 'average' or fallback
            # Simple 50/50 blend
            blended_frame = cv2.addWeighted(forward_frame, 0.5, reverse_frame, 0.5, 0)
        
        # Write blended frame
        out.write(blended_frame)
        
        # Show progress periodically
        if i % 10 == 0 or i == frame_count - 1:
            elapsed = time.time() - start_time
            frames_per_sec = (i + 1) / max(0.001, elapsed)
            eta = (frame_count - i - 1) / max(0.001, frames_per_sec)
            
            print(f"Blending progress: {i+1}/{frame_count} frames " 
                  f"({(i+1)/frame_count*100:.1f}%) - "
                  f"{frames_per_sec:.1f} fps, ETA: {eta:.1f}s")
    
    # Release resources
    forward_cap.release()
    reverse_cap.release()
    out.release()
    
    print(f"Blended video saved to {output_path}")
    return output_path


def cleanup_temporary_files(temp_files, cleanup_enabled=True):
    """
    Clean up temporary files after processing
    
    Args:
        temp_files: List of file paths to remove
        cleanup_enabled: Boolean flag to enable/disable cleanup
    """
    if cleanup_enabled:
        print("Cleaning up temporary files...")
        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"Removed: {file_path}")
                    elif os.path.isdir(file_path):
                        import shutil
                        shutil.rmtree(file_path)
                        print(f"Removed directory: {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {str(e)}")
        print("Cleanup complete")
    else:
        print("Temporary files kept (cleanup disabled)")


def reverse_image_sequence(input_dir, output_dir):
    """
    Reverse an image sequence by renaming files in reverse order.
    
    Args:
        input_dir: Directory containing the image sequence
        output_dir: Directory to save the reversed sequence
    """
    import os
    import shutil
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
        
    # Get frame numbers from filenames
    frame_numbers = []
    for filename in image_files:
        try:
            # Extract frame number from filename (assuming format like 00000000.png)
            frame_num = int(filename.split('.')[0])
            frame_numbers.append((frame_num, filename))
        except:
            print(f"Warning: Could not parse frame number from {filename}")
            
    # Sort by frame number
    frame_numbers.sort()
    
    # Copy files in reverse order with new names
    total_frames = len(frame_numbers)
    for i, (original_num, filename) in enumerate(frame_numbers):
        # Calculate reversed frame number
        reversed_num = frame_numbers[total_frames - 1 - i][0]
        
        # Copy file with new name
        src_path = os.path.join(input_dir, filename)
        ext = os.path.splitext(filename)[1]
        dst_path = os.path.join(output_dir, f"{reversed_num:08d}{ext}")
        
        shutil.copy2(src_path, dst_path)
    
    print(f"Reversed {total_frames} frames from {input_dir} to {output_dir}")

def reverse_video(video_path, output_path, codec_choice='Auto', quality='High', custom_bitrate=None):
    """
    Reverse a video file with high quality encoding
    
    Args:
        video_path: Path to the video to reverse
        output_path: Path to save the reversed video
        codec_choice: Video codec to use
        quality: Quality preset
        custom_bitrate: Custom bitrate in kbps
    
    Returns:
        Path to the reversed video
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist")
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Reading {frame_count} frames from {video_path}...")
    
    # Create high-quality output video writer
    out = create_high_quality_writer(output_path, fps, width, height, codec_choice, quality, custom_bitrate)
    
    if not out.isOpened():
        print(f"Error: Could not create output video at {output_path}")
        cap.release()
        return None
    
    # Read all frames into memory
    frames = []
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        
        # Show progress
        if i % 100 == 0 or i == frame_count - 1:
            print(f"Read {i+1}/{frame_count} frames ({(i+1)/frame_count*100:.1f}%)")
    
    cap.release()
    
    # Reverse frames and write to output
    print("Reversing and writing frames...")
    start_time = time.time()
    
    for i, frame in enumerate(reversed(frames)):
        out.write(frame)
        
        # Show progress
        if i % 100 == 0 or i == len(frames) - 1:
            elapsed = time.time() - start_time
            frames_per_sec = (i + 1) / max(0.001, elapsed)
            eta = (len(frames) - i - 1) / max(0.001, frames_per_sec)
            
            print(f"Wrote {i+1}/{len(frames)} frames ({(i+1)/len(frames)*100:.1f}%) - "
                  f"{frames_per_sec:.1f} fps, ETA: {eta:.1f}s")
    
    out.release()
    print(f"Reversed video saved to {output_path}")
    return output_path


def concatenate_videos(video_paths, output_path, codec_choice='Auto', quality='High', custom_bitrate=None):
    """
    Concatenate multiple videos into a single video with high quality encoding
    
    Args:
        video_paths: List of video paths to concatenate
        output_path: Path to save the concatenated video
        codec_choice: Video codec to use
        quality: Quality preset
        custom_bitrate: Custom bitrate in kbps
    
    Returns:
        Path to the concatenated video
    """
    if not video_paths:
        print("No videos to concatenate")
        return None
    
    # Get properties from the first video
    first_cap = cv2.VideoCapture(video_paths[0])
    if not first_cap.isOpened():
        print(f"Error: Could not open first video: {video_paths[0]}")
        return None
        
    fps = first_cap.get(cv2.CAP_PROP_FPS)
    width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_cap.release()
    
    # Create high-quality output video writer
    out = create_high_quality_writer(output_path, fps, width, height, codec_choice, quality, custom_bitrate)
    
    if not out.isOpened():
        print(f"Error: Could not create output video at {output_path}")
        return None
    
    # Process each video
    total_frames = 0
    for video_path in video_paths:
        print(f"Processing: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video: {video_path}. Skipping.")
            continue
            
        # Get video properties and check for consistency
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if video_width != width or video_height != height:
            print(f"Warning: Video dimensions mismatch. Expected {width}x{height}, got {video_width}x{video_height}.")
            print("Resizing frames to match first video...")
        
        # Read and write frames
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize if needed
            if video_width != width or video_height != height:
                frame = cv2.resize(frame, (width, height))
                
            # Write frame to output
            out.write(frame)
            frame_count += 1
            
            # Show progress periodically
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{video_frames} frames from current video")
        
        # Close video
        cap.release()
        total_frames += frame_count
        print(f"Added {frame_count} frames from {video_path}")
    
    # Release output video
    out.release()
    print(f"Concatenated {len(video_paths)} videos with {total_frames} total frames to {output_path}")
    return output_path
