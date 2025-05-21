# extraction_utils.py - v1.1737779600
# Updated: Friday, January 24, 2025 at 19:13:20 PST
# Changes in this version:
# - Fixed extract_last_frame to better handle corrupted or unreadable video files
# - Added comprehensive file existence and readability checks before attempting to open videos
# - Enhanced error handling with detailed logging for debugging video processing issues
# - Added fallback mechanisms for cases where video files are corrupted or partially written
# - Improved compatibility with various video formats and codecs
# - Added file size validation to detect empty or incomplete video files

"""
Extraction utilities for MatAnyone video processing.
Contains functions for extracting frames and segments from videos.
"""

import os
import cv2
import numpy as np
import traceback
import glob
import time

# Import video utilities for high-quality video writing
from utils.video_utils import create_high_quality_writer


def extract_frame(video_path, frame_idx, output_path=None):
    """
    Extract a specific frame from a video
    
    Args:
        video_path: Path to the video file
        frame_idx: Index of the frame to extract (0-based)
        output_path: Path to save the extracted frame (optional)
    
    Returns:
        The extracted frame as numpy array, and path to saved frame if output_path was provided
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get total frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Ensure frame index is within valid range
        if frame_idx < 0 or frame_idx >= frame_count:
            print(f"Warning: Frame index {frame_idx} out of range [0, {frame_count-1}]. Clamping.")
            frame_idx = max(0, min(frame_idx, frame_count - 1))
        
        # Seek to the specified frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
        
        # Save frame if output path provided
        if output_path:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, frame)
            print(f"Frame {frame_idx} extracted and saved to {output_path}")
            result_path = output_path
        else:
            result_path = None
        
        # Release the video capture
        cap.release()
        
        return frame, result_path
    
    except Exception as e:
        print(f"Error extracting frame {frame_idx}: {str(e)}")
        traceback.print_exc()
        return None, None


def extract_last_frame(video_or_dir_path, output_dir, name_prefix):
    """
    Extract the last frame from a video file or from a directory of image frames
    
    Args:
        video_or_dir_path: Path to the video file or directory containing image frames
        output_dir: Directory to save the frame
        name_prefix: Prefix for the output filename
    
    Returns:
        Path to the extracted frame
    """
    try:
        # First, check if the path actually exists
        if not os.path.exists(video_or_dir_path):
            raise ValueError(f"Path does not exist: {video_or_dir_path}")
        
        # Determine if input is a video file or directory
        if os.path.isfile(video_or_dir_path):
            # Handle video file
            print(f"Extracting last frame from video: {video_or_dir_path}")
            
            # Check if the file has a reasonable size (not empty or too small)
            file_size = os.path.getsize(video_or_dir_path)
            if file_size < 1024:  # Less than 1KB is suspicious for a video file
                print(f"Warning: Video file is very small ({file_size} bytes), might be corrupted: {video_or_dir_path}")
            
            # Wait a bit in case the file is still being written
            if file_size < 10240:  # If less than 10KB, wait to see if it's still being written
                print(f"Waiting for file to finish writing...")
                time.sleep(2)
                new_size = os.path.getsize(video_or_dir_path)
                if new_size != file_size:
                    print(f"File size changed from {file_size} to {new_size}, waiting a bit more...")
                    time.sleep(3)
            
            # Try to open the video with detailed error handling
            cap = cv2.VideoCapture(video_or_dir_path)
            if not cap.isOpened():
                # Try different backends if the default fails
                backends_to_try = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
                cap_opened = False
                
                for backend in backends_to_try:
                    try:
                        cap = cv2.VideoCapture(video_or_dir_path, backend)
                        if cap.isOpened():
                            print(f"Successfully opened video with backend {backend}")
                            cap_opened = True
                            break
                        cap.release()
                    except Exception as e:
                        print(f"Failed to open with backend {backend}: {str(e)}")
                        continue
                
                if not cap_opened:
                    raise ValueError(f"Could not open video with any backend: {video_or_dir_path}")
            
            # Check if the video has any frames
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0:
                cap.release()
                raise ValueError(f"Video has no frames: {video_or_dir_path}")
            
            print(f"Video has {frame_count} frames")
            
            # Try to seek to the last frame
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            except Exception as e:
                print(f"Warning: Could not seek to last frame, will try reading sequentially: {str(e)}")
            
            # Try to read the frame
            ret, frame = cap.read()
            
            # If seeking failed, try to read the last frame by reading through the video
            if not ret and frame_count > 1:
                print("Direct seek failed, trying sequential read...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Go to start
                last_frame = None
                
                # Read through frames to get the last valid one
                for i in range(min(frame_count, 100)):  # Don't read more than 100 frames
                    ret, current_frame = cap.read()
                    if ret:
                        last_frame = current_frame
                    else:
                        break
                
                if last_frame is not None:
                    frame = last_frame
                    ret = True
                else:
                    cap.release()
                    raise ValueError(f"Could not read any frames from: {video_or_dir_path}")
            
            if not ret:
                cap.release()
                raise ValueError(f"Could not read last frame from: {video_or_dir_path}")
            
            # Close video
            cap.release()
            
        elif os.path.isdir(video_or_dir_path):
            # Handle directory of image frames
            print(f"Extracting last frame from directory: {video_or_dir_path}")
            
            # Find all image files in the directory
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(video_or_dir_path, ext)))
                # Also try uppercase extensions
                image_files.extend(glob.glob(os.path.join(video_or_dir_path, ext.upper())))
            
            if not image_files:
                raise ValueError(f"No image files found in directory: {video_or_dir_path}")
            
            # Sort files to get the last one (assuming they're numbered)
            image_files.sort()
            last_image_path = image_files[-1]
            
            print(f"Using last image: {last_image_path}")
            
            # Read the last image
            frame = cv2.imread(last_image_path)
            if frame is None:
                raise ValueError(f"Could not read image: {last_image_path}")
        
        else:
            raise ValueError(f"Path is neither a file nor a directory: {video_or_dir_path}")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the frame
        output_path = os.path.join(output_dir, f"{name_prefix}_last_frame.png")
        
        # If the frame has 3 channels but is a mask, convert to grayscale
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            if np.all(frame[:,:,0] == frame[:,:,1]) and np.all(frame[:,:,1] == frame[:,:,2]):
                frame = frame[:,:,0]
        
        # Apply binary thresholding to ensure mask is either 0 or 255
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            _, frame = cv2.threshold(frame, 128, 255, cv2.THRESH_BINARY)
            
        # Save the frame
        success = cv2.imwrite(output_path, frame)
        if not success:
            raise ValueError(f"Failed to save frame to: {output_path}")
        
        print(f"Extracted last frame to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error extracting last frame: {str(e)}")
        traceback.print_exc()
        
        # Create a fallback empty mask
        try:
            print(f"Creating fallback empty mask...")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{name_prefix}_last_frame.png")
            
            # Try to determine appropriate dimensions
            fallback_width, fallback_height = 256, 256
            
            # If input was a video file, try to get its dimensions
            if os.path.isfile(video_or_dir_path):
                try:
                    cap = cv2.VideoCapture(video_or_dir_path)
                    if cap.isOpened():
                        fallback_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        fallback_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()
                        if fallback_width <= 0 or fallback_height <= 0:
                            fallback_width, fallback_height = 256, 256
                except:
                    pass
            
            # Create a small empty mask as fallback
            empty_mask = np.zeros((fallback_height, fallback_width), dtype=np.uint8)
            cv2.imwrite(output_path, empty_mask)
            print(f"Created fallback empty mask at {output_path} with dimensions {fallback_width}x{fallback_height}")
            return output_path
        except Exception as fallback_error:
            print(f"Error creating fallback mask: {str(fallback_error)}")
            return None


def extract_chunk_frame_range(input_path, output_path, start_frame, end_frame, start_x, end_x, start_y, end_y, fps,
                             video_codec='Auto', video_quality='High', custom_bitrate=None):
    """
    Extract a range of frames from a video and crop to the specified region with high-quality encoding
    
    Args:
        input_path: Path to the input video
        output_path: Path to save the output video
        start_frame: Starting frame index
        end_frame: Ending frame index
        start_x: Start X coordinate for cropping
        end_x: End X coordinate for cropping
        start_y: Start Y coordinate for cropping
        end_y: End Y coordinate for cropping
        fps: Frames per second for the output video
        video_codec: Video codec to use
        video_quality: Quality preset
        custom_bitrate: Custom bitrate in kbps
        
    Returns:
        Path to the output video
    """
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Calculate expected dimensions
        chunk_width = end_x - start_x
        chunk_height = end_y - start_y
        
        # Validate expected dimensions
        if chunk_width <= 0 or chunk_height <= 0:
            raise ValueError(f"Invalid chunk dimensions: {chunk_width}x{chunk_height}")
        
        print(f"Extracting chunk with dimensions {chunk_width}x{chunk_height} from position ({start_x},{start_y})")
        
        # Create high-quality output video writer
        out = create_high_quality_writer(output_path, fps, chunk_width, chunk_height,
                                        video_codec, video_quality, custom_bitrate)
        
        if not out.isOpened():
            raise ValueError(f"Could not create output video: {output_path}")
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Get original video dimensions
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Verify chunk boundaries
        if end_x > original_width or end_y > original_height:
            print(f"Warning: Chunk extends beyond video bounds. Video: {original_width}x{original_height}, Chunk: ({start_x},{start_y})-({end_x},{end_y})")
            
            # Adjust end coordinates but MAINTAIN chunk dimensions by shifting start position
            if end_x > original_width:
                start_x = max(0, original_width - chunk_width)
                end_x = start_x + chunk_width
            
            if end_y > original_height:
                start_y = max(0, original_height - chunk_height)
                end_y = start_y + chunk_height
                
            print(f"Adjusted chunk position to ({start_x},{start_y})-({end_x},{end_y}) to maintain {chunk_width}x{chunk_height} dimensions")
            
            # Final validation that chunk is within bounds
            if start_x < 0 or start_y < 0 or end_x > original_width or end_y > original_height:
                # Create a blank chunk as fallback
                print(f"Creating blank chunk of size {chunk_width}x{chunk_height} as fallback")
                for _ in range(end_frame - start_frame):
                    blank_frame = np.zeros((chunk_height, chunk_width, 3), dtype=np.uint8)
                    out.write(blank_frame)
                out.release()
                return output_path
        
        # Extract frames
        frame_count = 0
        total_frames = end_frame - start_frame
        
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Handle edge cases where frame might be smaller than expected
            if frame.shape[0] < end_y or frame.shape[1] < end_x:
                print(f"Warning: Frame {frame_idx} is smaller than expected: {frame.shape[1]}x{frame.shape[0]}")
                
                # Create a canvas of the right size
                canvas = np.zeros((original_height, original_width, 3), dtype=np.uint8)
                # Copy the frame data that exists
                canvas[:frame.shape[0], :frame.shape[1]] = frame
                frame = canvas
            
            # Ensure we don't exceed frame bounds
            safe_end_x = min(end_x, frame.shape[1])
            safe_end_y = min(end_y, frame.shape[0])
            safe_width = safe_end_x - start_x
            safe_height = safe_end_y - start_y
            
            # Extract region
            try:
                # Handle case where chunk might extend beyond frame
                if safe_width <= 0 or safe_height <= 0 or start_x >= frame.shape[1] or start_y >= frame.shape[0]:
                    # Create an empty frame of the correct size
                    region = np.zeros((chunk_height, chunk_width, 3), dtype=np.uint8)
                else:
                    # Extract region from frame
                    region = frame[start_y:safe_end_y, start_x:safe_end_x].copy()
                    
                    # If extracted region is smaller than expected, pad it
                    if region.shape[0] != chunk_height or region.shape[1] != chunk_width:
                        print(f"Resizing extracted region from {region.shape[1]}x{region.shape[0]} to {chunk_width}x{chunk_height}")
                        
                        # Create a canvas of the right size
                        padded_region = np.zeros((chunk_height, chunk_width, 3), dtype=np.uint8)
                        # Copy the extracted region
                        h, w = min(region.shape[0], chunk_height), min(region.shape[1], chunk_width)
                        padded_region[:h, :w] = region[:h, :w]
                        region = padded_region
            except Exception as e:
                print(f"Error extracting region from frame {frame_idx}: {str(e)}")
                # Create a blank frame of the correct size as fallback
                region = np.zeros((chunk_height, chunk_width, 3), dtype=np.uint8)
            
            # Final check to ensure correct dimensions
            if region.shape[0] != chunk_height or region.shape[1] != chunk_width:
                print(f"Resizing final region from {region.shape[1]}x{region.shape[0]} to {chunk_width}x{chunk_height}")
                region = cv2.resize(region, (chunk_width, chunk_height), interpolation=cv2.INTER_AREA)
            
            # Write to output
            out.write(region)
            frame_count += 1
            
            # Print progress
            if frame_count % 50 == 0:
                print(f"Extracted {frame_count}/{total_frames} frames for chunk ({start_x},{start_y})-({end_x},{end_y})")
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"Extracted {frame_count} frames from range {start_frame}-{end_frame} for chunk ({start_x},{start_y})-({end_x},{end_y})")
        return output_path
        
    except Exception as e:
        print(f"Error extracting frame range: {str(e)}")
        traceback.print_exc()
        
        # Try to create an empty video with the right dimensions as fallback
        try:
            chunk_width = end_x - start_x
            chunk_height = end_y - start_y
            
            # Create a blank chunk as fallback
            print(f"Creating blank chunk of size {chunk_width}x{chunk_height} as fallback after error")
            
            out = create_high_quality_writer(output_path, fps, chunk_width, chunk_height,
                                           video_codec, video_quality, custom_bitrate)
            
            for _ in range(end_frame - start_frame):
                blank_frame = np.zeros((chunk_height, chunk_width, 3), dtype=np.uint8)
                out.write(blank_frame)
            
            out.release()
            return output_path
            
        except Exception as fallback_error:
            print(f"Error creating fallback blank video: {str(fallback_error)}")
            return None


def extract_chunk_frame_range_reversed(input_path, output_path, start_frame, end_frame, start_x, end_x, start_y, end_y, fps,
                                      video_codec='Auto', video_quality='High', custom_bitrate=None):
    """
    Extract a range of frames from a video in reverse order and crop to the specified region with high-quality encoding
    
    Args:
        input_path: Path to the input video
        output_path: Path to save the output video
        start_frame: Starting frame index
        end_frame: Ending frame index
        start_x: Start X coordinate for cropping
        end_x: End X coordinate for cropping
        start_y: Start Y coordinate for cropping
        end_y: End Y coordinate for cropping
        fps: Frames per second for the output video
        video_codec: Video codec to use
        video_quality: Quality preset
        custom_bitrate: Custom bitrate in kbps
        
    Returns:
        Path to the output video
    """
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Calculate dimensions
        chunk_width = end_x - start_x
        chunk_height = end_y - start_y
        
        # Validate expected dimensions
        if chunk_width <= 0 or chunk_height <= 0:
            raise ValueError(f"Invalid chunk dimensions: {chunk_width}x{chunk_height}")
            
        print(f"Extracting reversed chunk with dimensions {chunk_width}x{chunk_height} from position ({start_x},{start_y})")
        
        # Create high-quality output video writer
        out = create_high_quality_writer(output_path, fps, chunk_width, chunk_height,
                                        video_codec, video_quality, custom_bitrate)
        
        if not out.isOpened():
            raise ValueError(f"Could not create output video: {output_path}")
        
        # Get original video dimensions
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Verify chunk boundaries
        if end_x > original_width or end_y > original_height:
            print(f"Warning: Chunk extends beyond video bounds. Video: {original_width}x{original_height}, Chunk: ({start_x},{start_y})-({end_x},{end_y})")
            
            # Adjust end coordinates but MAINTAIN chunk dimensions by shifting start position
            if end_x > original_width:
                start_x = max(0, original_width - chunk_width)
                end_x = start_x + chunk_width
            
            if end_y > original_height:
                start_y = max(0, original_height - chunk_height)
                end_y = start_y + chunk_height
                
            print(f"Adjusted chunk position to ({start_x},{start_y})-({end_x},{end_y}) to maintain {chunk_width}x{chunk_height} dimensions")
        
        # First, read all frames in the range into memory
        frames = []
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extract frames
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Ensure frame is large enough
            if frame.shape[0] < end_y or frame.shape[1] < end_x:
                print(f"Warning: Frame {frame_idx} is smaller than expected: {frame.shape[1]}x{frame.shape[0]}")
                
                # Create a canvas of the right size
                canvas = np.zeros((original_height, original_width, 3), dtype=np.uint8)
                # Copy the frame data that exists
                canvas[:frame.shape[0], :frame.shape[1]] = frame
                frame = canvas
            
            # Ensure we don't exceed frame bounds
            safe_end_x = min(end_x, frame.shape[1])
            safe_end_y = min(end_y, frame.shape[0])
            safe_width = safe_end_x - start_x
            safe_height = safe_end_y - start_y
            
            # Extract region
            try:
                # Handle case where chunk might extend beyond frame
                if safe_width <= 0 or safe_height <= 0 or start_x >= frame.shape[1] or start_y >= frame.shape[0]:
                    # Create an empty frame of the correct size
                    region = np.zeros((chunk_height, chunk_width, 3), dtype=np.uint8)
                else:
                    # Extract region from frame
                    region = frame[start_y:safe_end_y, start_x:safe_end_x].copy()
                    
                    # If extracted region is smaller than expected, pad it
                    if region.shape[0] != chunk_height or region.shape[1] != chunk_width:
                        print(f"Resizing extracted region from {region.shape[1]}x{region.shape[0]} to {chunk_width}x{chunk_height}")
                        
                        # Create a canvas of the right size
                        padded_region = np.zeros((chunk_height, chunk_width, 3), dtype=np.uint8)
                        # Copy the extracted region
                        h, w = min(region.shape[0], chunk_height), min(region.shape[1], chunk_width)
                        padded_region[:h, :w] = region[:h, :w]
                        region = padded_region
            except Exception as e:
                print(f"Error extracting region from frame {frame_idx}: {str(e)}")
                # Create a blank frame of the correct size as fallback
                region = np.zeros((chunk_height, chunk_width, 3), dtype=np.uint8)
            
            # Final check to ensure correct dimensions
            if region.shape[0] != chunk_height or region.shape[1] != chunk_width:
                print(f"Resizing final region from {region.shape[1]}x{region.shape[0]} to {chunk_width}x{chunk_height}")
                region = cv2.resize(region, (chunk_width, chunk_height), interpolation=cv2.INTER_AREA)
                
            frames.append(region)
                
        # Release input video
        cap.release()
        
        # Reverse the frames and write to output
        frames.reverse()
        
        for i, frame in enumerate(frames):
            out.write(frame)
            
            # Print progress
            if i % 50 == 0:
                print(f"Writing reversed frame {i+1}/{len(frames)} for chunk ({start_x},{start_y})-({end_x},{end_y})")
        
        # Release output video
        out.release()
        
        print(f"Extracted {len(frames)} frames in reverse order from range {start_frame}-{end_frame} for chunk ({start_x},{start_y})-({end_x},{end_y})")
        return output_path
        
    except Exception as e:
        print(f"Error extracting reversed frame range: {str(e)}")
        traceback.print_exc()
        
        # Try to create an empty video with the right dimensions as fallback
        try:
            chunk_width = end_x - start_x
            chunk_height = end_y - start_y
            
            # Create a blank chunk as fallback
            print(f"Creating blank reversed chunk of size {chunk_width}x{chunk_height} as fallback after error")
            
            out = create_high_quality_writer(output_path, fps, chunk_width, chunk_height,
                                           video_codec, video_quality, custom_bitrate)
            
            for _ in range(end_frame - start_frame):
                blank_frame = np.zeros((chunk_height, chunk_width, 3), dtype=np.uint8)
                out.write(blank_frame)
            
            out.release()
            return output_path
            
        except Exception as fallback_error:
            print(f"Error creating fallback blank reversed video: {str(fallback_error)}")
            return None


def create_full_video_from_ranges(range_results, frame_count, output_fgr_path, output_pha_path, width, height, fps,
                                 video_codec='Auto', video_quality='High', custom_bitrate=None):
    """
    Create full-length videos by filling gaps between ranges with empty frames using high-quality encoding
    
    Args:
        range_results: List of dictionaries with range info (start_frame, end_frame, fgr_path, pha_path)
        frame_count: Total number of frames in the full video
        output_fgr_path: Path to save the foreground video
        output_pha_path: Path to save the alpha video
        width: Frame width
        height: Frame height
        fps: Frames per second
        video_codec: Video codec to use
        video_quality: Quality preset
        custom_bitrate: Custom bitrate in kbps
        
    Returns:
        Tuple of (output_fgr_path, output_pha_path)
    """
    try:
        # Create high-quality output video writers
        fgr_writer = create_high_quality_writer(output_fgr_path, fps, width, height,
                                               video_codec, video_quality, custom_bitrate)
        pha_writer = create_high_quality_writer(output_pha_path, fps, width, height,
                                               video_codec, video_quality, custom_bitrate)
        
        if not fgr_writer.isOpened() or not pha_writer.isOpened():
            raise ValueError(f"Could not create output videos")
        
        # Sort ranges by start frame
        range_results.sort(key=lambda x: x['start_frame'])
        
        # Initialize frame index for tracking
        current_frame = 0
        
        # Process each range and fill gaps
        for range_info in range_results:
            start_frame = range_info['start_frame']
            end_frame = range_info['end_frame']
            fgr_path = range_info['fgr_path']
            pha_path = range_info['pha_path']
            
            # If there's a gap, fill with empty frames
            gap_size = start_frame - current_frame
            if gap_size > 0:
                print(f"Adding {gap_size} empty frames from {current_frame} to {start_frame-1}")
                # Create empty frames
                empty_fgr = np.zeros((height, width, 3), dtype=np.uint8)
                empty_pha = np.zeros((height, width, 3), dtype=np.uint8)
                
                for _ in range(gap_size):
                    fgr_writer.write(empty_fgr)
                    pha_writer.write(empty_pha)
            
            # Handle both video files and directories
            if os.path.isfile(fgr_path) and fgr_path.endswith('.mp4'):
                # Handle video files
                fgr_cap = cv2.VideoCapture(fgr_path)
                pha_cap = cv2.VideoCapture(pha_path)
                
                if not fgr_cap.isOpened() or not pha_cap.isOpened():
                    print(f"Warning: Could not open range videos {fgr_path} or {pha_path}")
                    # Fill with empty frames
                    empty_fgr = np.zeros((height, width, 3), dtype=np.uint8)
                    empty_pha = np.zeros((height, width, 3), dtype=np.uint8)
                    
                    for _ in range(end_frame - start_frame + 1):
                        fgr_writer.write(empty_fgr)
                        pha_writer.write(empty_pha)
                else:
                    # Add frames from video files
                    range_frame_count = int(fgr_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    print(f"Adding {range_frame_count} frames from range {start_frame} to {end_frame}")
                    
                    for _ in range(range_frame_count):
                        ret_fgr, fgr_frame = fgr_cap.read()
                        ret_pha, pha_frame = pha_cap.read()
                        
                        if not ret_fgr or not ret_pha:
                            break
                        
                        # Resize if needed
                        if fgr_frame.shape[0] != height or fgr_frame.shape[1] != width:
                            fgr_frame = cv2.resize(fgr_frame, (width, height), cv2.INTER_LINEAR)
                        
                        if pha_frame.shape[0] != height or pha_frame.shape[1] != width:
                            pha_frame = cv2.resize(pha_frame, (width, height), cv2.INTER_LINEAR)
                        
                        # Write frames
                        fgr_writer.write(fgr_frame)
                        pha_writer.write(pha_frame)
                    
                    # Release video captures
                    fgr_cap.release()
                    pha_cap.release()
                    
            elif os.path.isdir(fgr_path):
                # Handle directories of image files
                print(f"Processing image directories: {fgr_path} and {pha_path}")
                
                # Get image file lists
                fgr_images = sorted(glob.glob(os.path.join(fgr_path, "*.png")))
                pha_images = sorted(glob.glob(os.path.join(pha_path, "*.png")))
                
                if not fgr_images or not pha_images:
                    print(f"Warning: No images found in directories")
                    # Fill with empty frames
                    empty_fgr = np.zeros((height, width, 3), dtype=np.uint8)
                    empty_pha = np.zeros((height, width, 3), dtype=np.uint8)
                    
                    for _ in range(end_frame - start_frame + 1):
                        fgr_writer.write(empty_fgr)
                        pha_writer.write(empty_pha)
                else:
                    # Process image files
                    range_frame_count = min(len(fgr_images), len(pha_images))
                    print(f"Adding {range_frame_count} frames from image directories")
                    
                    for i in range(range_frame_count):
                        fgr_frame = cv2.imread(fgr_images[i])
                        pha_frame = cv2.imread(pha_images[i])
                        
                        if fgr_frame is None or pha_frame is None:
                            print(f"Warning: Could not read image files at index {i}")
                            continue
                        
                        # Resize if needed
                        if fgr_frame.shape[0] != height or fgr_frame.shape[1] != width:
                            fgr_frame = cv2.resize(fgr_frame, (width, height), cv2.INTER_LINEAR)
                        
                        if pha_frame.shape[0] != height or pha_frame.shape[1] != width:
                            pha_frame = cv2.resize(pha_frame, (width, height), cv2.INTER_LINEAR)
                        
                        # Write frames
                        fgr_writer.write(fgr_frame)
                        pha_writer.write(pha_frame)
            else:
                print(f"Warning: Unknown format for range output: {fgr_path}")
                # Fill with empty frames
                empty_fgr = np.zeros((height, width, 3), dtype=np.uint8)
                empty_pha = np.zeros((height, width, 3), dtype=np.uint8)
                
                for _ in range(end_frame - start_frame + 1):
                    fgr_writer.write(empty_fgr)
                    pha_writer.write(empty_pha)
            
            # Update current frame index
            current_frame = end_frame + 1
        
        # Fill any remaining frames with empty frames
        remaining_frames = frame_count - current_frame
        if remaining_frames > 0:
            print(f"Adding {remaining_frames} empty frames from {current_frame} to {frame_count-1}")
            # Create empty frames
            empty_fgr = np.zeros((height, width, 3), dtype=np.uint8)
            empty_pha = np.zeros((height, width, 3), dtype=np.uint8)
            
            for _ in range(remaining_frames):
                fgr_writer.write(empty_fgr)
                pha_writer.write(empty_pha)
        
        # Release output videos
        fgr_writer.release()
        pha_writer.release()
        
        print(f"Created full-length videos from ranges")
        return output_fgr_path, output_pha_path
        
    except Exception as e:
        print(f"Error creating full videos from ranges: {str(e)}")
        traceback.print_exc()
        return None, None