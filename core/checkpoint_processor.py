"""
Checkpoint-based video processing for MatAnyone.
Functions for processing videos with multiple keyframe checkpoints.
"""

import os
import cv2
import numpy as np
import time
import traceback

# Import video utilities
from utils.video_utils import reverse_video, blend_videos, concatenate_videos, cleanup_temporary_files


def process_with_checkpoints(processor, input_path, mask_paths, checkpoint_frames, output_path, 
                            bidirectional=True, blend_method='weighted', **kwargs):
    """
    Process a video with checkpoints for more precise matting
    
    Args:
        processor: The InterruptibleInferenceCore processor instance
        input_path: Path to the input video
        mask_paths: Dictionary mapping frame indices to mask file paths
        checkpoint_frames: Ordered list of checkpoint frame indices
        output_path: Directory to save outputs
        bidirectional: Whether to process segments bidirectionally
        blend_method: Method to blend bidirectional results
        **kwargs: Additional arguments to pass to process_video
    
    Returns:
        Tuple of paths to the final foreground and alpha videos
    """
    if len(checkpoint_frames) < 2:
        print("Need at least 2 checkpoints to use checkpoint mode. Falling back to standard processing.")
        # Use the first mask if available, otherwise use the provided mask_path
        if checkpoint_frames:
            mask_path = mask_paths[checkpoint_frames[0]]
            return processor.process_video(input_path, mask_path, output_path, **kwargs)
        else:
            return None
            
    # Get video info
    print(f"Processing video with {len(checkpoint_frames)} checkpoints")
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Create a base filename from input path
    video_name = os.path.basename(input_path)
    if video_name.endswith(('.mp4', '.mov', '.avi')):
        video_name = os.path.splitext(video_name)[0]
    
    # Create temporary directory for segments
    temp_dir = os.path.join(output_path, f"{video_name}_segments_temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create segment directories
    segment_dirs = []
    temp_files = [temp_dir]  # Keep track of temp files for cleanup
    
    # Define segments between checkpoints
    segments = []
    for i in range(len(checkpoint_frames) - 1):
        start_frame = checkpoint_frames[i]
        end_frame = checkpoint_frames[i + 1]
        
        start_mask = mask_paths[start_frame]
        end_mask = mask_paths[end_frame]
        
        # Create segment directory
        segment_dir = os.path.join(temp_dir, f"segment_{i}_{start_frame}_{end_frame}")
        os.makedirs(segment_dir, exist_ok=True)
        segment_dirs.append(segment_dir)
        temp_files.append(segment_dir)
        
        segments.append({
            'index': i,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'start_mask': start_mask,
            'end_mask': end_mask,
            'output_dir': segment_dir
        })
    
    # Extract and process each segment
    processed_segments = []
    
    for segment in segments:
        try:
            print(f"Processing segment {segment['index'] + 1}/{len(segments)}: "
                f"Frames {segment['start_frame']} to {segment['end_frame']}")
            
            segment_video = os.path.join(segment['output_dir'], f"segment_{segment['start_frame']}_{segment['end_frame']}.mp4")
            
            # Extract segment
            extracted_segment = extract_segment(input_path, segment['start_frame'], segment['end_frame'], segment_video)
            if not extracted_segment:
                continue
            
            # Process bidirectionally if requested
            if bidirectional:
                # Forward pass (using start frame mask)
                print(f"Forward pass for segment {segment['index'] + 1}...")
                forward_kwargs = kwargs.copy()
                forward_kwargs['suffix'] = 'forward'
                
                forward_fgr, forward_pha = processor.process_video(
                    input_path=segment_video,
                    mask_path=segment['start_mask'],
                    output_path=segment['output_dir'],
                    **forward_kwargs
                )
                
                # Clear memory
                processor.clear_memory()
                
                # Backward pass (using end frame mask)
                print(f"Backward pass for segment {segment['index'] + 1}...")
                # Create a reversed video
                reversed_video = os.path.join(segment['output_dir'], f"reversed_segment.mp4")
                reversed_video = reverse_video(segment_video, reversed_video)
                
                # Process with the end frame mask
                reverse_kwargs = kwargs.copy()
                reverse_kwargs['suffix'] = 'reverse'
                
                reverse_fgr, reverse_pha = processor.process_video(
                    input_path=reversed_video,
                    mask_path=segment['end_mask'],
                    output_path=segment['output_dir'],
                    **reverse_kwargs
                )
                
                # Re-reverse the results
                re_reversed_fgr = os.path.join(segment['output_dir'], f"re_reversed_fgr.mp4")
                re_reversed_pha = os.path.join(segment['output_dir'], f"re_reversed_pha.mp4")
                
                re_reversed_fgr = reverse_video(reverse_fgr, re_reversed_fgr)
                re_reversed_pha = reverse_video(reverse_pha, re_reversed_pha)
                
                # Blend forward and backward passes
                print(f"Blending passes for segment {segment['index'] + 1}...")
                blended_fgr = os.path.join(segment['output_dir'], f"blended_fgr.mp4")
                blended_pha = os.path.join(segment['output_dir'], f"blended_pha.mp4")
                
                blend_videos(forward_fgr, re_reversed_fgr, blended_fgr, blend_method)
                blend_videos(forward_pha, re_reversed_pha, blended_pha, blend_method)
                
                # Add to processed segments
                processed_segments.append({
                    'segment': segment,
                    'fgr': blended_fgr,
                    'pha': blended_pha
                })
                
                # Add temporary files for cleanup
                temp_files.extend([
                    segment_video, reversed_video, 
                    forward_fgr, forward_pha,
                    reverse_fgr, reverse_pha,
                    re_reversed_fgr, re_reversed_pha
                ])
            else:
                # Process normally using start frame mask
                print(f"Processing segment {segment['index'] + 1} with start frame mask...")
                fgr, pha = processor.process_video(
                    input_path=segment_video,
                    mask_path=segment['start_mask'],
                    output_path=segment['output_dir'],
                    **kwargs
                )
                
                # Add to processed segments
                processed_segments.append({
                    'segment': segment,
                    'fgr': fgr,
                    'pha': pha
                })
                
                # Add temporary files for cleanup
                temp_files.append(segment_video)
        
        except Exception as e:
            print(f"Error processing segment {segment['index'] + 1}: {str(e)}")
            traceback.print_exc()
    
    # If no segments were processed successfully, return None
    if not processed_segments:
        print("No segments were processed successfully.")
        return None
    
    # Combine the processed segments into final output
    print("Combining processed segments...")
    
    final_fgr_path = os.path.join(output_path, f"{video_name}_checkpoints_fgr.mp4")
    final_pha_path = os.path.join(output_path, f"{video_name}_checkpoints_pha.mp4")
    
    # Check if segments need to be concatenated
    if len(processed_segments) > 1:
        # Create lists of segment videos for concatenation
        fgr_segments = [seg['fgr'] for seg in processed_segments]
        pha_segments = [seg['pha'] for seg in processed_segments]
        
        # Concatenate segments
        concatenate_videos(fgr_segments, final_fgr_path)
        concatenate_videos(pha_segments, final_pha_path)
    else:
        # Just copy the single segment to the final output
        import shutil
        shutil.copy(processed_segments[0]['fgr'], final_fgr_path)
        shutil.copy(processed_segments[0]['pha'], final_pha_path)
    
    # Clean up temporary files
    print("Cleaning up temporary files...")
    cleanup_temporary_files(temp_files, True)
    
    print("Checkpoint processing complete!")
    return final_fgr_path, final_pha_path


def extract_segment(input_path, start_frame, end_frame, output_path):
    """
    Extract a segment from the video
    
    Args:
        input_path: Path to the input video
        start_frame: Starting frame index
        end_frame: Ending frame index
        output_path: Path to save the segment
        
    Returns:
        Path to the extracted segment, or None if extraction failed
    """
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get original video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set up output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read frames and write to output
        frame_count = 0
        frames_to_extract = end_frame - start_frame
        
        while frame_count < frames_to_extract:
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Extracted {frame_count}/{frames_to_extract} frames")
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"Extracted segment from frame {start_frame} to {end_frame}")
        return output_path
    
    except Exception as e:
        print(f"Error extracting segment: {str(e)}")
        traceback.print_exc()
        return None
