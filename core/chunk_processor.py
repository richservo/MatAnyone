# chunk_processor.py - v1.1737774900
# Updated: Friday, January 24, 2025 at 17:55:00 PST
# Changes in this version:
# - Updated all video creation to use high-quality video writers
# - Added video codec, quality, and bitrate parameters to process_video_in_chunks
# - Pass video quality settings through to all video creation functions
# - Enhanced error handling for video quality-related issues

"""
Chunk-based video processing for MatAnyone.
Functions for splitting videos into chunks and reassembling them.
"""

import os
import cv2
import numpy as np
import time
import traceback
import torch
from matanyone.utils.inference_utils import gen_dilate

# Import video utilities
from utils.video_utils import (
    blend_videos, cleanup_temporary_files, reverse_video,
    create_high_quality_writer
)


def process_video_in_chunks(processor, input_path, mask_path, output_path, num_chunks=1, 
                           bidirectional=False, blend_method='weighted', reverse_dilate=15, 
                           cleanup_temp=True, mask_skip_threshold=5, allow_native_resolution=True,
                           video_codec='Auto', video_quality='High', custom_bitrate=None, 
                           progress_callback=None, **kwargs):
    """
    Process a video by breaking it into chunks with dimensions that are compatible with the model,
    processing each chunk separately, then reassembling with smooth blending.
    
    Now supports bidirectional processing per chunk and mask-based chunk skipping.
    
    Args:
        processor: The InterruptibleInferenceCore processor instance
        input_path: Path to the input video or directory of frames
        mask_path: Path to the mask image
        output_path: Base path for the output files
        num_chunks: Number of horizontal chunks to split each frame into (1 = no splitting)
        bidirectional: Whether to use bidirectional processing for each chunk
        blend_method: Method to blend forward and reverse passes ('weighted', 'max_alpha', 'min_alpha', 'average')
        reverse_dilate: Dilation radius for the mask in reverse pass
        cleanup_temp: Whether to clean up temporary files
        mask_skip_threshold: Percentage threshold of non-zero pixels to consider a mask chunk worth processing
        allow_native_resolution: Whether to allow processing at native resolution
        video_codec: Video codec to use ('Auto', 'H.264', 'H.265', 'VP9')
        video_quality: Quality preset ('Low', 'Medium', 'High', 'Very High', 'Lossless')
        custom_bitrate: Custom bitrate in kbps (overrides quality preset)
        progress_callback: Optional callback function for progress updates (percentage, stage, status)
        **kwargs: Additional arguments to pass to process_video
    
    Returns:
        Tuple of paths to the foreground and alpha videos
    """
    # Helper function for reporting progress
    def report_progress(percentage, stage, status):
        if progress_callback and callable(progress_callback):
            try:
                progress_callback(percentage, stage, status)
            except Exception as e:
                print(f"Error in progress callback: {e}")
                pass
        # Always print progress to console for monitoring
        print(f"Progress: {percentage:.1f}% - {stage} - {status}")
    
    if num_chunks <= 1:
        # No chunking needed, just process as normal
        report_progress(0, "Forwarding", "Processing without chunking")
        return processor.process_video(input_path=input_path, mask_path=mask_path, 
                                     output_path=output_path, 
                                     bidirectional=bidirectional, 
                                     blend_method=blend_method, 
                                     reverse_dilate=reverse_dilate,
                                     cleanup_temp=cleanup_temp,
                                     video_codec=video_codec,
                                     video_quality=video_quality,
                                     custom_bitrate=custom_bitrate,
                                     progress_callback=progress_callback,
                                     **kwargs)
    
    # Make sure num_chunks is reasonable
    num_chunks = max(2, min(8, num_chunks))  # Limit to 2-8 chunks for sanity
    print(f"Processing video in {num_chunks} horizontal chunks")
    
    # Report progress starting chunked processing
    report_progress(2, "Setup", f"Starting processing with {num_chunks} chunks")
    
    if bidirectional:
        print(f"Using bidirectional processing for each chunk with {blend_method} blending")
    
    # Print video quality settings
    if custom_bitrate:
        print(f"Using video codec: {video_codec}, quality: {video_quality}, custom bitrate: {custom_bitrate} kbps")
    else:
        print(f"Using video codec: {video_codec}, quality: {video_quality}")
    
    # Determine if we're dealing with a video or image sequence
    is_video = os.path.isfile(input_path) and input_path.endswith(('.mp4', '.mov', '.avi'))
    
    # Create temporary directories for chunks and outputs
    temp_dir = os.path.join(output_path, "chunks_temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Keep track of temporary files and directories for cleanup
    temp_files = [temp_dir]  # Add temp_dir to the list for cleanup
    
    # Default video parameters in case we need to fall back
    video_name = os.path.basename(input_path)
    if video_name.endswith(('.mp4', '.mov', '.avi')):
        video_name = os.path.splitext(video_name)[0]
    
    # Process video in chunks
    if is_video:
        try:
            # Get video information first
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {input_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            
            # Critical: Ensure dimensions are model-friendly
            # MatAnyone models require dimensions divisible by 8
            MODEL_FACTOR = 8
            
            # Adjust height to ensure it's divisible by MODEL_FACTOR
            adjusted_height = (height // MODEL_FACTOR) * MODEL_FACTOR
            if adjusted_height != height:
                print(f"Adjusting height from {height} to {adjusted_height} to ensure divisibility by {MODEL_FACTOR}")
                height = adjusted_height
            
            # Determine max_size value based on user preference
            max_size_value = kwargs.get('max_size', -1)
            if max_size_value == -1:  # Native resolution
                # Check if allow_native_resolution flag is set
                if allow_native_resolution:
                    print("Using native resolution as requested. This may require more GPU memory.")
                else:
                    print("Native resolution requested but allow_native_resolution is False.")
                    print("Using 1024 as reasonable default to prevent memory issues.")
                    max_size_value = 1024  # Use a reasonable default
            else:
                print(f"Using specified max_size value of {max_size_value}")
            
            # Ensure max_size is divisible by MODEL_FACTOR
            if max_size_value != -1:
                max_size_value = (max_size_value // MODEL_FACTOR) * MODEL_FACTOR
                print(f"Adjusted max_size value to {max_size_value} to ensure divisibility by {MODEL_FACTOR}")
            
            # Calculate base width for each chunk, ensuring divisibility by MODEL_FACTOR
            base_chunk_width = width // num_chunks
            base_chunk_width = (base_chunk_width // MODEL_FACTOR) * MODEL_FACTOR
            base_chunk_width = max(MODEL_FACTOR * 2, base_chunk_width)  # Ensure minimum width
            
            # Define overlap for blending with special case for 2-chunk processing
            if num_chunks == 2:
                # Special case for 2 chunks - use larger overlap (33% of base width)
                overlap_pixels = base_chunk_width // 3
                overlap_pixels = (overlap_pixels // MODEL_FACTOR) * MODEL_FACTOR
                overlap_pixels = max(MODEL_FACTOR * 8, min(overlap_pixels, base_chunk_width // 2))
                print(f"Using enhanced overlap of {overlap_pixels} pixels for 2-chunk processing")
            else:
                # Standard overlap for 3+ chunks (20% of base width)
                overlap_pixels = base_chunk_width // 5
                overlap_pixels = (overlap_pixels // MODEL_FACTOR) * MODEL_FACTOR
                overlap_pixels = max(MODEL_FACTOR * 2, min(overlap_pixels, base_chunk_width // 3))
                print(f"Using {overlap_pixels} pixels overlap between chunks")
            
            # Define chunk segments with safer calculations
            chunk_segments = []
            for i in range(num_chunks):
                # Calculate start position
                if i == 0:
                    start_x = 0
                else:
                    start_x = i * base_chunk_width - overlap_pixels
                    start_x = max(0, (start_x // MODEL_FACTOR) * MODEL_FACTOR)
                
                # Calculate end position
                if i == num_chunks - 1:
                    end_x = width - (width % MODEL_FACTOR)  # Ensure divisibility
                else:
                    end_x = (i + 1) * base_chunk_width + overlap_pixels
                    end_x = min((end_x // MODEL_FACTOR) * MODEL_FACTOR, width - (width % MODEL_FACTOR))
                
                # Validate chunk dimensions
                if end_x <= start_x or end_x > width or start_x < 0:
                    print(f"Warning: Invalid chunk dimensions: start={start_x}, end={end_x}. Skipping this chunk.")
                    continue
                
                # Ensure width is divisible by MODEL_FACTOR
                chunk_width = end_x - start_x
                
                # Only add the chunk if width >= MODEL_FACTOR * 2
                if chunk_width >= MODEL_FACTOR * 2:
                    chunk_segments.append((start_x, end_x))
                    print(f"Chunk {len(chunk_segments)}: {start_x} to {end_x}, Width: {end_x - start_x}")
            
            # Update num_chunks based on valid segments
            num_chunks = len(chunk_segments)
            print(f"Final chunk count: {num_chunks}")
            
            if num_chunks == 0:
                print("No valid chunks could be created. Falling back to processing without chunking.")
                return processor.process_video(input_path=input_path, mask_path=mask_path, 
                                             output_path=output_path, **kwargs)
            
            # Load and prepare mask
            print(f"Loading mask from {mask_path}")
            try:
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask is None:
                    raise ValueError(f"Failed to load mask: {mask_path}")
            except Exception as e:
                print(f"Error loading mask: {str(e)}")
                raise
            
            # Resize mask to match video dimensions if needed
            if mask.shape[0] != height or mask.shape[1] != width:
                print(f"Resizing mask from {mask.shape} to {width}x{height}")
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Prepare mask chunks and check which ones have content
            mask_chunks = []
            mask_has_content_flags = []
            
            for i, (start_x, end_x) in enumerate(chunk_segments):
                # Extract chunk with boundary checks
                safe_end_x = min(end_x, mask.shape[1])
                if start_x >= safe_end_x:
                    print(f"Warning: Invalid mask chunk dimensions: start={start_x}, end={safe_end_x}")
                    mask_chunk = np.zeros((height, MODEL_FACTOR * 2), dtype=mask.dtype)
                else:
                    mask_chunk = mask[:height, start_x:safe_end_x].copy()
                
                # Ensure dimensions are as expected
                expected_width = end_x - start_x
                if mask_chunk.shape[1] != expected_width:
                    print(f"Resizing mask chunk {i+1} from {mask_chunk.shape[1]} to {expected_width}")
                    try:
                        mask_chunk = cv2.resize(mask_chunk, (expected_width, height), 
                                              interpolation=cv2.INTER_NEAREST)
                    except Exception as e:
                        print(f"Error resizing mask chunk: {str(e)}")
                        # Create a blank mask of the right size as fallback
                        mask_chunk = np.zeros((height, expected_width), dtype=mask.dtype)
                
                # Save mask chunk
                mask_chunk_path = os.path.join(temp_dir, f"mask_chunk_{i}.png")
                cv2.imwrite(mask_chunk_path, mask_chunk)
                mask_chunks.append(mask_chunk_path)
                
                # Check if this mask chunk has any content
                non_zero_count = np.count_nonzero(mask_chunk)
                total_pixels = mask_chunk.size
                non_zero_percentage = (non_zero_count / total_pixels) * 100
                
                has_content = non_zero_percentage >= mask_skip_threshold
                mask_has_content_flags.append(has_content)
                
                if has_content:
                    print(f"Chunk {i+1}/{num_chunks} mask has {non_zero_percentage:.2f}% content - will process")
                else:
                    print(f"Chunk {i+1}/{num_chunks} mask has only {non_zero_percentage:.2f}% content - will skip")
                
                print(f"Created mask chunk {i+1}/{num_chunks} - {mask_chunk.shape}")
            
            # Split video into chunks - report progress
            report_progress(5, "Splitting", f"Splitting video into {num_chunks} chunks")
            print(f"Splitting video {input_path} into {num_chunks} chunks")
            chunk_paths = []
            
            # Create a VideoCapture for the input video
            cap = cv2.VideoCapture(input_path)
            
            # Create video writers for chunks
            chunk_writers = []
            for i, (start_x, end_x) in enumerate(chunk_segments):
                chunk_path = os.path.join(temp_dir, f"chunk_{i}.mp4")
                chunk_paths.append(chunk_path)
                
                chunk_shape = (end_x - start_x, height)
                
                # Use high-quality video writer
                writer = create_high_quality_writer(
                    chunk_path, fps, chunk_shape[0], chunk_shape[1],
                    video_codec, video_quality, custom_bitrate
                )
                chunk_writers.append(writer)
            
            # Report progress before frame splitting
            report_progress(8, "Splitting", "Beginning frame segmentation into chunks")
            
            # Process frames and write chunks
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Report progress periodically during splitting
                if frame_idx % 100 == 0:
                    # Calculate progress (8-20% range for splitting)
                    split_progress = 8 + ((frame_idx / frame_count) * 12)
                    report_progress(split_progress, "Splitting", f"Split {frame_idx}/{frame_count} frames into chunks")
                
                # If the frame height needs adjustment, resize
                if frame.shape[0] != height:
                    frame = cv2.resize(frame, (frame.shape[1], height), interpolation=cv2.INTER_AREA)
                
                for i, (start_x, end_x) in enumerate(chunk_segments):
                    # Get the chunk with boundary checks
                    safe_end_x = min(end_x, frame.shape[1])
                    if start_x >= safe_end_x:
                        print(f"Warning: Invalid frame chunk {i}: start={start_x}, end={safe_end_x}")
                        # Create a blank frame
                        chunk = np.zeros((height, end_x - start_x, 3), dtype=np.uint8)
                    else:
                        chunk = frame[:, start_x:safe_end_x].copy()
                    
                    # Ensure dimensions are as expected
                    expected_width = end_x - start_x
                    if chunk.shape[1] != expected_width:
                        print(f"Padding frame chunk {i} from width {chunk.shape[1]} to {expected_width}")
                        # Create a correctly sized frame
                        padded_chunk = np.zeros((height, expected_width, 3), dtype=np.uint8)
                        # Copy the available data
                        padded_chunk[:, :chunk.shape[1]] = chunk
                        chunk = padded_chunk
                    
                    # Write the chunk
                    chunk_writers[i].write(chunk)
                
                frame_idx += 1
                if frame_idx % 100 == 0:
                    print(f"Split {frame_idx}/{frame_count} frames into chunks")
                
                # Check for interrupts
                processor.check_interrupt()
            
            # Release resources
            cap.release()
            for writer in chunk_writers:
                writer.release()
            
            # Process each chunk with corresponding mask
            chunk_outputs = []
            
            # Report progress - starting chunk processing
            report_progress(20, "Processing", f"Starting to process {num_chunks} video chunks")
            
            # Process each chunk with its own processor instance
            for i, (start_x, end_x) in enumerate(chunk_segments):
                # Calculate progress at start of each chunk (20-80% range for chunk processing)
                chunk_progress = 20 + (i / num_chunks * 60)
                report_progress(chunk_progress, "Processing", f"Processing chunk {i+1}/{num_chunks}")
                print(f"Processing chunk {i+1}/{num_chunks}")
                
                # Setup output for this chunk
                chunk_output_dir = os.path.join(temp_dir, f"output_chunk_{i}")
                os.makedirs(chunk_output_dir, exist_ok=True)
                
                # Add to temp files for cleanup
                temp_files.append(chunk_output_dir)
                
                # Check if this chunk has any mask content worth processing
                if not mask_has_content_flags[i]:
                    print(f"Chunk {i+1}/{num_chunks} has no significant mask content - skipping processing")
                    
                    # Create empty (black) output videos for this chunk
                    chunk_width = end_x - start_x
                    chunk_fgr = os.path.join(chunk_output_dir, f"empty_fgr.mp4")
                    chunk_pha = os.path.join(chunk_output_dir, f"empty_pha.mp4")
                    
                    # Create empty foreground and alpha videos with high quality
                    processor.create_empty_video(
                        chunk_fgr, chunk_width, height, fps, frame_count, alpha=False,
                        video_codec=video_codec, video_quality=video_quality, custom_bitrate=custom_bitrate
                    )
                    processor.create_empty_video(
                        chunk_pha, chunk_width, height, fps, frame_count, alpha=True,
                        video_codec=video_codec, video_quality=video_quality, custom_bitrate=custom_bitrate
                    )
                    
                    # Add to chunk outputs
                    chunk_outputs.append({
                        'fgr_path': chunk_fgr,
                        'pha_path': chunk_pha,
                        'start_x': start_x,
                        'end_x': end_x,
                        'width': end_x - start_x
                    })
                    
                    print(f"Created empty output for chunk {i+1}/{num_chunks}")
                    continue
                
                # Process this chunk with the same settings
                try:
                    # Create a new processor instance for each chunk to avoid shared state issues
                    from core.inference_core import InterruptibleInferenceCore
                    chunk_processor = InterruptibleInferenceCore(processor.model_path) if i > 0 else processor
                    
                    # Use a copy of kwargs to avoid modifying the original
                    chunk_kwargs = kwargs.copy()
                    
                    # Set max_size parameter
                    chunk_kwargs['max_size'] = max_size_value
                    # Add video quality parameters
                    chunk_kwargs['video_codec'] = video_codec
                    chunk_kwargs['video_quality'] = video_quality
                    chunk_kwargs['custom_bitrate'] = custom_bitrate
                    print(f"Using max_size of {max_size_value} for chunk {i+1}")
                    
                    # If bidirectional is enabled, process in both directions
                    chunk_fgr = chunk_pha = None
                    
                    if bidirectional:
                        print(f"Bidirectional processing for chunk {i+1}/{num_chunks}")
                        
                        # Forward pass
                        print(f"Forward pass for chunk {i+1}")
                        chunk_kwargs['suffix'] = f'forward'
                        forward_fgr, forward_pha = chunk_processor.process_video(
                            input_path=chunk_paths[i],
                            mask_path=mask_chunks[i],
                            output_path=chunk_output_dir,
                            **chunk_kwargs
                        )
                        
                        print(f"Forward pass complete for chunk {i+1}")
                        
                        # Get the last frame from the forward pass to use as a mask for the reverse pass
                        last_frame_path = None
                        
                        if os.path.isdir(forward_pha):
                            # If we have individual frames, get the last one
                            frame_files = [f for f in os.listdir(forward_pha) if f.endswith('.png')]
                            if frame_files:
                                frame_files.sort(key=lambda x: int(os.path.splitext(x)[0]) if x.split('.')[0].isdigit() else 0)
                                last_frame_path = os.path.join(forward_pha, frame_files[-1])
                        elif os.path.isfile(forward_pha) and forward_pha.endswith('.mp4'):
                            # If we have a video, extract the last frame
                            cap = cv2.VideoCapture(forward_pha)
                            if cap.isOpened():
                                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                if frame_count > 0:
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                                    ret, last_frame = cap.read()
                                    if ret:
                                        # Create a temporary file to save the last frame
                                        last_frame_path = os.path.join(chunk_output_dir, f"chunk_{i}_lastframe.png")
                                        cv2.imwrite(last_frame_path, last_frame)
                                cap.release()
                        
                        if last_frame_path is not None:
                            print(f"Using last frame as mask for reverse pass: {last_frame_path}")
                            
                            # Dilate the mask before using it for reverse pass
                            if reverse_dilate > 0:
                                mask = cv2.imread(last_frame_path, cv2.IMREAD_GRAYSCALE)
                                mask = gen_dilate(mask, reverse_dilate, reverse_dilate)
                                # Save the dilated mask
                                dilated_mask_path = os.path.join(chunk_output_dir, f"chunk_{i}_dilated_mask.png")
                                cv2.imwrite(dilated_mask_path, mask)
                                last_frame_path = dilated_mask_path
                                print(f"Dilated mask saved to {dilated_mask_path}")
                            
                            # Clear memory for the reverse pass
                            chunk_processor.clear_memory()
                            
                            # Create a reversed version of the chunk video
                            print(f"Creating reversed chunk for chunk {i+1}")
                            reversed_chunk_path = os.path.join(chunk_output_dir, f"chunk_{i}_reversed.mp4")
                            reversed_chunk_path = reverse_video(
                                chunk_paths[i], reversed_chunk_path,
                                video_codec, video_quality, custom_bitrate
                            )
                            
                            if reversed_chunk_path:
                                # Process the reversed chunk
                                print(f"Reverse pass for chunk {i+1}")
                                reverse_kwargs = chunk_kwargs.copy()
                                reverse_kwargs['suffix'] = 'reverse'
                                reverse_kwargs['mask_path'] = last_frame_path
                                
                                reverse_fgr, reverse_pha = chunk_processor.process_video(
                                    input_path=reversed_chunk_path,
                                    output_path=chunk_output_dir,
                                    **reverse_kwargs
                                )
                                
                                print(f"Reverse pass complete for chunk {i+1}")
                                
                                # Re-reverse the output videos
                                re_reversed_fgr = os.path.join(chunk_output_dir, f"chunk_{i}_re_reversed_fgr.mp4")
                                re_reversed_pha = os.path.join(chunk_output_dir, f"chunk_{i}_re_reversed_pha.mp4")
                                
                                re_reversed_fgr = reverse_video(
                                    reverse_fgr, re_reversed_fgr,
                                    video_codec, video_quality, custom_bitrate
                                )
                                re_reversed_pha = reverse_video(
                                    reverse_pha, re_reversed_pha,
                                    video_codec, video_quality, custom_bitrate
                                )
                                
                                # Blend forward and reverse passes
                                print(f"Blending passes for chunk {i+1} using {blend_method} method")
                                blended_fgr = os.path.join(chunk_output_dir, f"chunk_{i}_bidirectional_fgr.mp4")
                                blended_pha = os.path.join(chunk_output_dir, f"chunk_{i}_bidirectional_pha.mp4")
                                
                                blend_videos(
                                    forward_fgr, re_reversed_fgr, blended_fgr, blend_method,
                                    video_codec, video_quality, custom_bitrate
                                )
                                blend_videos(
                                    forward_pha, re_reversed_pha, blended_pha, blend_method,
                                    video_codec, video_quality, custom_bitrate
                                )
                                
                                # Set the blended videos as our result
                                chunk_fgr, chunk_pha = blended_fgr, blended_pha
                                
                                # Add to temp files for cleanup
                                if cleanup_temp:
                                    temp_files.extend([
                                        forward_fgr, forward_pha,
                                        reverse_fgr, reverse_pha,
                                        reversed_chunk_path, last_frame_path,
                                        re_reversed_fgr, re_reversed_pha
                                    ])
                            else:
                                print(f"Could not create reversed video for chunk {i+1}. Using forward pass only.")
                                chunk_fgr, chunk_pha = forward_fgr, forward_pha
                        else:
                            print(f"Could not extract last frame for chunk {i+1}. Using forward pass only.")
                            chunk_fgr, chunk_pha = forward_fgr, forward_pha
                    else:
                        # Process in forward direction only
                        chunk_fgr, chunk_pha = chunk_processor.process_video(
                            input_path=chunk_paths[i],
                            mask_path=mask_chunks[i],
                            output_path=chunk_output_dir,
                            **chunk_kwargs
                        )
                    
                    # Validate outputs
                    if not os.path.exists(chunk_fgr) or not os.path.exists(chunk_pha):
                        print(f"Warning: Output files for chunk {i+1} not found. Skipping this chunk.")
                        continue
                    
                    chunk_outputs.append({
                        'fgr_path': chunk_fgr,
                        'pha_path': chunk_pha,
                        'start_x': start_x,
                        'end_x': end_x,
                        'width': end_x - start_x
                    })
                    print(f"Completed chunk {i+1}/{num_chunks}")
                    
                    # Clear memory for next chunk
                    try:
                        # Force CUDA cache clearing
                        torch.cuda.empty_cache()
                        
                        # Try to clear MPS cache (Apple Silicon)
                        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                            # There's no direct MPS cache clearing, but we can help garbage collection
                            import gc
                            gc.collect()
                    except:
                        pass
                    
                except Exception as e:
                    print(f"Error processing chunk {i+1}: {str(e)}")
                    traceback.print_exc()
                    # Continue with other chunks
                    continue
            
            # Check if any chunks were processed successfully
            if not chunk_outputs:
                print("No chunks were processed successfully. Falling back to processing without chunking.")
                report_progress(20, "Fallback", "No chunks processed successfully, falling back to regular processing")
                # Clean up temp files
                cleanup_temporary_files(temp_files, True)
                # Try processing without chunking as fallback
                return processor.process_video(input_path=input_path, mask_path=mask_path, 
                                            output_path=output_path, 
                                            progress_callback=progress_callback,
                                            **kwargs)
            
            # Reassemble chunks into final output
            report_progress(80, "Reassembling", f"Reassembling {len(chunk_outputs)} chunks with smooth blending")
            print(f"Reassembling {len(chunk_outputs)} chunks with smooth blending")
            
            # Check if outputs are video files or directories
            first_output = chunk_outputs[0]
            is_output_video = os.path.isfile(first_output['fgr_path']) and first_output['fgr_path'].endswith('.mp4')
            
            if is_output_video:
                # Handle video reassembly
                # Create appropriate output filename with bidirectional indication if needed
                if bidirectional:
                    final_fgr_path = os.path.join(output_path, f"{video_name}_bidirectional_fgr.mp4")
                    final_pha_path = os.path.join(output_path, f"{video_name}_bidirectional_pha.mp4")
                else:
                    final_fgr_path = os.path.join(output_path, f"{video_name}_fgr.mp4")
                    final_pha_path = os.path.join(output_path, f"{video_name}_pha.mp4")
                
                # Get properties from first chunk
                fgr_cap = cv2.VideoCapture(first_output['fgr_path'])
                fps = fgr_cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(fgr_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                chunk_height = int(fgr_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fgr_cap.release()
                
                # Sort outputs by start_x
                chunk_outputs.sort(key=lambda x: x['start_x'])
                
                # Calculate output width
                output_width = max([chunk['end_x'] for chunk in chunk_outputs])
                output_width = (output_width // MODEL_FACTOR) * MODEL_FACTOR  # Ensure divisibility
                
                # Create high-quality output video writers
                fgr_writer = create_high_quality_writer(
                    final_fgr_path, fps, output_width, chunk_height,
                    video_codec, video_quality, custom_bitrate
                )
                pha_writer = create_high_quality_writer(
                    final_pha_path, fps, output_width, chunk_height,
                    video_codec, video_quality, custom_bitrate
                )
                
                # Open all chunk videos
                chunk_videos = []
                for chunk in chunk_outputs:
                    fgr_cap = cv2.VideoCapture(chunk['fgr_path'])
                    pha_cap = cv2.VideoCapture(chunk['pha_path'])
                    
                    if not fgr_cap.isOpened() or not pha_cap.isOpened():
                        print(f"Warning: Could not open output videos for chunk at {chunk['start_x']}. Skipping.")
                        continue
                    
                    chunk_videos.append({
                        'fgr_cap': fgr_cap,
                        'pha_cap': pha_cap,
                        'start_x': chunk['start_x'],
                        'end_x': chunk['end_x'],
                        'width': chunk['width']
                    })
                
                # Create blending weights for each chunk with enhanced 2-chunk handling
                print("Creating blending weights...")
                blend_weights = []
                
                # Special weight function for 2-chunk case
                is_two_chunks = len(chunk_videos) == 2
                
                for i, chunk in enumerate(chunk_videos):
                    # Create weight mask (1.0 in center, fade at edges)
                    weight = np.ones((chunk_height, chunk['width']), dtype=np.float32)
                    
                    # Calculate feather width for left and right edges
                    left_feather = right_feather = 0
                    
                    # Check for overlap with left neighbor
                    if i > 0:
                        prev_chunk = chunk_videos[i-1]
                        left_overlap = max(0, prev_chunk['end_x'] - chunk['start_x'])
                        if left_overlap > 0:
                            if is_two_chunks:
                                # For 2-chunk case, use larger feather for smoother transition
                                left_feather = min(left_overlap, chunk['width'] // 2)
                            else:
                                left_feather = min(left_overlap, chunk['width'] // 3)
                    
                    # Check for overlap with right neighbor
                    if i < len(chunk_videos) - 1:
                        next_chunk = chunk_videos[i+1]
                        right_overlap = max(0, chunk['end_x'] - next_chunk['start_x'])
                        if right_overlap > 0:
                            if is_two_chunks:
                                # For 2-chunk case, use larger feather for smoother transition
                                right_feather = min(right_overlap, chunk['width'] // 2)
                            else:
                                right_feather = min(right_overlap, chunk['width'] // 3)
                    
                    # Apply left feathering if needed
                    if left_feather > 0:
                        for x in range(left_feather):
                            if is_two_chunks:
                                # Use more pronounced S-curve for 2-chunk case
                                t = x / left_feather
                                # Enhanced cubic ease with more central contrast
                                alpha = t * t * t * (10 + t * (6 * t - 15))
                            else:
                                # Standard cubic ease for 3+ chunks
                                t = x / left_feather
                                alpha = t * t * (3 - 2 * t)  # Smooth S-curve
                            weight[:, x] = alpha
                    
                    # Apply right feathering if needed
                    if right_feather > 0:
                        for x in range(right_feather):
                            idx = chunk['width'] - x - 1  # Right edge
                            if idx >= 0:  # Sanity check
                                if is_two_chunks:
                                    # Use more pronounced S-curve for 2-chunk case
                                    t = x / right_feather
                                    # Enhanced cubic ease with more central contrast
                                    alpha = t * t * t * (10 + t * (6 * t - 15))
                                else:
                                    # Standard cubic ease for 3+ chunks
                                    t = x / right_feather
                                    alpha = t * t * (3 - 2 * t)  # Smooth S-curve
                                weight[:, idx] = alpha
                    
                    # Expand to 3 channels for RGB blending
                    weight_3ch = np.repeat(weight[:, :, np.newaxis], 3, axis=2)
                    blend_weights.append(weight_3ch)
                    
                    # Save a debug image of the weight mask for visualization
                    debug_path = os.path.join(temp_dir, f"weight_mask_{i}.png")
                    cv2.imwrite(debug_path, (weight * 255).astype(np.uint8))
                
                # Process all frames
                report_progress(82, "Reassembling", "Beginning frame reassembly")
                print("Reassembling frames...")
                start_time = time.time()
                
                # Use a reduced frame count if any video has fewer frames
                min_frames = frame_count
                for chunk in chunk_videos:
                    chunk_frames = int(chunk['fgr_cap'].get(cv2.CAP_PROP_FRAME_COUNT))
                    min_frames = min(min_frames, chunk_frames)
                
                if min_frames < frame_count:
                    print(f"Warning: Using {min_frames} frames instead of {frame_count} (mismatch detected)")
                    frame_count = min_frames
                
                # Process each frame
                for frame_idx in range(frame_count):
                    try:
                        # Create output canvas
                        fgr_canvas = np.zeros((chunk_height, output_width, 3), dtype=np.float32)
                        pha_canvas = np.zeros((chunk_height, output_width, 3), dtype=np.float32)
                        
                        # Track accumulated weights
                        weight_sum = np.zeros((chunk_height, output_width, 3), dtype=np.float32)
                        
                        # Add each chunk with blending
                        for i, chunk in enumerate(chunk_videos):
                            # Read frames
                            ret_fgr, fgr_frame = chunk['fgr_cap'].read()
                            ret_pha, pha_frame = chunk['pha_cap'].read()
                            
                            if not ret_fgr or not ret_pha:
                                continue  # Skip this chunk if we can't read frames
                            
                            # Get the region in the output canvas
                            start_x = chunk['start_x']
                            end_x = min(start_x + chunk['width'], output_width)
                            
                            if start_x >= end_x or start_x >= output_width:
                                continue  # Skip invalid regions
                            
                            # Make sure frames match expected dimensions
                            expected_width = chunk['width']
                            actual_width = fgr_frame.shape[1]
                            
                            # If dimensions don't match, resize the frames
                            if fgr_frame.shape[1] != expected_width or fgr_frame.shape[0] != chunk_height:
                                fgr_frame = cv2.resize(fgr_frame, (expected_width, chunk_height), interpolation=cv2.INTER_LINEAR)
                                print(f"Resized frame for chunk {i} from {actual_width}x{fgr_frame.shape[0]} to {expected_width}x{chunk_height}")
                                
                            if pha_frame.shape[1] != expected_width or pha_frame.shape[0] != chunk_height:
                                pha_frame = cv2.resize(pha_frame, (expected_width, chunk_height), interpolation=cv2.INTER_LINEAR)
                            
                            # Convert to float32 for calculations
                            fgr_float = fgr_frame.astype(np.float32)
                            pha_float = pha_frame.astype(np.float32)
                            
                            # Make sure blend weights match frame dimensions
                            weight = blend_weights[i]
                            if weight.shape[1] != expected_width or weight.shape[0] != chunk_height:
                                weight = cv2.resize(weight, (expected_width, chunk_height), interpolation=cv2.INTER_LINEAR)
                            
                            # Apply weight mask
                            fgr_weighted = fgr_float * weight
                            pha_weighted = pha_float * weight
                            
                            # Ensure we don't try to use more of the chunk than exists
                            width_to_use = min(end_x - start_x, expected_width)
                            end_x = start_x + width_to_use
                            
                            try:
                                # Add weighted contribution
                                fgr_canvas[:, start_x:end_x] += fgr_weighted[:, :width_to_use]
                                pha_canvas[:, start_x:end_x] += pha_weighted[:, :width_to_use]
                                
                                # Add to weight sum
                                weight_sum[:, start_x:end_x] += weight[:, :width_to_use]
                            except Exception as e:
                                print(f"Error blending chunk {i} at position {start_x}:{end_x}: {str(e)}")
                                print(f"Shapes - Canvas:{fgr_canvas[:, start_x:end_x].shape}, Weighted:{fgr_weighted[:, :width_to_use].shape}")
                                continue
                        
                        # Normalize by weight sum (avoid division by zero)
                        weight_sum = np.maximum(weight_sum, 1e-6)
                        fgr_output = fgr_canvas / weight_sum
                        pha_output = pha_canvas / weight_sum
                        
                        # Convert to uint8
                        fgr_output = np.clip(fgr_output, 0, 255).astype(np.uint8)
                        pha_output = np.clip(pha_output, 0, 255).astype(np.uint8)
                        
                        # Write frames
                        fgr_writer.write(fgr_output)
                        pha_writer.write(pha_output)
                        
                        # Show progress periodically
                        if frame_idx % 5 == 0 or frame_idx == frame_count - 1:
                            elapsed = time.time() - start_time
                            fps_rate = (frame_idx + 1) / max(0.001, elapsed)
                            eta = (frame_count - frame_idx - 1) / max(0.001, fps_rate)
                            
                            # Calculate progress (82-95% range for reassembly)
                            reassembly_progress = 82 + ((frame_idx + 1) / frame_count * 13)
                            
                            progress_msg = f"Reassembled {frame_idx+1}/{frame_count} frames ({(frame_idx+1)/frame_count*100:.1f}%)"
                            report_progress(reassembly_progress, "Reassembling", progress_msg)
                            
                            print(f"Reassembled {frame_idx+1}/{frame_count} frames "
                                f"({(frame_idx+1)/frame_count*100:.1f}%) - "
                                f"{fps_rate:.1f} fps, ETA: {eta:.1f}s")
                        
                        # Check for interrupts
                        processor.check_interrupt()
                        
                    except Exception as e:
                        print(f"Error reassembling frame {frame_idx}: {str(e)}")
                        # Try to continue with next frame
                
                # Release resources
                for chunk in chunk_videos:
                    chunk['fgr_cap'].release()
                    chunk['pha_cap'].release()
                
                fgr_writer.release()
                pha_writer.release()
                
                report_progress(95, "Finishing", "Reassembly complete")
                print("Reassembly complete!")
                
                # Clean up temporary files if enabled
                if cleanup_temp:
                    cleanup_temporary_files(temp_files, True)
                
                # Report completion
                report_progress(100, "Complete", "Chunked video processing completed successfully")
                
                # Return paths to the output videos
                return final_fgr_path, final_pha_path
            
            else:
                # Image sequence not yet supported
                print("Image sequence chunking is not yet fully implemented.")
                print("Processing without chunking instead.")
                # Clean up temp files
                cleanup_temporary_files(temp_files, True)
                # Report fallback progress
                report_progress(10, "Fallback", "Falling back to processing without chunking")
                
                # Fall back to processing without chunking
                return processor.process_video(input_path=input_path, mask_path=mask_path, 
                                            output_path=output_path,
                                            progress_callback=progress_callback,
                                            **kwargs)
        
        except Exception as e:
            print(f"Error during chunked processing: {str(e)}")
            traceback.print_exc()
            print("Falling back to processing without chunking.")
            
            # Clean up temporary files
            cleanup_temporary_files(temp_files, True)
            
            # Report error and fallback
            report_progress(10, "Error", f"Error during chunked processing: {str(e)}")
            
            # Try processing without chunking as fallback
            return processor.process_video(input_path=input_path, mask_path=mask_path, 
                                        output_path=output_path,
                                        progress_callback=progress_callback,
                                        **kwargs)
    
    else:
        # Handle image sequence input
        print("Image sequence chunking is not yet implemented.")
        print("Processing without chunking instead.")
        return processor.process_video(input_path=input_path, mask_path=mask_path, 
                                    output_path=output_path, **kwargs)
