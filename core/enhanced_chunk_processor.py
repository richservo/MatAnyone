"""
# enhanced_chunk_processor.py - v1.1734584255
# Updated: Wednesday, May 22, 2025
# Changes in this version:
# - Integrated new memory_utils with tensor pooling and improved memory management
# - Added MaskManager-based operations for efficient mask processing
# - Implemented parallel chunk processing for better resource utilization
# - Integrated VideoProcessor for optimized video operations
# - Reduced disk I/O through caching and in-memory processing
# - Improved error handling and resource cleanup
# - Fixed temp directory creation to prevent empty folders
"""

import os
import cv2
import numpy as np
import time
import traceback
import gc
from pathlib import Path

# Import video utilities
from utils.video_utils import blend_videos, cleanup_temporary_files, reverse_video, concatenate_videos
from utils.video_processor import VideoProcessor

# Import utility modules
from chunking.chunking_utils import (
    get_strip_chunks, 
    get_grid_chunks, 
    create_low_res_video,
    get_autochunk_segments
)
from mask.mask_analysis import (
    upscale_and_binarize_mask_video,
    analyze_masks_for_optimal_ranges,
    create_optimal_mask_for_range
)
from utils.extraction_utils import (
    extract_last_frame,
    extract_chunk_frame_range,
    extract_chunk_frame_range_reversed,
    create_full_video_from_ranges
)
from utils.reassembly_utils import (
    reassemble_strip_chunks,
    reassemble_grid_chunks,
    reassemble_arbitrary_chunks
)

# Import enhanced memory management
from utils.memory_utils import (
    clear_gpu_memory,
    get_cached_processor,
    get_tensor_from_pool,
    return_tensor_to_pool,
    print_memory_stats,
    clear_processor_pool
)

# Import enhanced mask operations
from mask.mask_operations import (
    mask_manager,
    dilate_mask,
    erode_mask,
    combine_masks,
    batch_process_masks,
    clear_mask_cache
)

# Import parallel processing
from chunking.parallel_processor import (
    process_chunks_parallel,
    ParallelChunkProcessor
)

from chunking.chunk_optimizer import propagate_mask_data  # Import for mask propagation

# Import heat map based chunk placement
from chunking.heat_map_analyzer import HeatMapAnalyzer
from chunking.smart_chunk_placer import SmartChunkPlacer


def process_video_with_enhanced_chunking(processor, input_path, mask_path, output_path, num_chunks=2, 
                                         bidirectional=True, blend_method='weighted', lowres_blend_method=None, 
                                         reverse_dilate=15, cleanup_temp=True, mask_skip_threshold=5, 
                                         allow_native_resolution=True, low_res_scale=0.25, chunk_type='strips',
                                         prioritize_faces=True, use_autochunk=False, apply_expanded_mask=True,
                                         optimize_masks=True, parallel_processing=True, max_workers=None, 
                                         use_heat_map_chunking=False, face_priority_weight=3.0, **kwargs):
    """
    Helper function to check for interrupts throughout the enhanced chunk processing
    """
    def check_for_interrupt():
        if hasattr(processor, 'check_interrupt') and callable(processor.check_interrupt):
            processor.check_interrupt()
            
    # Check for interrupt at start of processing
    check_for_interrupt()
    """
    Enhanced chunk processing with low-resolution preprocessing to ensure continuous masks across chunks.
    
    This approach:
    1. Creates a low-resolution bidirectional mask of the entire video
    2. Analyzes each chunk to find frame ranges with meaningful mask content
    3. Identifies frames with optimal mask coverage for each range
    4. Processes only those ranges and creates empty videos for the rest
    5. Reassembles everything into a continuous result with proper memory management
    
    Args:
        processor: The InterruptibleInferenceCore processor instance
        input_path: Path to the input video
        mask_path: Path to mask image
        output_path: Directory to save outputs
        num_chunks: Number of chunks to split the video into (strips or grid)
                   [Ignored when use_autochunk is True]
        bidirectional: Whether to use bidirectional processing for each chunk
        blend_method: Method to blend forward and reverse passes ('weighted', 'max_alpha', 'min_alpha', 'average')
        lowres_blend_method: Method to blend forward and reverse passes for the low-res mask (defaults to blend_method if None)
        reverse_dilate: Dilation radius for the mask in reverse pass
        cleanup_temp: Whether to clean up temporary files
        mask_skip_threshold: Percentage threshold of non-zero pixels to consider a mask chunk worth processing
        allow_native_resolution: Whether to allow processing at native resolution
        low_res_scale: Scale factor for low-resolution preprocessing (0.25 = quarter resolution)
        chunk_type: Type of chunking - 'strips' for horizontal strips or 'grid' for aspect-ratio preserving rectangles
        prioritize_faces: Whether to prioritize frames with faces as keyframes
        use_autochunk: Whether to automatically determine chunks based on low-res resolution
        apply_expanded_mask: Whether to multiply final output with expanded original mask to eliminate artifacts
        optimize_masks: Whether to use mask optimization to maximize mask information across chunks
        parallel_processing: Whether to use parallel processing for chunks
        max_workers: Maximum number of parallel workers (None = auto-detect)
        **kwargs: Additional arguments to pass to process_video
    
    Returns:
        Tuple of paths to the final foreground and alpha videos
    """
    # Initialize video processor for efficient video operations
    video_processor = VideoProcessor(temp_dir=os.path.join(output_path, "video_processor_temp"),
                                   cleanup_temp=True, verbose=True)

    # If use_autochunk is enabled, we will completely ignore the num_chunks parameter
    # We'll calculate the optimal number of chunks based on the low-res dimensions
    if use_autochunk:
        print("Auto-chunk mode enabled - will automatically determine the optimal number of chunks")
        # We'll set a minimum of 2 chunks for enhanced processing, but actual chunk count 
        # will be determined by get_autochunk_segments function
        forced_num_chunks = 2
    else:
        # When auto-chunk is disabled, use the user-specified number of chunks
        forced_num_chunks = num_chunks
    
    if forced_num_chunks <= 1 and not use_autochunk:
        # No chunking needed, just process as normal
        return processor.process_video(input_path=input_path, mask_path=mask_path, 
                                     output_path=output_path, 
                                     bidirectional=bidirectional, 
                                     blend_method=blend_method, 
                                     reverse_dilate=reverse_dilate,
                                     cleanup_temp=cleanup_temp, 
                                     **kwargs)
    
    # Ensure bidirectional is enabled - we always want to do bidirectional processing
    if not bidirectional:
        print("Notice: Bidirectional processing is always enabled in enhanced chunk processing for best results")
        bidirectional = True
    
    # Determine if we're dealing with a video or image sequence
    is_video = os.path.isfile(input_path) and input_path.endswith(('.mp4', '.mov', '.avi'))
    
    if not is_video:
        print("Enhanced chunk processing is currently only supported for video files, not image sequences")
        print("Falling back to standard chunk processing")
        from core.chunk_processor import process_video_in_chunks
        return process_video_in_chunks(processor, input_path, mask_path, output_path, forced_num_chunks,
                                     bidirectional, blend_method, reverse_dilate, cleanup_temp, 
                                     mask_skip_threshold, allow_native_resolution, **kwargs)
    
    # Create temporary directories for chunks and outputs
    temp_dir = os.path.join(output_path, "enhanced_chunks_temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Keep track of temporary files and directories for cleanup
    temp_files = [temp_dir]  # Add temp_dir to the list for cleanup
    
    # Default video parameters in case we need to fall back
    video_name = os.path.basename(input_path)
    if video_name.endswith(('.mp4', '.mov', '.avi')):
        video_name = os.path.splitext(video_name)[0]
    
    try:
        # Get video information using VideoProcessor
        video_info = video_processor.get_video_info(input_path)
        
        if "error" in video_info:
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        fps = video_info["fps"]
        frame_count = video_info["frame_count"]
        original_height = video_info["height"]
        original_width = video_info["width"]
        
        # Critical: Ensure dimensions are model-friendly
        # MatAnyone models require dimensions divisible by 8
        MODEL_FACTOR = 8
        
        # Adjust dimensions to ensure they're divisible by MODEL_FACTOR
        adjusted_height = (original_height // MODEL_FACTOR) * MODEL_FACTOR
        adjusted_width = (original_width // MODEL_FACTOR) * MODEL_FACTOR
        
        if adjusted_height != original_height or adjusted_width != original_width:
            print(f"Adjusting dimensions from {original_width}x{original_height} to {adjusted_width}x{adjusted_height} to ensure divisibility by {MODEL_FACTOR}")
            
        # Check for interrupt before Step 1
        check_for_interrupt()
        
        # Step 1: Create low-resolution mask
        print(f"STEP 1: Creating low-resolution bidirectional mask for the entire video (scale factor: {low_res_scale})")
        
        # Create a low-resolution version of the input video
        low_res_height = int(adjusted_height * low_res_scale)
        low_res_width = int(adjusted_width * low_res_scale)
        
        # Make sure low res dimensions are divisible by 8 as well
        low_res_height = (low_res_height // MODEL_FACTOR) * MODEL_FACTOR
        low_res_width = (low_res_width // MODEL_FACTOR) * MODEL_FACTOR
        
        print(f"Low-resolution dimensions: {low_res_width}x{low_res_height}")
        
        # Create low-res video using VideoProcessor
        low_res_video_path = os.path.join(temp_dir, f"{video_name}_low_res.mp4")
        temp_files.append(low_res_video_path)
        
        video_processor.process_video(
            input_path=input_path,
            operations=[
                {
                    "type": "resize",
                    "width": low_res_width,
                    "height": low_res_height,
                    "maintain_aspect_ratio": False
                }
            ],
            output_path=low_res_video_path
        )
        
        # Create a low-res version of the mask using MaskManager
        low_res_mask_path = os.path.join(temp_dir, f"{video_name}_low_res_mask.png")
        temp_files.append(low_res_mask_path)
        
        # Load the original mask via MaskManager
        original_mask = mask_manager.get_mask(mask_path)
        if original_mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")
        
        # Check for keyframe metadata in mask
        from mask.mask_utils import get_keyframe_metadata_from_mask
        keyframe_number = get_keyframe_metadata_from_mask(mask_path)
        
        if keyframe_number is not None:
            print(f"Found keyframe metadata: frame {keyframe_number}. Using keyframe-based Step 1 processing.")
        else:
            print("No keyframe metadata found. Using standard Step 1 processing.")
        
        # Resize mask using mask_operations
        mask_operations = [
            {
                "op_type": "resize",
                "input_path": mask_path,
                "output_path": low_res_mask_path,
                "params": {
                    "width": low_res_width,
                    "height": low_res_height
                }
            }
        ]
        
        # Execute mask operations in batch
        batch_result = batch_process_masks(mask_operations)
        if not batch_result.get(0, {}).get("success", False):
            raise ValueError(f"Error creating low-res mask: {batch_result.get(0, {}).get('error', 'Unknown error')}")
        
        # Check for interrupt before processing
        check_for_interrupt()
        
        # Process the low-res video bidirectionally to create a continuous mask
        low_res_process_params = kwargs.copy()
        
        # Remove ProPainter-specific parameters that MatAnyone doesn't understand
        propainter_params = ['invert_mask', 'neighbor_length', 'ref_stride', 'mask_dilation', 
                           'subvideo_length', 'fp16', 'resize_ratio', 'low_res_width', 'low_res_height']
        for param in propainter_params:
            low_res_process_params.pop(param, None)
        
        # Override parameters for low-res processing
        low_res_process_params.update({
            'max_size': -1,  # Use native resolution since it's already low-res
            'save_image': False,  # Don't save individual frames for low-res version
            'suffix': 'low_res',
            'bidirectional': False,  # We handle bidirectional manually
            'blend_method': blend_method if lowres_blend_method is None else lowres_blend_method
        })
        
        # Process low-res video - clear cache first to ensure clean state
        clear_gpu_memory(processor, force_full_cleanup=True)
        
        # Check for interrupt before processing
        check_for_interrupt()
        
        # ALWAYS use MatAnyone for low-res mask generation, regardless of selected model
        # This is because we need MatAnyone's mask propagation capabilities
        lowres_processor = processor
        if hasattr(processor, 'model_type') and processor.model_type.lower() != 'matanyone':
            print("Using MatAnyone for low-res mask generation (required for mask propagation)")
            from core.inference_core import InterruptibleInferenceCore
            lowres_processor = InterruptibleInferenceCore(model_type="matanyone")
        else:
            print("Using MatAnyone for low-res mask generation")
        
        if keyframe_number is not None:
            # NEW KEYFRAME-BASED STEP 1 LOGIC
            print(f"Processing low-resolution video with keyframe {keyframe_number} as pivot point...")
            
            # Cut video into two segments at keyframe
            keyframe_start_segment = os.path.join(temp_dir, f"{video_name}_low_res_start_to_keyframe.mp4")
            keyframe_end_segment = os.path.join(temp_dir, f"{video_name}_low_res_keyframe_to_end.mp4")
            temp_files.extend([keyframe_start_segment, keyframe_end_segment])
            
            # Extract keyframe->end segment (forward processing)
            print(f"Extracting keyframe->end segment (frames {keyframe_number} to {frame_count-1})")
            video_processor.extract_chunk(
                input_path=low_res_video_path,
                output_path=keyframe_end_segment,
                start_frame=keyframe_number,
                end_frame=frame_count
            )
            
            # Extract start->keyframe segment (backward processing)
            print(f"Extracting start->keyframe segment (frames 0 to {keyframe_number})")
            video_processor.extract_chunk(
                input_path=low_res_video_path,
                output_path=keyframe_start_segment,
                start_frame=0,
                end_frame=keyframe_number + 1
            )
            
            # Process keyframe->end segment forward
            print("Processing keyframe->end segment forward...")
            forward_params = low_res_process_params.copy()
            forward_params['suffix'] = 'low_res_forward'
            
            low_res_fgr_forward, low_res_pha_forward = lowres_processor.process_video(
                input_path=keyframe_end_segment,
                mask_path=low_res_mask_path,
                output_path=temp_dir,
                **forward_params
            )
            
            # Clear memory before backward processing
            clear_gpu_memory(processor, force_full_cleanup=True)
            
            # Check for interrupt before backward processing
            check_for_interrupt()
            
            # Process start->keyframe segment backward (reverse video first)
            print("Processing start->keyframe segment backward...")
            keyframe_start_reversed = os.path.join(temp_dir, f"{video_name}_low_res_start_to_keyframe_reversed.mp4")
            video_processor.reverse_video(keyframe_start_segment, keyframe_start_reversed)
            temp_files.append(keyframe_start_reversed)
            
            reverse_params = low_res_process_params.copy()
            reverse_params['suffix'] = 'low_res_reverse'
            
            low_res_fgr_reverse, low_res_pha_reverse = lowres_processor.process_video(
                input_path=keyframe_start_reversed,
                mask_path=low_res_mask_path,
                output_path=temp_dir,
                **reverse_params
            )
            
            # Re-reverse the backward segment to correct temporal order
            low_res_fgr_backward_corrected = os.path.join(temp_dir, f"{video_name}_low_res_backward_corrected_fgr.mp4")
            low_res_pha_backward_corrected = os.path.join(temp_dir, f"{video_name}_low_res_backward_corrected_pha.mp4")
            
            video_processor.reverse_video(low_res_fgr_reverse, low_res_fgr_backward_corrected)
            video_processor.reverse_video(low_res_pha_reverse, low_res_pha_backward_corrected)
            
            temp_files.extend([
                low_res_fgr_forward, low_res_pha_forward,
                low_res_fgr_reverse, low_res_pha_reverse,
                low_res_fgr_backward_corrected, low_res_pha_backward_corrected
            ])
            
            # Recombine segments ensuring no duplicate keyframe
            print("Recombining segments with perfect frame alignment...")
            low_res_fgr_blended = os.path.join(temp_dir, f"{video_name}_low_res_fgr_blended.mp4")
            low_res_pha_blended = os.path.join(temp_dir, f"{video_name}_low_res_pha_blended.mp4")
            
            # Concatenate: backward_corrected (0 to keyframe-1) + forward (keyframe to end)
            # Extract frames 0 to keyframe-1 from backward_corrected
            backward_trimmed_fgr = os.path.join(temp_dir, f"{video_name}_backward_trimmed_fgr.mp4")
            backward_trimmed_pha = os.path.join(temp_dir, f"{video_name}_backward_trimmed_pha.mp4")
            
            if keyframe_number > 0:
                video_processor.extract_chunk(
                    input_path=low_res_fgr_backward_corrected,
                    output_path=backward_trimmed_fgr,
                    start_frame=0,
                    end_frame=keyframe_number
                )
                video_processor.extract_chunk(
                    input_path=low_res_pha_backward_corrected,
                    output_path=backward_trimmed_pha,
                    start_frame=0,
                    end_frame=keyframe_number
                )
                
                # Concatenate backward_trimmed + forward
                concatenate_videos([backward_trimmed_fgr, low_res_fgr_forward], low_res_fgr_blended)
                concatenate_videos([backward_trimmed_pha, low_res_pha_forward], low_res_pha_blended)
                
                temp_files.extend([backward_trimmed_fgr, backward_trimmed_pha])
            else:
                # If keyframe is 0, just use forward segment
                low_res_fgr_blended = low_res_fgr_forward
                low_res_pha_blended = low_res_pha_forward
                
        else:
            # STANDARD STEP 1 LOGIC (no keyframe metadata)
            print("Processing low-resolution video bidirectionally to create continuous mask...")
            
            # Process the low-res video bidirectionally
            print("Processing low-res forward pass...")
            
            # Forward pass
            forward_params = low_res_process_params.copy()
            forward_params['suffix'] = 'low_res_forward'
            
            low_res_fgr_forward, low_res_pha_forward = lowres_processor.process_video(
                input_path=low_res_video_path,
                mask_path=low_res_mask_path,  # Use the low-res mask for forward pass
                output_path=temp_dir,
                **forward_params
            )
            
            # Clear memory before reverse pass
            clear_gpu_memory(lowres_processor, force_full_cleanup=True)
            
            # Check for interrupt before reverse pass
            check_for_interrupt()
            
            print("Processing low-res reverse pass...")
            
            # Create reversed video for backward pass
            low_res_reversed_path = os.path.join(temp_dir, f"{video_name}_low_res_reversed.mp4")
            video_processor.reverse_video(low_res_video_path, low_res_reversed_path)
            temp_files.append(low_res_reversed_path)
            
            # Get last frame from forward pass to use as mask for reverse pass
            last_frame_mask_path = extract_last_frame(low_res_pha_forward, temp_dir, f"{video_name}_low_res_last_frame")
            
            # Dilate the mask for reverse pass if requested using mask_operations
            if reverse_dilate > 0:
                dilated_mask_path = os.path.join(temp_dir, f"{video_name}_low_res_dilated_mask.png")
                dilate_mask(last_frame_mask_path, dilated_mask_path, reverse_dilate)
                last_frame_mask_path = dilated_mask_path
                temp_files.append(dilated_mask_path)
            
            temp_files.append(last_frame_mask_path)
            
            # Process the reversed low-res video
            reverse_params = low_res_process_params.copy()
            reverse_params['suffix'] = 'low_res_reverse'
            
            low_res_fgr_reverse, low_res_pha_reverse = lowres_processor.process_video(
                input_path=low_res_reversed_path,
                mask_path=last_frame_mask_path,  # Use the last frame from forward pass as mask for reverse
                output_path=temp_dir,
                **reverse_params
            )
            
            # Re-reverse the output using VideoProcessor
            low_res_fgr_re_reversed = os.path.join(temp_dir, f"{video_name}_low_res_re_reversed_fgr.mp4")
            low_res_pha_re_reversed = os.path.join(temp_dir, f"{video_name}_low_res_re_reversed_pha.mp4")
            
            video_processor.reverse_video(low_res_fgr_reverse, low_res_fgr_re_reversed)
            video_processor.reverse_video(low_res_pha_reverse, low_res_pha_re_reversed)
            
            temp_files.extend([
                low_res_fgr_forward, low_res_pha_forward,
                low_res_fgr_reverse, low_res_pha_reverse,
                low_res_fgr_re_reversed, low_res_pha_re_reversed
            ])
            
            # Check for interrupt before blending
            check_for_interrupt()
            
            # Blend forward and reverse passes using VideoProcessor
            print("Blending low-res passes...")
            
            # Use separate blending method for low-res if specified, otherwise use the main blend method
            lowres_blend = lowres_blend_method if lowres_blend_method is not None else blend_method
            print(f"Using '{lowres_blend}' blending method for low-resolution mask")
                
            low_res_fgr_blended = os.path.join(temp_dir, f"{video_name}_low_res_fgr_blended.mp4")
            low_res_pha_blended = os.path.join(temp_dir, f"{video_name}_low_res_pha_blended.mp4")
            
            # Use video_processor for blending
            blend_videos(low_res_fgr_forward, low_res_fgr_re_reversed, low_res_fgr_blended, lowres_blend)
            blend_videos(low_res_pha_forward, low_res_pha_re_reversed, low_res_pha_blended, lowres_blend)
        
        temp_files.extend([low_res_fgr_blended, low_res_pha_blended])
        
        # Extract frames from the blended low-res mask video for analysis
        full_res_mask_dir = os.path.join(temp_dir, "full_res_masks")
        os.makedirs(full_res_mask_dir, exist_ok=True)
        temp_files.append(full_res_mask_dir)
        
        # Clear memory again for next steps
        clear_gpu_memory(processor, force_full_cleanup=True)  # Updated with force_full_cleanup
        
        # The low_res_pha_blended is our continuous mask, but we need to upscale it to full resolution
        print("STEP 2: Upscaling low-resolution mask to full resolution and binarizing")
        
        # Extract frames from the low-res mask video, upscale, and binarize them
        upscale_and_binarize_mask_video(
            low_res_pha_blended, 
            full_res_mask_dir, 
            adjusted_width, 
            adjusted_height,
            threshold=128  # Threshold for binary mask
        )
        
        # Now we have a directory with full-resolution binary mask frames
        
        # Check for interrupt before step 3
        check_for_interrupt()
        
        # Step 3: Divide the video into chunks based on chunk_type
        print(f"STEP 3: Dividing video into chunks")
        
        # Use heat map based chunking if requested
        if use_heat_map_chunking:
            print("Using heat map based intelligent chunk placement")
            
            # Create heat map analyzer
            heat_map_analyzer = HeatMapAnalyzer(face_priority_weight=face_priority_weight)
            
            # Analyze the full-resolution mask frames to create heat map
            # Also try to get original frames for face detection
            original_frames_dir = None
            if prioritize_faces:
                # Check if we have access to original frames
                potential_frames_dir = os.path.join(temp_dir, f"{video_name}_frames")
                if os.path.exists(potential_frames_dir):
                    original_frames_dir = potential_frames_dir
                else:
                    print("Note: Original frames not available for face detection, using mask-only heat map")
            
            # Generate heat map
            heat_map = heat_map_analyzer.analyze_mask_sequence(full_res_mask_dir, original_frames_dir)
            
            # Save heat map visualization for debugging
            heat_map_vis_path = os.path.join(temp_dir, f"{video_name}_heat_map.png")
            heat_map_analyzer.save_heat_map_visualization(heat_map_vis_path)
            temp_files.append(heat_map_vis_path)
            
            # Get activity statistics
            activity_stats = heat_map_analyzer.get_activity_stats()
            print(f"Heat map stats: mean activity={activity_stats['mean_activity']:.3f}, "
                  f"active pixels={activity_stats['activity_ratio']*100:.1f}%")
            
            # Use smart chunk placer to find optimal positions
            chunk_placer = SmartChunkPlacer(overlap_ratio=0.2)  # 20% overlap as before
            chunk_segments = chunk_placer.find_optimal_chunk_placement(
                heat_map, low_res_width, low_res_height, MODEL_FACTOR
            )
            
            # Save chunk placement visualization
            chunk_vis_path = os.path.join(temp_dir, f"{video_name}_chunk_placement.png")
            chunk_placer.visualize_chunk_placement(heat_map, chunk_vis_path)
            temp_files.append(chunk_vis_path)
            
            # Get coverage statistics
            coverage_stats = chunk_placer.get_chunk_coverage_stats(heat_map)
            print(f"Chunk placement achieved {coverage_stats['coverage']*100:.1f}% coverage "
                  f"with {coverage_stats['num_chunks']} chunks")
            
            # Debug: Print all chunk dimensions
            print("\nChunk dimensions summary:")
            for i, chunk in enumerate(chunk_segments):
                orientation = chunk.get('orientation', 'unknown')
                print(f"  Chunk {i}: {chunk['width']}x{chunk['height']} ({orientation})")
            
        # Use auto-chunking mode if requested
        elif use_autochunk:
            print("Using auto-chunking mode based on low-res resolution")
            chunk_segments = get_autochunk_segments(
                adjusted_width, 
                adjusted_height, 
                low_res_width, 
                low_res_height, 
                MODEL_FACTOR
            )
            
            # Verify that all chunks have exactly the same size as the low-res dimensions
            for i, chunk in enumerate(chunk_segments):
                chunk_width = chunk['x_range'][1] - chunk['x_range'][0]
                chunk_height = chunk['y_range'][1] - chunk['y_range'][0]
                
                if chunk_width != low_res_width or chunk_height != low_res_height:
                    print(f"WARNING: Chunk {i} has incorrect dimensions: {chunk_width}x{chunk_height}, expected {low_res_width}x{low_res_height}")
                    print("This may cause processing failures. Attempting to fix...")
                    
                    # Attempt to fix the chunk dimensions
                    start_x, end_x = chunk['x_range']
                    start_y, end_y = chunk['y_range']
                    
                    end_x = start_x + low_res_width
                    end_y = start_y + low_res_height
                    
                    # Check boundary conditions
                    if end_x > adjusted_width:
                        # Adjust start_x to maintain chunk size
                        start_x = adjusted_width - low_res_width
                        end_x = adjusted_width
                    
                    if end_y > adjusted_height:
                        # Adjust start_y to maintain chunk size
                        start_y = adjusted_height - low_res_height
                        end_y = adjusted_height
                    
                    # Update the chunk info
                    chunk_segments[i]['x_range'] = (start_x, end_x)
                    chunk_segments[i]['y_range'] = (start_y, end_y)
                    chunk_segments[i]['width'] = end_x - start_x
                    chunk_segments[i]['height'] = end_y - start_y
                    
                    print(f"Fixed chunk {i}: X={start_x}-{end_x}, Y={start_y}-{end_y}, Size: {end_x-start_x}x{end_y-start_y}")
            
            # Report the actual number of chunks calculated
            if chunk_segments:
                num_chunks_x = max(col for _, _, _, _, (row, col) in [
                    (c['x_range'][0], c['x_range'][1], c['y_range'][0], c['y_range'][1], c.get('grid_pos', (0, 0))) 
                    for c in chunk_segments
                ]) + 1
                
                num_chunks_y = max(row for _, _, _, _, (row, col) in [
                    (c['x_range'][0], c['x_range'][1], c['y_range'][0], c['y_range'][1], c.get('grid_pos', (0, 0))) 
                    for c in chunk_segments
                ]) + 1
                
                print(f"Auto-chunking created a {num_chunks_y}x{num_chunks_x} grid of chunks")
                print(f"Total chunks: {len(chunk_segments)}")
            else:
                print("Auto-chunking failed to create any valid chunks")
        else:
            # Get chunk segments based on chunk type
            if chunk_type == 'strips':
                chunk_segments = get_strip_chunks(adjusted_width, adjusted_height, forced_num_chunks, MODEL_FACTOR)
            else:  # grid
                chunk_segments = get_grid_chunks(adjusted_width, adjusted_height, forced_num_chunks, MODEL_FACTOR)
                
        # Update num_chunks based on valid segments
        actual_num_chunks = len(chunk_segments)
        print(f"Final chunk count: {actual_num_chunks}")
        
        if actual_num_chunks == 0:
            print("ERROR: No valid chunks could be created. Cannot continue.")
            cleanup_temporary_files(temp_files, cleanup_temp)
            video_processor.cleanup()
            clear_mask_cache()
            raise ValueError("No valid chunks could be created with current settings")
        
        # Check for interrupt before step 4
        check_for_interrupt()
        
        # Step 4: For each chunk, analyze mask to find optimal ranges and frames
        # Clean up lowres processor if we created one
        if lowres_processor != processor:
            clear_gpu_memory(lowres_processor, force_full_cleanup=True)
            lowres_processor = None
            
        print(f"STEP 4: Analyzing masks to identify optimal processing ranges (threshold: {mask_skip_threshold}%)")
        if prioritize_faces:
            print("Using face detection to prioritize frames with faces as keyframes (only within mask regions)")
        
        # Create a structure to hold frame ranges for each chunk
        chunk_optimal_ranges = []
        
        # For each chunk, analyze frame-by-frame to find ranges with mask content
        for chunk_idx, chunk_info in enumerate(chunk_segments):
            # Check for interrupt during chunk analysis
            check_for_interrupt()
            print(f"Analyzing chunk {chunk_idx+1}/{actual_num_chunks}...")
            
            # Extract chunk coordinates
            start_x, end_x = chunk_info['x_range']
            start_y, end_y = chunk_info['y_range']
            
            # Analyze masks to find optimal ranges with best coverage
            optimal_ranges = analyze_masks_for_optimal_ranges(
                full_res_mask_dir, 
                frame_count, 
                start_x, 
                end_x, 
                start_y,
                end_y,
                mask_skip_threshold,
                prioritize_faces=prioritize_faces,
                use_original_frames=True,
                original_video_path=input_path,
                only_detect_faces_in_mask=True  # Only detect faces within the mask region
            )
            
            if optimal_ranges:
                print(f"Chunk {chunk_idx+1} has {len(optimal_ranges)} optimal processing ranges")
                for i, (start_frame, end_frame, keyframe) in enumerate(optimal_ranges):
                    print(f"  Range {i+1}: Frames {start_frame}-{end_frame} with keyframe at {keyframe}")
            else:
                print(f"Chunk {chunk_idx+1} has no significant mask content")
            
            # Add to our list
            chunk_optimal_ranges.append(optimal_ranges)
        
        # Check for interrupt before step 5
        check_for_interrupt()
        
        # Step 5: Create and process chunk videos for each optimal frame range
        print("STEP 5: Processing chunk videos for each optimal range")
        
        # Prepare shared arguments for parallel processing
        shared_args = {
            'processor_model_path': processor.model_path,
            'processor_model_type': getattr(processor, 'model_type', 'matanyone'),  # Get model type, default to matanyone
            'input_path': input_path,
            'original_mask': original_mask,
            'full_res_mask_dir': full_res_mask_dir,
            'temp_dir': temp_dir,
            'frame_count': frame_count,
            'fps': fps,
            'kwargs': kwargs,
            'low_res_width': low_res_width,
            'low_res_height': low_res_height
        }
        
        # Define the chunk processing function for parallel execution
        def process_chunk_function(chunk_info, chunk_idx, shared_args):
            # Check for interrupt at the start of chunk processing
            if 'processor_model_path' in shared_args:
                # Get processor for checking interrupt
                from utils.memory_utils import get_cached_processor
                proc = get_cached_processor(shared_args['processor_model_path'], shared_args.get('processor_model_type', 'matanyone'))
                if hasattr(proc, 'check_interrupt') and callable(proc.check_interrupt):
                    proc.check_interrupt()
            # Extract shared arguments
            processor_model_path = shared_args['processor_model_path']
            processor_model_type = shared_args.get('processor_model_type', 'matanyone')
            low_res_width = shared_args['low_res_width']
            low_res_height = shared_args['low_res_height']
            input_path = shared_args['input_path']
            original_mask = shared_args['original_mask']
            full_res_mask_dir = shared_args['full_res_mask_dir']
            temp_dir = shared_args['temp_dir']
            frame_count = shared_args['frame_count']
            fps = shared_args['fps']
            kwargs = shared_args['kwargs']
            
            # Extract chunk coordinates
            start_x, end_x = chunk_info['x_range']
            start_y, end_y = chunk_info['y_range']
            
            # Get optimal ranges for this chunk
            optimal_ranges = chunk_optimal_ranges[chunk_idx]
            
            # Create directory for this chunk's outputs
            chunk_output_dir = os.path.join(temp_dir, f"output_chunk_{chunk_idx}")
            os.makedirs(chunk_output_dir, exist_ok=True)
            
            # If no ranges with content, create empty output
            if not optimal_ranges:
                print(f"Chunk {chunk_idx+1}/{actual_num_chunks} has no frame ranges with mask content")
                
                # Create empty output videos
                chunk_width = end_x - start_x
                chunk_height = end_y - start_y
                chunk_fgr = os.path.join(chunk_output_dir, f"empty_fgr.mp4")
                chunk_pha = os.path.join(chunk_output_dir, f"empty_pha.mp4")
                
                # Get a processor from cache or create new one
                chunk_processor = get_cached_processor(processor_model_path, processor_model_type)
                chunk_processor.create_empty_video(chunk_fgr, chunk_width, chunk_height, fps, frame_count, alpha=False)
                chunk_processor.create_empty_video(chunk_pha, chunk_width, chunk_height, fps, frame_count, alpha=True)
                
                # Add to chunk outputs
                return {
                    'fgr_path': chunk_fgr,
                    'pha_path': chunk_pha,
                    'x_range': (start_x, end_x),
                    'y_range': (start_y, end_y),
                    'width': chunk_width,
                    'height': chunk_height
                }
            
            # Create a list to store results for this chunk
            range_results = []
            
            # Initialize video processor for this chunk
            chunk_video_processor = VideoProcessor(
                temp_dir=os.path.join(chunk_output_dir, "video_temp"),
                cleanup_temp=True
            )
            
            # Process each optimal range
            for range_idx, (start_frame, end_frame, keyframe) in enumerate(optimal_ranges):
                # Check for interrupt in the range processing loop
                if 'processor_model_path' in shared_args:
                    # Get processor for checking interrupt
                    from utils.memory_utils import get_cached_processor
                    proc = get_cached_processor(shared_args['processor_model_path'], shared_args.get('processor_model_type', 'matanyone'))
                    if hasattr(proc, 'check_interrupt') and callable(proc.check_interrupt):
                        proc.check_interrupt()
                print(f"Processing chunk {chunk_idx+1}/{actual_num_chunks}, range {range_idx+1}/{len(optimal_ranges)} (frames {start_frame}-{end_frame}, keyframe {keyframe})")
                
                # If the range is very short (less than 3 frames), just process it directly
                if end_frame - start_frame < 3:
                    # Create a sub-video for this chunk and frame range
                    range_output_dir = os.path.join(chunk_output_dir, f"range_{range_idx}_output")
                    os.makedirs(range_output_dir, exist_ok=True)
                    
                    sub_video_path = os.path.join(chunk_output_dir, f"range_{range_idx}_{start_frame}_{end_frame}.mp4")
                    extract_chunk_frame_range(
                        input_path, 
                        sub_video_path, 
                        start_frame, 
                        end_frame, 
                        start_x, 
                        end_x,
                        start_y,
                        end_y,
                        fps
                    )
                    
                    # Create mask for this range
                    if processor_model_type.lower() == "propainter":
                        # ProPainter needs a mask sequence
                        from mask.mask_analysis import create_mask_sequence_for_chunk
                        range_mask_dir = os.path.join(chunk_output_dir, f"range_{range_idx}_masks")
                        range_mask_path = create_mask_sequence_for_chunk(
                            full_res_mask_dir,
                            start_frame,
                            end_frame,
                            start_x,
                            end_x,
                            start_y,
                            end_y,
                            range_mask_dir,
                            original_mask=np.array(original_mask)
                        )
                    else:
                        # MatAnyone uses a single mask  
                        range_mask_path = create_optimal_mask_for_range(
                            original_mask, 
                            full_res_mask_dir,
                            start_frame,  # Use first frame of range as mask
                            start_x,
                            end_x,
                            start_y,
                            end_y,
                            os.path.join(chunk_output_dir, f"range_{range_idx}_{start_frame}_{end_frame}_mask.png")
                        )
                    
                    # Get a processor from cache or create new one
                    from core.inference_core import InterruptibleInferenceCore
                    range_processor = get_cached_processor(processor_model_path, processor_model_type)
                    
                    # Process this range directly
                    range_kwargs = kwargs.copy()
                    
                    # For ProPainter, process at low resolution (same as mask resolution)
                    if processor_model_type.lower() == "propainter":
                        range_kwargs['max_size'] = max(low_res_width, low_res_height)
                        range_kwargs['low_res_width'] = low_res_width
                        range_kwargs['low_res_height'] = low_res_height
                        print(f"Processing ProPainter chunk at {low_res_width}x{low_res_height} resolution (same as low-res mask)")
                    else:
                        range_kwargs['max_size'] = kwargs.get('max_size', 1024)
                    
                    range_fgr, range_pha = range_processor.process_video(
                        input_path=sub_video_path,
                        mask_path=range_mask_path,
                        output_path=range_output_dir,
                        **range_kwargs
                    )
                    
                    # Add to range results
                    range_results.append({
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'fgr_path': range_fgr,
                        'pha_path': range_pha
                    })
                    
                    # Clear memory
                    clear_gpu_memory(range_processor, force_full_cleanup=True)
                    
                    continue
                
                # For longer ranges, ALWAYS process bidirectionally with the keyframe
                # 1. Process from keyframe to end_frame in forward direction
                # 2. Process from keyframe to start_frame in backward direction
                
                # Create output directory for this range
                range_output_dir = os.path.join(chunk_output_dir, f"range_{range_idx}_output")
                os.makedirs(range_output_dir, exist_ok=True)
                
                # Declare variables in outer scope to avoid reference issues
                forward_fgr = None
                forward_pha = None
                backward_fgr = None
                backward_pha = None
                forward_output_dir = None
                backward_output_dir = None
                
                # Process forward segment (keyframe to end_frame)
                forward_segment_processed = False
                
                if keyframe < end_frame:
                    # Extract forward segment
                    forward_video_path = os.path.join(chunk_output_dir, f"range_{range_idx}_forward_{keyframe}_{end_frame}.mp4")
                    extract_chunk_frame_range(
                        input_path, 
                        forward_video_path, 
                        keyframe, 
                        end_frame, 
                        start_x, 
                        end_x,
                        start_y,
                        end_y,
                        fps
                    )
                    
                    # Create mask for this chunk
                    if processor_model_type.lower() == "propainter":
                        # ProPainter needs a mask sequence
                        from mask.mask_analysis import create_mask_sequence_for_chunk
                        forward_mask_dir = os.path.join(chunk_output_dir, f"range_{range_idx}_forward_masks")
                        forward_mask_path = create_mask_sequence_for_chunk(
                            full_res_mask_dir,
                            keyframe,  # start_frame
                            end_frame,  # end_frame
                            start_x,
                            end_x,
                            start_y,
                            end_y,
                            forward_mask_dir,
                            original_mask=np.array(original_mask)
                        )
                    else:
                        # MatAnyone uses a single mask
                        forward_mask_path = create_optimal_mask_for_range(
                            original_mask, 
                            full_res_mask_dir,
                            keyframe,  # Use keyframe as mask for forward pass
                            start_x,
                            end_x,
                            start_y,
                            end_y,
                            os.path.join(chunk_output_dir, f"range_{range_idx}_forward_mask_{keyframe}.png")
                        )
                    
                    # Get a processor from cache or create new one
                    from core.inference_core import InterruptibleInferenceCore
                    forward_processor = get_cached_processor(processor_model_path, processor_model_type)
                    
                    forward_kwargs = kwargs.copy()
                    forward_kwargs['suffix'] = f'range_{range_idx}_forward'
                    
                    # For ProPainter, process at low resolution (same as mask resolution)
                    if processor_model_type.lower() == "propainter":
                        # Process at the same resolution as the low-res mask
                        forward_kwargs['max_size'] = max(low_res_width, low_res_height)
                        print(f"Processing ProPainter chunk at {low_res_width}x{low_res_height} resolution (same as low-res mask)")
                        # Add low_res_width and low_res_height to kwargs for ProPainter
                        forward_kwargs['low_res_width'] = low_res_width
                        forward_kwargs['low_res_height'] = low_res_height
                    else:
                        forward_kwargs['max_size'] = kwargs.get('max_size', 1024)
                    
                    forward_output_dir = os.path.join(range_output_dir, "forward")
                    os.makedirs(forward_output_dir, exist_ok=True)
                    
                    # Process with the keyframe mask from full_res_mask_dir
                    forward_fgr, forward_pha = forward_processor.process_video(
                        input_path=forward_video_path,
                        mask_path=forward_mask_path,
                        output_path=forward_output_dir,
                        **forward_kwargs
                    )
                    
                    forward_segment_processed = True
                    
                    # Clear memory
                    clear_gpu_memory(forward_processor, force_full_cleanup=True)
                
                # Process backward segment (keyframe to start_frame)
                backward_segment_processed = False
                
                if keyframe > start_frame:
                    # Extract backward segment
                    backward_video_path = os.path.join(chunk_output_dir, f"range_{range_idx}_backward_{keyframe}_{start_frame}.mp4")
                    
                    # Extract frames in reverse order
                    backwards_segment = extract_chunk_frame_range_reversed(
                        input_path, 
                        backward_video_path, 
                        start_frame, 
                        keyframe, 
                        start_x, 
                        end_x,
                        start_y,
                        end_y,
                        fps
                    )
                    
                    # Create mask for backward segment
                    backward_first_frame = keyframe - 1
                    if processor_model_type.lower() == "propainter":
                        # ProPainter needs a mask sequence (reversed)
                        from mask.mask_analysis import create_mask_sequence_for_chunk
                        backward_mask_dir = os.path.join(chunk_output_dir, f"range_{range_idx}_backward_masks")
                        
                        # Create mask sequence for backward range
                        backward_mask_path = create_mask_sequence_for_chunk(
                            full_res_mask_dir,
                            start_frame,  # start_frame
                            backward_first_frame,  # end_frame (keyframe-1)
                            start_x,
                            end_x,
                            start_y,
                            end_y,
                            backward_mask_dir,
                            original_mask=np.array(original_mask)
                        )
                        
                        # Reverse the mask sequence to match reversed video
                        from utils.video_utils import reverse_image_sequence
                        reverse_image_sequence(backward_mask_dir, backward_mask_dir + "_reversed")
                        backward_mask_path = backward_mask_dir + "_reversed"
                    else:
                        # MatAnyone uses a single mask
                        backward_mask_path = create_optimal_mask_for_range(
                            original_mask, 
                            full_res_mask_dir,
                            backward_first_frame,  # Use the actual first frame of the reversed segment
                            start_x,
                            end_x,
                            start_y,
                            end_y,
                            os.path.join(chunk_output_dir, f"range_{range_idx}_backward_mask_{backward_first_frame}.png")
                        )
                    
                    # Get a processor from cache or create new one
                    from core.inference_core import InterruptibleInferenceCore
                    backward_processor = get_cached_processor(processor_model_path, processor_model_type)
                    
                    backward_kwargs = kwargs.copy()
                    backward_kwargs['suffix'] = f'range_{range_idx}_backward'
                    
                    # For ProPainter, process at low resolution (same as mask resolution)
                    if processor_model_type.lower() == "propainter":
                        backward_kwargs['max_size'] = max(low_res_width, low_res_height)
                        backward_kwargs['low_res_width'] = low_res_width
                        backward_kwargs['low_res_height'] = low_res_height
                    else:
                        backward_kwargs['max_size'] = kwargs.get('max_size', 1024)
                    
                    backward_output_dir = os.path.join(range_output_dir, "backward")
                    os.makedirs(backward_output_dir, exist_ok=True)
                    
                    # Process with the mask from full_res_mask_dir
                    backward_fgr, backward_pha = backward_processor.process_video(
                        input_path=backward_video_path,
                        mask_path=backward_mask_path,
                        output_path=backward_output_dir,
                        **backward_kwargs
                    )
                    
                    # Re-reverse the backward segment using VideoProcessor
                    backward_fgr_reversed = os.path.join(backward_output_dir, f"backward_fgr_reversed.mp4")
                    backward_pha_reversed = os.path.join(backward_output_dir, f"backward_pha_reversed.mp4")
                    
                    chunk_video_processor.reverse_video(backward_fgr, backward_fgr_reversed)
                    chunk_video_processor.reverse_video(backward_pha, backward_pha_reversed)
                    
                    backward_segment_processed = True
                    
                    # Clear memory
                    clear_gpu_memory(backward_processor, force_full_cleanup=True)
                    
                    # Update backward paths to the re-reversed versions
                    backward_fgr = backward_fgr_reversed
                    backward_pha = backward_pha_reversed
                
                # Special case: if keyframe is the first or last frame, we need to ensure bidirectional processing
                if keyframe == start_frame and not forward_segment_processed:
                    # Keyframe is the first frame, we only processed backward - need to also process forward
                    # This is a special case where we process the same range in the opposite direction
                    print(f"Keyframe is the first frame - ensuring bidirectional processing by adding forward pass")
                    
                    # Extract forward segment (same as backward but in forward order)
                    forward_video_path = os.path.join(chunk_output_dir, f"range_{range_idx}_forward_{start_frame}_{end_frame}.mp4")
                    extract_chunk_frame_range(
                        input_path, 
                        forward_video_path, 
                        start_frame, 
                        end_frame, 
                        start_x, 
                        end_x,
                        start_y,
                        end_y,
                        fps
                    )
                    
                    # Create mask from full_res_mask_dir using the start frame (first frame of forward)
                    forward_mask_path = create_optimal_mask_for_range(
                        original_mask, 
                        full_res_mask_dir,
                        start_frame,  # Use start frame for this special case
                        start_x,
                        end_x,
                        start_y,
                        end_y,
                        os.path.join(chunk_output_dir, f"range_{range_idx}_forward_mask_{start_frame}.png")
                    )
                    
                    # Get a processor from cache
                    from core.inference_core import InterruptibleInferenceCore
                    forward_processor = get_cached_processor(processor_model_path, processor_model_type)
                    
                    forward_kwargs = kwargs.copy()
                    forward_kwargs['suffix'] = f'range_{range_idx}_forward'
                    forward_kwargs['max_size'] = kwargs.get('max_size', 1024)
                    
                    forward_output_dir = os.path.join(range_output_dir, "forward")
                    os.makedirs(forward_output_dir, exist_ok=True)
                    
                    # Process with the first frame mask
                    forward_fgr, forward_pha = forward_processor.process_video(
                        input_path=forward_video_path,
                        mask_path=forward_mask_path,
                        output_path=forward_output_dir,
                        **forward_kwargs
                    )
                    
                    forward_segment_processed = True
                    
                    # Clear memory
                    clear_gpu_memory(forward_processor, force_full_cleanup=True)
                
                elif keyframe == end_frame and not backward_segment_processed:
                    # Keyframe is the last frame, we only processed forward - need to also process backward
                    # This is a special case where we process the same range in the opposite direction
                    print(f"Keyframe is the last frame - ensuring bidirectional processing by adding backward pass")
                    
                    # Create reversed version of the segment
                    backward_video_path = os.path.join(chunk_output_dir, f"range_{range_idx}_backward_{end_frame}_{start_frame}.mp4")
                    
                    # Extract frames in reverse order (starting from the end frame)
                    backwards_segment = extract_chunk_frame_range_reversed(
                        input_path, 
                        backward_video_path, 
                        start_frame, 
                        end_frame, 
                        start_x, 
                        end_x,
                        start_y,
                        end_y,
                        fps
                    )
                    
                    # Create mask from full_res_mask_dir using the end frame (last frame of range, first of reverse)
                    backward_mask_path = create_optimal_mask_for_range(
                        original_mask, 
                        full_res_mask_dir,
                        end_frame,  # Use end frame for this special case
                        start_x,
                        end_x,
                        start_y,
                        end_y,
                        os.path.join(chunk_output_dir, f"range_{range_idx}_backward_mask_{end_frame}.png")
                    )
                    
                    # Get a processor from cache
                    from core.inference_core import InterruptibleInferenceCore
                    backward_processor = get_cached_processor(processor_model_path, processor_model_type)
                    
                    backward_kwargs = kwargs.copy()
                    backward_kwargs['suffix'] = f'range_{range_idx}_backward'
                    backward_kwargs['max_size'] = kwargs.get('max_size', 1024)
                    
                    backward_output_dir = os.path.join(range_output_dir, "backward")
                    os.makedirs(backward_output_dir, exist_ok=True)
                    
                    # Process with the end frame mask
                    backward_fgr, backward_pha = backward_processor.process_video(
                        input_path=backward_video_path,
                        mask_path=backward_mask_path,
                        output_path=backward_output_dir,
                        **backward_kwargs
                    )
                    
                    # Re-reverse the backward segment using VideoProcessor
                    backward_fgr_reversed = os.path.join(backward_output_dir, f"backward_fgr_reversed.mp4")
                    backward_pha_reversed = os.path.join(backward_output_dir, f"backward_pha_reversed.mp4")
                    
                    chunk_video_processor.reverse_video(backward_fgr, backward_fgr_reversed)
                    chunk_video_processor.reverse_video(backward_pha, backward_pha_reversed)
                    
                    backward_segment_processed = True
                    
                    # Clear memory
                    clear_gpu_memory(backward_processor, force_full_cleanup=True)
                    
                    # Update backward paths to the re-reversed versions
                    backward_fgr = backward_fgr_reversed
                    backward_pha = backward_pha_reversed
                
                # Now combine the forward and backward segments
                combined_fgr = os.path.join(range_output_dir, f"range_{range_idx}_combined_fgr.mp4")
                combined_pha = os.path.join(range_output_dir, f"range_{range_idx}_combined_pha.mp4")
                
                # Prepare segment paths based on what we processed
                fgr_segments = []
                pha_segments = []
                
                # Only include valid segments
                if backward_fgr is not None and os.path.exists(backward_fgr):
                    fgr_segments.append(backward_fgr)
                    pha_segments.append(backward_pha)
                
                if forward_fgr is not None and os.path.exists(forward_fgr):
                    fgr_segments.append(forward_fgr)
                    pha_segments.append(forward_pha)
                
                # Combine segments if we have multiple
                if len(fgr_segments) > 1:
                    concatenate_videos(fgr_segments, combined_fgr)
                    concatenate_videos(pha_segments, combined_pha)
                elif len(fgr_segments) == 1:
                    # Just use the single segment
                    combined_fgr = fgr_segments[0]
                    combined_pha = pha_segments[0]
                else:
                    # No segments, create empty videos
                    print(f"Warning: No segments were processed for range {range_idx}. Creating empty outputs.")
                    chunk_width = end_x - start_x
                    chunk_height = end_y - start_y
                    frame_count = end_frame - start_frame + 1
                    
                    # Get a processor to create empty videos
                    chunk_processor = get_cached_processor(processor_model_path, processor_model_type)
                    chunk_processor.create_empty_video(combined_fgr, chunk_width, chunk_height, fps, frame_count, alpha=False)
                    chunk_processor.create_empty_video(combined_pha, chunk_width, chunk_height, fps, frame_count, alpha=True)
                
                # Add to range results
                range_results.append({
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'fgr_path': combined_fgr,
                    'pha_path': combined_pha
                })
                
                # Clear all possible memory
                clear_gpu_memory(None, force_full_cleanup=True)
            
            # After processing all ranges for this chunk, create a full-length output
            # by assembling the range results in order
            if range_results:
                # Create output videos for this chunk
                chunk_fgr = os.path.join(chunk_output_dir, f"chunk_{chunk_idx}_fgr.mp4")
                chunk_pha = os.path.join(chunk_output_dir, f"chunk_{chunk_idx}_pha.mp4")
                
                # Create full-length videos by filling gaps between ranges with empty frames
                chunk_width = end_x - start_x
                chunk_height = end_y - start_y
                
                create_full_video_from_ranges(
                    range_results, 
                    frame_count, 
                    chunk_fgr, 
                    chunk_pha,
                    chunk_width,
                    chunk_height,
                    fps
                )
                
                # Add to chunk outputs
                result = {
                    'fgr_path': chunk_fgr,
                    'pha_path': chunk_pha,
                    'x_range': (start_x, end_x),
                    'y_range': (start_y, end_y),
                    'width': chunk_width,
                    'height': chunk_height
                }
                
                # Clean up chunk video processor
                chunk_video_processor.cleanup()
                
                return result
            else:
                # No valid ranges, create empty output
                print(f"No valid ranges for chunk {chunk_idx+1}/{actual_num_chunks}, creating empty output")
                
                chunk_width = end_x - start_x
                chunk_height = end_y - start_y
                chunk_fgr = os.path.join(chunk_output_dir, f"empty_fgr.mp4")
                chunk_pha = os.path.join(chunk_output_dir, f"empty_pha.mp4")
                
                # Get a processor from cache
                chunk_processor = get_cached_processor(processor_model_path, processor_model_type)
                chunk_processor.create_empty_video(chunk_fgr, chunk_width, chunk_height, fps, frame_count, alpha=False)
                chunk_processor.create_empty_video(chunk_pha, chunk_height, chunk_height, fps, frame_count, alpha=True)
                
                # Add to chunk outputs
                result = {
                    'fgr_path': chunk_fgr,
                    'pha_path': chunk_pha,
                    'x_range': (start_x, end_x),
                    'y_range': (start_y, end_y),
                    'width': chunk_width,
                    'height': chunk_height
                }
                
                # Clean up chunk video processor
                chunk_video_processor.cleanup()
                
                return result
        
        # Check for interrupt before chunk processing
        check_for_interrupt()
        
        # Process chunks (either in parallel or sequentially)
        if parallel_processing:
            print(f"Processing {actual_num_chunks} chunks in parallel")
            # Use parallel processor
            chunk_outputs = process_chunks_parallel(
                chunk_segments,
                process_chunk_function,
                shared_args,
                max_workers=max_workers,
                use_gpu_lock=True
            )
        else:
            print(f"Processing {actual_num_chunks} chunks sequentially")
            # Process chunks sequentially
            chunk_outputs = []
            last_orientation = None
            
            for chunk_idx, chunk_info in enumerate(chunk_segments):
                # Check if orientation changed
                current_orientation = chunk_info.get('orientation', 'horizontal')
                if use_heat_map_chunking and last_orientation and last_orientation != current_orientation:
                    print(f"Orientation change detected: {last_orientation} -> {current_orientation}")
                    print("Clearing processor cache to handle new dimensions...")
                    # Clear the processor pool to force model reload
                    clear_processor_pool()
                    clear_gpu_memory(processor, force_full_cleanup=True)
                
                last_orientation = current_orientation
                
                print(f"Processing chunk {chunk_idx+1}/{actual_num_chunks}")
                result = process_chunk_function(chunk_info, chunk_idx, shared_args)
                if result:
                    chunk_outputs.append(result)
                # Clear memory for next chunk
                clear_gpu_memory(processor, force_full_cleanup=True)
        
        # Check for interrupt before mask propagation
        check_for_interrupt()
        
        # CRITICAL IMPROVEMENT: Propagate mask data between chunks before reassembly
        # This explicitly shares mask data from overlapping chunks to ensure complete coverage
        print("STEP 5.5: Propagating mask data between overlapping chunks...")
        
        if optimize_masks:
            # Process a sample frame to identify overlapping regions and mask content
            sample_frame = 0
            propagated_data = propagate_mask_data(chunk_outputs, sample_frame, temp_dir)
            
            if propagated_data:
                print("Successfully created propagated mask data. All chunks will share mask content.")
                
                # Show some stats about the propagation
                if 'enhanced_masks' in propagated_data and 'original_masks' in propagated_data:
                    for chunk_idx in propagated_data['enhanced_masks']:
                        if chunk_idx in propagated_data['original_masks']:
                            original = propagated_data['original_masks'][chunk_idx]
                            enhanced = propagated_data['enhanced_masks'][chunk_idx]
                            
                            # Count pixels that were added through propagation
                            diff = np.maximum(0, enhanced - original)
                            pixels_added = np.sum(diff > 0)
                            
                            if pixels_added > 0:
                                print(f"  Chunk {chunk_idx}: Added {pixels_added} pixels from neighboring chunks")
            else:
                print("Warning: Could not propagate mask data between chunks. Will rely on reassembly optimization only.")
        
        # Check for interrupt before reassembly
        check_for_interrupt()
        
        # Step 6: Reassemble chunks into final output
        print("STEP 6: Reassembling chunks into final output")
        
        if not chunk_outputs:
            print("No chunks were processed successfully. Falling back to processing without chunking.")
            # Clean up temp files and memory
            cleanup_temporary_files(temp_files, True)
            video_processor.cleanup()
            clear_mask_cache()
            clear_processor_pool()
            # Try processing without chunking as fallback
            return processor.process_video(input_path=input_path, mask_path=mask_path, 
                                         output_path=output_path, **kwargs)
        
        # Check for interrupt before final output
        check_for_interrupt()
        
        # Create appropriate output filename with bidirectional indication if needed
        chunk_type_suffix = "grid" if chunk_type == "grid" else "chunks"
        if use_autochunk:
            chunk_type_suffix = "auto_" + chunk_type_suffix
            
        if bidirectional:
            final_fgr_path = os.path.join(output_path, f"{video_name}_enhanced_{chunk_type_suffix}_bidirectional_fgr.mp4")
            final_pha_path = os.path.join(output_path, f"{video_name}_enhanced_{chunk_type_suffix}_bidirectional_pha.mp4")
        else:
            final_fgr_path = os.path.join(output_path, f"{video_name}_enhanced_{chunk_type_suffix}_fgr.mp4")
            final_pha_path = os.path.join(output_path, f"{video_name}_enhanced_{chunk_type_suffix}_pha.mp4")
        
        # Indicate whether mask optimization is enabled
        if optimize_masks:
            print("Mask optimization is enabled - maximizing all mask content across chunks")
        
        # Check for interrupt before reassembly process
        check_for_interrupt()
        
        # Reassemble with appropriate method and apply expanded mask if requested
        if use_heat_map_chunking:
            # Use arbitrary chunk reassembly for heat map based placement
            final_fgr_path, final_pha_path = reassemble_arbitrary_chunks(
                chunk_outputs, 
                adjusted_width, 
                adjusted_height, 
                fps, 
                frame_count, 
                final_fgr_path, 
                final_pha_path,
                blend_method,
                temp_dir,  # Pass temp_dir for weight mask debug images
                apply_expanded_mask=apply_expanded_mask,  # Apply expanded mask to final output
                full_res_mask_dir=full_res_mask_dir if apply_expanded_mask else None,  # Use full_res_mask_dir for expanded mask
                maximize_mask=optimize_masks  # Pass the optimize_masks flag
            )
        elif chunk_type == 'strips' or use_autochunk:
            final_fgr_path, final_pha_path = reassemble_strip_chunks(
                chunk_outputs, 
                adjusted_width, 
                adjusted_height, 
                fps, 
                frame_count, 
                final_fgr_path, 
                final_pha_path,
                blend_method,
                temp_dir,  # Pass temp_dir for weight mask debug images
                apply_expanded_mask=apply_expanded_mask,  # Apply expanded mask to final output
                full_res_mask_dir=full_res_mask_dir if apply_expanded_mask else None,  # Use full_res_mask_dir for expanded mask
                maximize_mask=optimize_masks  # Pass the optimize_masks flag
            )
        else:  # grid
            final_fgr_path, final_pha_path = reassemble_grid_chunks(
                chunk_outputs, 
                adjusted_width, 
                adjusted_height, 
                fps, 
                frame_count, 
                final_fgr_path, 
                final_pha_path,
                blend_method,
                temp_dir,  # Pass temp_dir for weight mask debug images
                apply_expanded_mask=apply_expanded_mask,  # Apply expanded mask to final output
                full_res_mask_dir=full_res_mask_dir if apply_expanded_mask else None,  # Use full_res_mask_dir for expanded mask
                maximize_mask=optimize_masks  # Pass the optimize_masks flag
            )
        
        print("Reassembly complete!")
        
        # Clean up temporary files if enabled
        if cleanup_temp:
            cleanup_temporary_files(temp_files, True)
        
        # Clean up all caches and memory
        video_processor.cleanup()
        clear_mask_cache()
        clear_processor_pool()
        
        # Print memory statistics
        print_memory_stats()
        
        # Return paths to the output videos
        return final_fgr_path, final_pha_path
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        print("Cleaning up temporary files and resources after interruption...")
        
        # Always clean up temporary files on interruption
        cleanup_temporary_files(temp_files, cleanup_temp)
        if 'video_processor' in locals():
            video_processor.cleanup()
        clear_mask_cache()
        clear_processor_pool()
        
        # Re-raise to propagate the interruption
        raise
        
    except Exception as e:
        print(f"Error during enhanced chunk processing: {str(e)}")
        traceback.print_exc()
        print("Falling back to standard chunk processing")
        
        # Clean up temporary files and memory
        cleanup_temporary_files(temp_files, True)
        if 'video_processor' in locals():
            video_processor.cleanup()
        clear_mask_cache()
        clear_processor_pool()
        
        # Fall back to standard chunking
        from core.chunk_processor import process_video_in_chunks
        return process_video_in_chunks(processor, input_path, mask_path, output_path, forced_num_chunks,
                                     bidirectional, blend_method, reverse_dilate, cleanup_temp, 
                                     mask_skip_threshold, allow_native_resolution, **kwargs)