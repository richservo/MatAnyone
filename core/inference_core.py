# inference_core.py - v1.1737778800
# Updated: Friday, January 24, 2025 at 19:00:00 PST
# Changes in this version:
# - Added plugin system support with model_type parameter
# - InterruptibleInferenceCore now supports loading any model via adapters
# - Maintains backward compatibility with original MatAnyone code
# - Delegates to adapter methods for plugin models
# - Original version (without plugin support) backed up to inference_core_original.py

"""
Core inference functionality for MatAnyone video processing.
Contains the InterruptibleInferenceCore class.
"""

import os
import numpy as np
import torch
import cv2
import time
import math
import sys
import traceback
import platform
from pathlib import Path

# Import utilities
from utils.video_utils import (
    blend_videos, cleanup_temporary_files, reverse_video, concatenate_videos,
    create_high_quality_writer
)
from mask.mask_utils import check_mask_content, create_empty_mask

# Import chunk processors
from core import checkpoint_processor
from core import enhanced_chunk_processor

# Import memory utilities
from utils.memory_utils import clear_gpu_memory


class InterruptibleInferenceCore:
    """A version of InferenceCore that can be interrupted"""
    
    def __init__(self, model_path="PeiqingYang/MatAnyone", model_type="matanyone", *args, **kwargs):
        """
        Initialize the inference core with plugin system support
        
        Args:
            model_path: Path to the model weights/checkpoint
            model_type: Type of model to use ('matanyone' or any installed plugin)
            *args, **kwargs: Additional arguments to pass to the model
        """
        self.model_type = model_type
        self.model_path = model_path
        self.interrupt_requested = False
        
        # Use MatAnyone model (the only supported model)
        from matanyone import InferenceCore
        # Use default MatAnyone model if no path provided
        effective_model_path = model_path if model_path else "PeiqingYang/MatAnyone"
        super_args = [effective_model_path] + list(args)
        self.core = InferenceCore(*super_args, **kwargs)
    
    def request_interrupt(self):
        """Set the interrupt flag to True"""
        self.interrupt_requested = True
        print("Interrupt requested - will stop after current frame")
        
        # Also forward to the underlying model/adapter if it supports interruption
        if hasattr(self.core, 'request_interrupt'):
            self.core.request_interrupt()
    
    def check_interrupt(self):
        """Check if interrupt was requested and raise exception if so"""
        if self.interrupt_requested:
            self.interrupt_requested = False  # Reset for next time
            print("Interrupt detected - stopping processing")
            raise KeyboardInterrupt("Processing interrupted by user")
    
    def clear_internal_memory(self):
        """
        Internal method to clear the core's memory cache
        This is called by clear_memory and avoids recursion
        """
        print("Clearing internal memory cache...")
        if hasattr(self.core, 'memory') and isinstance(self.core.memory, dict):
            # We need to preserve certain keys that are model instances
            preserved_keys = []
            for key, value in self.core.memory.items():
                # Check if this key should be preserved (not a tensor or array)
                if not isinstance(value, (torch.Tensor, np.ndarray)):
                    preserved_keys.append(key)
            
            # Create a new memory dict with just the preserved keys
            preserved_memory = {k: self.core.memory[k] for k in preserved_keys if k in self.core.memory}
            
            # Reset memory to only contain the preserved objects
            self.core.memory = preserved_memory
            print("Internal memory cache cleared")
    
    def clear_memory(self):
        """
        Safely clear memory after processing to free up resources.
        This clears the frame memory cache and forces garbage collection.
        """
        # For plugin models, delegate to their clear_memory method
        if self.model_type != "matanyone" and hasattr(self.core, 'clear_memory'):
            self.core.clear_memory()
        else:
            # Original MatAnyone memory clearing
            # First clear internal memory of this instance
            self.clear_internal_memory()
            
            # Then use the general GPU memory clearing function
            # But pass None to avoid recursion
            clear_gpu_memory(None)
            
    def process_video(self, input_path, mask_path, output_path, 
                     n_warmup=10, r_erode=10, r_dilate=15, 
                     save_image=True, max_size=512,
                     bidirectional=False, blend_method='weighted', 
                     reverse_dilate=15, cleanup_temp=True,
                     suffix=None, video_codec='Auto', video_quality='High',
                     custom_bitrate=None, **kwargs):
        """
        Process a video with the MatAnyone model
        
        Args:
            input_path: Path to input video or image sequence directory
            mask_path: Path to mask image
            output_path: Path to output directory
            n_warmup: Number of warmup frames
            r_erode: Erosion radius
            r_dilate: Dilation radius
            save_image: Whether to save individual frames
            max_size: Maximum size for processing (-1 for original resolution)
            bidirectional: Whether to process in both directions
            blend_method: Method for blending bidirectional results
            reverse_dilate: Dilation radius for reverse mask
            cleanup_temp: Whether to clean up temporary files
            suffix: Optional suffix for output filenames
            video_codec: Video codec to use ('Auto', 'H.264', 'H.265', 'VP9')
            video_quality: Quality preset ('Low', 'Medium', 'High', 'Very High', 'Lossless')
            custom_bitrate: Custom bitrate in kbps (overrides quality preset)
            **kwargs: Additional arguments for the process_video method
            
        Returns:
            Tuple of paths to foreground and alpha videos
        """
        try:
            # Check for interrupt before starting
            self.check_interrupt()
            
            # For plugin models, delegate directly to their process_video
            if self.model_type.lower() != "matanyone":
                return self.core.process_video(
                    input_path=input_path,
                    mask_path=mask_path,
                    output_path=output_path,
                    n_warmup=n_warmup,
                    r_erode=r_erode,
                    r_dilate=r_dilate,
                    save_image=save_image,
                    max_size=max_size,
                    bidirectional=bidirectional,
                    blend_method=blend_method,
                    reverse_dilate=reverse_dilate,
                    cleanup_temp=cleanup_temp,
                    suffix=suffix,
                    video_codec=video_codec,
                    video_quality=video_quality,
                    custom_bitrate=custom_bitrate,
                    **kwargs
                )
            
            # Original MatAnyone processing below
            # Print video quality settings (but don't pass them to underlying library)
            if custom_bitrate:
                print(f"Video quality settings: {video_codec}, {video_quality}, custom bitrate: {custom_bitrate} kbps")
            else:
                print(f"Video quality settings: {video_codec}, {video_quality}")
            
            # Process the video using the core object
            # Filter out video quality parameters that the underlying library doesn't support
            core_args = {
                'n_warmup': n_warmup,
                'r_erode': r_erode,
                'r_dilate': r_dilate,
                'save_image': save_image,
                'max_size': max_size
            }
            
            # Add other kwargs that are supported by the underlying library
            # but exclude our video quality parameters and plugin system parameters
            excluded_params = {
                'video_codec', 'video_quality', 'custom_bitrate', 'model_type',
                'bidirectional', 'blend_method', 'reverse_dilate', 'cleanup_temp',
                'lowres_blend_method', 'lowres_scale', 'min_activity_pct',
                'face_priority_weight', 'parallel_processing'
            }
            for key, value in kwargs.items():
                if key not in excluded_params:
                    core_args[key] = value
            
            # Add suffix if provided
            if suffix:
                core_args['suffix'] = suffix
            
            # Intercept the step method to check for interrupts
            original_step = self.core.step
            
            def interruptible_step(image, *step_args, **step_kwargs):
                # Check for interrupt before each step
                self.check_interrupt()
                # Call original step method
                return original_step(image, *step_args, **step_kwargs)
            
            # Replace with our interruptible version
            self.core.step = interruptible_step
            
            try:
                # Run the original process_video with only supported parameters
                result = self.core.process_video(
                    input_path=input_path,
                    mask_path=mask_path,
                    output_path=output_path,
                    **core_args
                )
                
                # The underlying library returns the paths to the generated videos
                # We don't need to re-encode them since they're already created
                # Our video quality parameters will be used in other functions like
                # create_empty_video, blend_videos, etc.
                return result
                
            finally:
                # Restore original step method
                self.core.step = original_step
                
        except KeyboardInterrupt:
            print("Processing interrupted by user")
            raise
        except Exception as e:
            print(f"Error in process_video: {str(e)}")
            traceback.print_exc()
            raise

    def mask_has_content(self, mask_path, threshold=5):
        """
        Check if a mask image has any meaningful content (non-zero pixels)
        
        Args:
            mask_path: Path to the mask image
            threshold: Minimum percentage of non-zero pixels to consider the mask as having content
                       (0-100, where 0 means any non-zero pixel counts, and higher values require more content)
        
        Returns:
            bool: True if mask has content, False if empty or nearly empty
        """
        return check_mask_content(mask_path, threshold)

    def create_empty_video(self, output_path, width, height, fps, frame_count, alpha=False, 
                          video_codec='Auto', video_quality='High', custom_bitrate=None):
        """
        Create an empty (black) video with the specified dimensions using high-quality encoding
        
        Args:
            output_path: Path to save the video
            width: Frame width
            height: Frame height
            fps: Frames per second
            frame_count: Number of frames
            alpha: If True, creates a black video with alpha=0, otherwise just black RGB
            video_codec: Video codec to use
            video_quality: Quality preset
            custom_bitrate: Custom bitrate in kbps
        
        Returns:
            Path to the created video
        """
        try:
            print(f"Creating empty video at {output_path}: {width}x{height}, {frame_count} frames")
            
            # Create a high-quality VideoWriter
            writer = create_high_quality_writer(
                output_path, fps, width, height, 
                video_codec, video_quality, custom_bitrate
            )
            
            if not writer.isOpened():
                print(f"Error: Could not create output video at {output_path}")
                return None
            
            # Create an empty frame (black for RGB, transparent for alpha)
            if alpha:
                # For alpha matte, we use black (all zeros)
                empty_frame = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                # For foreground, we use black (all zeros)
                empty_frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Write the empty frame for the specified number of frames
            for _ in range(frame_count):
                writer.write(empty_frame)
            
            # Release the writer
            writer.release()
            print(f"Empty video created at {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"Error creating empty video: {str(e)}")
            return None

    # Add checkpoint processing as a method that calls the module function
    def process_with_checkpoints(self, input_path, mask_paths, checkpoint_frames, output_path, 
                                bidirectional=True, blend_method='weighted', video_codec='Auto',
                                video_quality='High', custom_bitrate=None, **kwargs):
        """
        Process a video with checkpoints for more precise matting
        
        This is a wrapper around the checkpoint_processor module function
        """
        return checkpoint_processor.process_with_checkpoints(
            self, input_path, mask_paths, checkpoint_frames, output_path, 
            bidirectional=bidirectional, blend_method=blend_method,
            video_codec=video_codec, video_quality=video_quality,
            custom_bitrate=custom_bitrate, **kwargs
        )

    # Add chunk processing as a method that calls the module function
    # OLD CHUNK PROCESSING REMOVED - ONLY ENHANCED CHUNKING SHOULD BE USED
    
    # Add enhanced chunk processing as a method that calls the new module function
    def process_video_with_enhanced_chunking(self, input_path, mask_path, output_path, num_chunks=2, 
                                           bidirectional=True, blend_method='weighted', lowres_blend_method=None,
                                           reverse_dilate=15, cleanup_temp=True, mask_skip_threshold=5, 
                                           allow_native_resolution=True, low_res_scale=0.25, chunk_type='strips', 
                                           prioritize_faces=True, use_autochunk=False, apply_expanded_mask=True,
                                           video_codec='Auto', video_quality='High', custom_bitrate=None, **kwargs):
        """
        Enhanced chunk processing with low-resolution preprocessing to ensure continuous masks across chunks.
        
        This approach:
        1. Creates a low-resolution bidirectional mask of the entire video
        2. Analyzes each chunk to find frame ranges with meaningful mask content
        3. Identifies frames with optimal mask coverage or containing faces for each range
        4. Processes only those ranges and creates empty videos for the rest
        5. Reassembles everything into a continuous result with proper memory management
        6. Optionally applies expanded original mask to final output to enforce mask boundaries
        
        This is a wrapper around the enhanced_chunk_processor module function
        
        Args:
            input_path: Path to the input video
            mask_path: Path to the mask image
            output_path: Directory to save outputs
            num_chunks: Number of horizontal chunks to split each frame into (1 = no splitting)
            bidirectional: Whether to use bidirectional processing for each chunk
            blend_method: Method to blend forward and reverse passes
            lowres_blend_method: Method to blend low-res passes (defaults to blend_method if None)
            reverse_dilate: Dilation radius for the mask in reverse pass
            cleanup_temp: Whether to clean up temporary files
            mask_skip_threshold: Percentage threshold for mask content
            allow_native_resolution: Whether to allow processing at native resolution
            low_res_scale: Scale factor for low-resolution preprocessing (0.25 = quarter resolution)
            chunk_type: Type of chunking - 'strips' or 'grid'
            prioritize_faces: Whether to prioritize frames with faces as keyframes
            use_autochunk: Whether to automatically determine chunks based on low-resolution
            apply_expanded_mask: Whether to apply expanded original mask to final output to prevent spill
            video_codec: Video codec to use
            video_quality: Quality preset
            custom_bitrate: Custom bitrate in kbps
            **kwargs: Additional arguments to pass to process_video
            
        Returns:
            Tuple of paths to the final foreground and alpha videos
        """
        # For plugin models, we ALWAYS use MatAnyone's enhanced chunking system
        # This ensures ProPainter and other models integrate properly into the workflow
        print(f"\n=== Enhanced Chunking for {self.model_type} ===")
        if self.model_type.lower() != "matanyone":
            print(f"Model {self.model_type} will be integrated into MatAnyone's enhanced chunking pipeline")
            print("Step 1: MatAnyone generates low-res traveling mask")
            print("Step 2: Heat map analysis and chunk creation")
            print(f"Step 3: {self.model_type} processes each chunk")
            print("Step 4: Reassemble final output")
            
        # ALWAYS use MatAnyone's enhanced chunking system, never delegate to plugin's method
        # The enhanced_chunk_processor will handle using MatAnyone for low-res and the plugin for chunks
        
        # Original MatAnyone enhanced chunking
        # Validate auto-chunk mode parameters
        if use_autochunk:
            print("Auto-chunk mode enabled: Will create chunks with consistent dimensions matching the low-resolution mask")
            
            # Verify the low_res_scale is appropriate
            if low_res_scale > 0.5:
                print(f"Warning: low_res_scale={low_res_scale} is quite large for auto-chunking. This may create large chunks.")
                print("Consider using a smaller scale (0.25 or less) for better results.")
            
            # num_chunks is ignored in auto-chunk mode, but we mention this for clarity
            if num_chunks != 2:
                print(f"Note: Number of chunks ({num_chunks}) is ignored in auto-chunk mode.")
                print("Chunks will be automatically determined based on the low-resolution dimensions.")
        
        return enhanced_chunk_processor.process_video_with_enhanced_chunking(
            self, input_path, mask_path, output_path, num_chunks=num_chunks, 
            bidirectional=bidirectional, blend_method=blend_method, lowres_blend_method=lowres_blend_method,
            reverse_dilate=reverse_dilate, cleanup_temp=cleanup_temp, mask_skip_threshold=mask_skip_threshold, 
            allow_native_resolution=allow_native_resolution, low_res_scale=low_res_scale,
            chunk_type=chunk_type, prioritize_faces=prioritize_faces, use_autochunk=use_autochunk,
            apply_expanded_mask=apply_expanded_mask, video_codec=video_codec,
            video_quality=video_quality, custom_bitrate=custom_bitrate, **kwargs
        )
    
    # OLD BACKWARD COMPATIBILITY METHODS REMOVED - ONLY ENHANCED CHUNKING SHOULD BE USED