"""
# image_processing.py - v1.1684090210
# Updated: Saturday, May 11, 2025
Main image processing module for MatAnyone.
Imports components from other modules to provide the complete functionality.
This is a compatibility layer to ensure existing code continues to work.
"""

# Import components from specialized modules
from core.inference_core import InterruptibleInferenceCore
from utils.video_utils import blend_videos, cleanup_temporary_files, reverse_video, concatenate_videos
from mask.mask_utils import generate_mask_for_video, check_mask_content
from mask import apply_mask_optimization
from core import chunk_processor
from core import checkpoint_processor
from core import enhanced_chunk_processor

# Create an alias for backward compatibility
class ProcessorWrapper(InterruptibleInferenceCore):
    """Wrapper class to maintain backward compatibility"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# Export components for backward compatibility
__all__ = [
    'InterruptibleInferenceCore',
    'ProcessorWrapper',
    'blend_videos',
    'cleanup_temporary_files',
    'reverse_video',
    'concatenate_videos',
    'generate_mask_for_video',
    'check_mask_content'
]
