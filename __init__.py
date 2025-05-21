"""
# __init__.py - v1.1684356848
# Updated: Wednesday, May 15, 2025
MatAnyone Video Processing Package.
Provides comprehensive video matting and processing functionality.
"""

# Import the main components for convenience
from core.inference_core import InterruptibleInferenceCore
from utils.video_utils import blend_videos, cleanup_temporary_files, reverse_video, concatenate_videos
from mask.mask_utils import generate_mask_for_video, check_mask_content, create_empty_mask

# Import utility modules
import chunking_utils
import mask_analysis
import extraction_utils
import reassembly_utils
import memory_utils
import mask_operations

# Import processing modules
import chunk_processor
import checkpoint_processor
import enhanced_chunk_processor

__version__ = "1.0.1"
