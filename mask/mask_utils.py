"""
# mask_utils.py - v1.1684356848
# Updated: Wednesday, May 15, 2025
Mask utility functions for MatAnyone video processing.
Contains functions for generating and working with masks.
"""

import os
import cv2
import numpy as np
import traceback
from PIL import Image
from PIL.PngImagePlugin import PngInfo


def generate_mask_for_video(video_path, output_mask_path=None, model_type="vit_b"):
    """
    Generate a mask for a video using SAM
    
    Args:
        video_path: Path to the input video
        output_mask_path: Path to save the mask (optional)
        model_type: SAM model type to use (vit_b, vit_l, vit_h)
        
    Returns:
        Path to the generated mask image
    """
    try:
        from mask.sam_generator import SAMMaskGenerator
        
        # Create mask generator
        mask_gen = SAMMaskGenerator(model_type=model_type)
        
        # Extract first frame
        print("Extracting first frame from video...")
        frame = mask_gen.extract_frame(video_path, 0)
        
        # Set default output path if not provided
        if output_mask_path is None:
            video_name = os.path.basename(video_path)
            if video_name.endswith(('.mp4', '.mov', '.avi')):
                video_name = os.path.splitext(video_name)[0]
            
            # Create a mask in the same directory as the video
            output_mask_path = os.path.join(
                os.path.dirname(video_path),
                f"{video_name}_sam_mask.png"
            )
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        
        print(f"Starting interactive SAM interface for mask generation...")
        print(f"This should be launched from the GUI. If you see this message in the console,")
        print(f"you should use the GUI to generate masks interactively.")
        
        return output_mask_path
        
    except ImportError:
        print("Error: mask_generator module not found.")
        print("Please make sure the mask_generator.py file is in the same directory.")
        return None
    except Exception as e:
        print(f"Error generating mask: {str(e)}")
        traceback.print_exc()
        return None


def check_mask_content(mask_path, threshold=5):
    """
    Check if a mask image has any meaningful content (non-zero pixels)
    
    Args:
        mask_path: Path to the mask image
        threshold: Minimum percentage of non-zero pixels to consider the mask as having content
                   (0-100, where 0 means any non-zero pixel counts, and higher values require more content)
    
    Returns:
        bool: True if mask has content, False if empty or nearly empty
    """
    try:
        # Read the mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read mask at {mask_path}")
            return True  # Assume content if we can't read the mask (safer)
        
        # Count non-zero pixels
        non_zero_pixels = cv2.countNonZero(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        
        # Calculate percentage
        non_zero_percentage = (non_zero_pixels / total_pixels) * 100
        
        if non_zero_percentage < threshold:
            print(f"Mask has only {non_zero_percentage:.2f}% non-zero pixels (below {threshold}% threshold)")
            return False
        else:
            print(f"Mask has {non_zero_percentage:.2f}% non-zero pixels")
            return True
            
    except Exception as e:
        print(f"Error checking mask content: {str(e)}")
        return True  # Assume content in case of error (safer)


def create_empty_mask(output_path, width, height, value=0):
    """
    Create an empty (black) mask with the specified dimensions
    
    Args:
        output_path: Path to save the mask
        width: Mask width
        height: Mask height
        value: Pixel value (0=black, 255=white)
        
    Returns:
        Path to the created mask
    """
    try:
        # Create empty mask
        mask = np.full((height, width), value, dtype=np.uint8)
        
        # Save mask
        cv2.imwrite(output_path, mask)
        
        return output_path
    except Exception as e:
        print(f"Error creating empty mask: {str(e)}")
        return None


def add_keyframe_metadata_to_mask(mask_path, keyframe_number, output_path=None):
    """
    Add keyframe metadata to a PNG mask file
    
    Args:
        mask_path: Path to the input mask
        keyframe_number: Frame number to store as keyframe metadata
        output_path: Path to save the mask with metadata (if None, overwrites original)
        
    Returns:
        Path to the mask with metadata
    """
    try:
        if output_path is None:
            output_path = mask_path
            
        # Open the image
        img = Image.open(mask_path)
        
        # Create PNG metadata
        metadata = PngInfo()
        metadata.add_text("keyframe", str(keyframe_number))
        
        # Save with metadata
        img.save(output_path, "PNG", pnginfo=metadata)
        
        print(f"Added keyframe metadata: frame {keyframe_number} to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error adding keyframe metadata: {str(e)}")
        traceback.print_exc()
        return mask_path  # Return original path if failed


def get_keyframe_metadata_from_mask(mask_path):
    """
    Read keyframe metadata from a PNG mask file
    
    Args:
        mask_path: Path to the mask image
        
    Returns:
        int or None: Keyframe number if found, None if no metadata exists
    """
    try:
        # Open the image
        img = Image.open(mask_path)
        
        # Get metadata
        if hasattr(img, 'text') and img.text:
            keyframe_str = img.text.get("keyframe")
            if keyframe_str:
                keyframe_number = int(keyframe_str)
                print(f"Found keyframe metadata: frame {keyframe_number} in {mask_path}")
                return keyframe_number
        
        # No metadata found
        return None
        
    except Exception as e:
        print(f"Error reading keyframe metadata: {str(e)}")
        return None
