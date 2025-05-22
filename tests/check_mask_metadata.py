#!/usr/bin/env python3
"""
Quick test script to check mask metadata
"""
import sys
import os
sys.path.append('/Volumes/Storage/Richard/MatAnyone')

from mask.mask_utils import get_keyframe_metadata_from_mask

# Check the uploaded mask image
mask_path = "[Image #1]"  # This is the path provided

print(f"Checking mask metadata for: {mask_path}")

try:
    metadata = get_keyframe_metadata_from_mask(mask_path)
    print(f"Keyframe metadata result: {metadata}")
    
    if metadata is not None:
        print(f"✓ FOUND keyframe metadata: frame {metadata}")
    else:
        print("✗ NO keyframe metadata found")
        
except Exception as e:
    print(f"Error reading metadata: {e}")