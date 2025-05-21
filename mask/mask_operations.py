"""
# mask_operations.py - v1.1734487623
# Updated: Tuesday, May 21, 2025
# Changes in this version:
# - Completely redesigned mask operations to minimize disk I/O
# - Added in-memory cache for frequently used masks
# - Implemented batch processing for multiple mask operations
# - Added vectorized implementations for mask transformations
# - Improved dimension checking and error handling
# - Introduced mask manager for better caching and operations

Mask manipulation operations for MatAnyone video processing.
Contains functions for processing and transforming masks.
"""

import os
import cv2
import numpy as np
import traceback
import hashlib
from typing import Dict, List, Tuple, Union, Optional, Any, Callable


# Global mask cache to reduce disk I/O
_mask_cache = {}
_mask_stats = {"cache_hits": 0, "cache_misses": 0, "disk_reads": 0, "disk_writes": 0}


class MaskManager:
    """
    Manager for mask operations with caching capabilities
    """
    
    def __init__(self, cache_size_mb=500):
        """
        Initialize the mask manager
        
        Args:
            cache_size_mb: Maximum cache size in MB
        """
        self.cache = {}  # Path -> mask mapping
        self.cache_size_mb = cache_size_mb
        self.cache_used_mb = 0
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0, 
            "disk_reads": 0,
            "disk_writes": 0,
            "operations": {}
        }
    
    def get_mask(self, mask_path: str, force_reload: bool = False) -> Optional[np.ndarray]:
        """
        Get a mask from cache or disk
        
        Args:
            mask_path: Path to the mask
            force_reload: Whether to force reloading from disk
            
        Returns:
            Mask as numpy array or None if not found
        """
        # Check cache first
        if not force_reload and mask_path in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[mask_path].copy()  # Return a copy to prevent modifying cached version
        
        # Load from disk
        try:
            self.stats["disk_reads"] += 1
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"Error: Could not read mask: {mask_path}")
                return None
            
            # Cache the mask
            self._add_to_cache(mask_path, mask)
            
            self.stats["cache_misses"] += 1
            return mask
        except Exception as e:
            print(f"Error reading mask {mask_path}: {str(e)}")
            traceback.print_exc()
            return None
    
    def save_mask(self, mask: np.ndarray, output_path: str) -> bool:
        """
        Save a mask to disk and cache
        
        Args:
            mask: Mask to save
            output_path: Path to save the mask
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save to disk
            self.stats["disk_writes"] += 1
            cv2.imwrite(output_path, mask)
            
            # Cache the mask
            self._add_to_cache(output_path, mask)
            
            return True
        except Exception as e:
            print(f"Error saving mask to {output_path}: {str(e)}")
            traceback.print_exc()
            return False
    
    def _add_to_cache(self, mask_path: str, mask: np.ndarray) -> None:
        """
        Add a mask to the cache
        
        Args:
            mask_path: Path to the mask (used as key)
            mask: Mask data
        """
        # Calculate mask size in MB
        mask_size_mb = mask.nbytes / (1024 * 1024)
        
        # Check if we need to free up cache space
        if self.cache_used_mb + mask_size_mb > self.cache_size_mb:
            self._cleanup_cache(needed_mb=mask_size_mb)
        
        # Add to cache
        self.cache[mask_path] = mask.copy()  # Store a copy to prevent external modifications
        self.cache_used_mb += mask_size_mb
    
    def _cleanup_cache(self, needed_mb: float) -> None:
        """
        Free up cache space to accommodate a new mask
        
        Args:
            needed_mb: How much space needed in MB
        """
        # Simple LRU-like cleanup - just remove oldest items
        paths = list(self.cache.keys())
        
        for path in paths:
            if self.cache_used_mb + needed_mb <= self.cache_size_mb:
                break
                
            mask = self.cache.pop(path)
            mask_size_mb = mask.nbytes / (1024 * 1024)
            self.cache_used_mb -= mask_size_mb
    
    def clear_cache(self) -> None:
        """
        Clear the mask cache completely
        """
        self.cache = {}
        self.cache_used_mb = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size_mb": self.cache_size_mb,
            "cache_used_mb": self.cache_used_mb,
            "cache_count": len(self.cache),
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "hit_rate": self.stats["cache_hits"] / (self.stats["cache_hits"] + self.stats["cache_misses"]) if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0 else 0,
            "disk_reads": self.stats["disk_reads"],
            "disk_writes": self.stats["disk_writes"],
            "operations": self.stats["operations"]
        }
    
    def _log_operation(self, operation: str) -> None:
        """
        Log an operation for statistics
        
        Args:
            operation: Name of the operation
        """
        if operation not in self.stats["operations"]:
            self.stats["operations"][operation] = 0
        self.stats["operations"][operation] += 1


# Create a global mask manager instance
mask_manager = MaskManager()


def dilate_mask(mask_path: str, output_path: str, kernel_size: int = 15) -> str:
    """
    Dilate a mask to expand white regions
    
    Args:
        mask_path: Path to input mask
        output_path: Path to save dilated mask
        kernel_size: Size of dilation kernel
        
    Returns:
        Path to dilated mask
    """
    mask_manager._log_operation("dilate")
    
    # Get mask from cache or disk
    mask = mask_manager.get_mask(mask_path)
    if mask is None:
        return None
    
    try:
        # Create kernel for dilation
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Dilate mask
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Save result to disk and cache
        if mask_manager.save_mask(dilated_mask, output_path):
            print(f"Dilated mask: {mask.shape} -> {dilated_mask.shape}, kernel size: {kernel_size}")
            return output_path
        return None
    except Exception as e:
        print(f"Error dilating mask: {str(e)}")
        traceback.print_exc()
        return None


def erode_mask(mask_path: str, output_path: str, kernel_size: int = 15) -> str:
    """
    Erode a mask to shrink white regions
    
    Args:
        mask_path: Path to input mask
        output_path: Path to save eroded mask
        kernel_size: Size of erosion kernel
        
    Returns:
        Path to eroded mask
    """
    mask_manager._log_operation("erode")
    
    # Get mask from cache or disk
    mask = mask_manager.get_mask(mask_path)
    if mask is None:
        return None
    
    try:
        # Create kernel for erosion
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Erode mask
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
        
        # Save result to disk and cache
        if mask_manager.save_mask(eroded_mask, output_path):
            print(f"Eroded mask: {mask.shape} -> {eroded_mask.shape}, kernel size: {kernel_size}")
            return output_path
        return None
    except Exception as e:
        print(f"Error eroding mask: {str(e)}")
        traceback.print_exc()
        return None


def combine_masks(mask_paths: List[str], output_path: str, method: str = 'max') -> str:
    """
    Combine multiple masks into one
    
    Args:
        mask_paths: List of paths to input masks
        output_path: Path to save combined mask
        method: Combination method ('max', 'min', 'average')
        
    Returns:
        Path to combined mask
    """
    mask_manager._log_operation(f"combine_{method}")
    
    try:
        if not mask_paths:
            print("Error: No mask paths provided")
            return None
        
        # Read first mask to get dimensions
        first_mask = mask_manager.get_mask(mask_paths[0])
        if first_mask is None:
            return None
        
        height, width = first_mask.shape
        print(f"Combining masks with dimensions {width}x{height} using {method} method")
        
        # Vectorized implementation for better performance
        masks = []
        for mask_path in mask_paths:
            mask = mask_manager.get_mask(mask_path)
            if mask is None:
                continue
                
            # Resize if needed
            if mask.shape[0] != height or mask.shape[1] != width:
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                
            masks.append(mask)
        
        # Process based on method using vectorized operations
        if method == 'max':
            # Use numpy's maximum function for efficient computation
            combined_mask = np.maximum.reduce(masks) if masks else np.zeros((height, width), dtype=np.uint8)
        elif method == 'min':
            # Use numpy's minimum function for efficient computation
            combined_mask = np.minimum.reduce(masks) if masks else np.ones((height, width), dtype=np.uint8) * 255
        elif method == 'average':
            # Stack and average along new axis
            masks_array = np.stack(masks, axis=0) if masks else np.zeros((1, height, width), dtype=np.float32)
            combined_mask = np.mean(masks_array, axis=0)
            combined_mask = combined_mask.astype(np.uint8)
        else:
            print(f"Invalid combination method: {method}")
            return None
        
        # Ensure binary mask
        _, combined_mask = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Save result
        if mask_manager.save_mask(combined_mask, output_path):
            print(f"Created combined mask at {output_path}")
            return output_path
        return None
    except Exception as e:
        print(f"Error combining masks: {str(e)}")
        traceback.print_exc()
        return None


def threshold_mask(mask_path: str, output_path: str, threshold: int = 127) -> str:
    """
    Apply binary thresholding to a mask
    
    Args:
        mask_path: Path to input mask
        output_path: Path to save thresholded mask
        threshold: Threshold value (0-255)
        
    Returns:
        Path to thresholded mask
    """
    mask_manager._log_operation("threshold")
    
    # Get mask from cache or disk
    mask = mask_manager.get_mask(mask_path)
    if mask is None:
        return None
    
    try:
        # Apply binary thresholding
        _, thresholded_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
        
        # Save result
        if mask_manager.save_mask(thresholded_mask, output_path):
            print(f"Created thresholded mask at {output_path}")
            return output_path
        return None
    except Exception as e:
        print(f"Error thresholding mask: {str(e)}")
        traceback.print_exc()
        return None


def resize_mask_to_dimensions(mask_path: str, output_path: str, width: int, height: int) -> str:
    """
    Resize a mask to specific dimensions
    
    Args:
        mask_path: Path to input mask
        output_path: Path to save resized mask
        width: Target width
        height: Target height
        
    Returns:
        Path to resized mask
    """
    mask_manager._log_operation("resize")
    
    # Get mask from cache or disk
    mask = mask_manager.get_mask(mask_path)
    if mask is None:
        return None
    
    try:
        # Check if resize is needed
        if mask.shape[1] == width and mask.shape[0] == height:
            print(f"Mask already has dimensions {width}x{height}, no resize needed")
            
            # Still save to output path if different from input
            if output_path != mask_path:
                if mask_manager.save_mask(mask, output_path):
                    return output_path
                return None
            return mask_path
        
        # Resize the mask
        resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Ensure binary mask (0 and 255 values only)
        _, resized_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Save result
        if mask_manager.save_mask(resized_mask, output_path):
            print(f"Resized mask from {mask.shape[1]}x{mask.shape[0]} to {width}x{height}")
            return output_path
        return None
    except Exception as e:
        print(f"Error resizing mask: {str(e)}")
        traceback.print_exc()
        return None


def crop_mask_to_chunk(mask_path: str, output_path: str, start_x: int, end_x: int, 
                      start_y: int, end_y: int) -> str:
    """
    Crop a mask to a specific chunk region
    
    Args:
        mask_path: Path to input mask
        output_path: Path to save cropped mask
        start_x: Starting X coordinate
        end_x: Ending X coordinate
        start_y: Starting Y coordinate
        end_y: Ending Y coordinate
        
    Returns:
        Path to cropped mask
    """
    mask_manager._log_operation("crop")
    
    # Get mask from cache or disk
    mask = mask_manager.get_mask(mask_path)
    if mask is None:
        return None
    
    try:
        # Calculate chunk dimensions
        chunk_width = end_x - start_x
        chunk_height = end_y - start_y
        
        # Handle out-of-bounds coordinates
        if start_x >= mask.shape[1] or start_y >= mask.shape[0]:
            print(f"Error: Chunk start coordinates ({start_x}, {start_y}) are outside mask dimensions {mask.shape[1]}x{mask.shape[0]}")
            
            # Create a blank mask of the correct size
            cropped_mask = np.zeros((chunk_height, chunk_width), dtype=np.uint8)
        else:
            # Adjust end coordinates if they exceed mask dimensions
            end_x = min(end_x, mask.shape[1])
            end_y = min(end_y, mask.shape[0])
            
            # If the adjusted dimensions don't match the expected chunk size, we'll need to pad
            actual_width = end_x - start_x
            actual_height = end_y - start_y
            
            if actual_width != chunk_width or actual_height != chunk_height:
                print(f"Warning: Adjusted chunk dimensions ({actual_width}x{actual_height}) don't match expected ({chunk_width}x{chunk_height})")
                
                # Extract the available portion
                partial_mask = mask[start_y:end_y, start_x:end_x]
                
                # Create a blank canvas of the correct size
                cropped_mask = np.zeros((chunk_height, chunk_width), dtype=np.uint8)
                
                # Copy the available portion
                cropped_mask[:min(actual_height, chunk_height), :min(actual_width, chunk_width)] = \
                    partial_mask[:min(actual_height, chunk_height), :min(actual_width, chunk_width)]
            else:
                # Extract the chunk region
                cropped_mask = mask[start_y:end_y, start_x:end_x]
        
        # Save result
        if mask_manager.save_mask(cropped_mask, output_path):
            print(f"Cropped mask to chunk region ({start_x},{start_y})-({end_x},{end_y}), size: {chunk_width}x{chunk_height}")
            return output_path
        return None
    except Exception as e:
        print(f"Error cropping mask to chunk: {str(e)}")
        traceback.print_exc()
        return None


def batch_process_masks(operations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process multiple mask operations in a single batch
    
    Args:
        operations: List of operations, each as a dictionary with:
            - op_type: Operation type (dilate, erode, resize, etc.)
            - input_path: Path to input mask
            - output_path: Path to output mask
            - params: Dictionary of operation-specific parameters
            
    Returns:
        Dictionary mapping operation indices to success status and output paths
    """
    mask_manager._log_operation("batch_process")
    
    results = {}
    
    for i, op in enumerate(operations):
        op_type = op.get('op_type')
        input_path = op.get('input_path')
        output_path = op.get('output_path')
        params = op.get('params', {})
        
        # Validate operation
        if not op_type or not input_path or not output_path:
            results[i] = {"success": False, "error": "Missing required parameters"}
            continue
        
        # Execute operation
        try:
            if op_type == 'dilate':
                kernel_size = params.get('kernel_size', 15)
                result_path = dilate_mask(input_path, output_path, kernel_size)
                results[i] = {"success": result_path is not None, "output_path": result_path}
            
            elif op_type == 'erode':
                kernel_size = params.get('kernel_size', 15)
                result_path = erode_mask(input_path, output_path, kernel_size)
                results[i] = {"success": result_path is not None, "output_path": result_path}
            
            elif op_type == 'threshold':
                threshold = params.get('threshold', 127)
                result_path = threshold_mask(input_path, output_path, threshold)
                results[i] = {"success": result_path is not None, "output_path": result_path}
            
            elif op_type == 'resize':
                width = params.get('width')
                height = params.get('height')
                if not width or not height:
                    results[i] = {"success": False, "error": "Missing width or height for resize"}
                    continue
                result_path = resize_mask_to_dimensions(input_path, output_path, width, height)
                results[i] = {"success": result_path is not None, "output_path": result_path}
            
            elif op_type == 'crop':
                start_x = params.get('start_x')
                end_x = params.get('end_x')
                start_y = params.get('start_y')
                end_y = params.get('end_y')
                if start_x is None or end_x is None or start_y is None or end_y is None:
                    results[i] = {"success": False, "error": "Missing coordinates for crop"}
                    continue
                result_path = crop_mask_to_chunk(input_path, output_path, start_x, end_x, start_y, end_y)
                results[i] = {"success": result_path is not None, "output_path": result_path}
            
            elif op_type == 'combine':
                method = params.get('method', 'max')
                mask_paths = params.get('mask_paths', [])
                if not mask_paths:
                    results[i] = {"success": False, "error": "No mask paths provided for combine"}
                    continue
                result_path = combine_masks(mask_paths, output_path, method)
                results[i] = {"success": result_path is not None, "output_path": result_path}
            
            else:
                results[i] = {"success": False, "error": f"Unknown operation type: {op_type}"}
        
        except Exception as e:
            results[i] = {"success": False, "error": str(e)}
            traceback.print_exc()
    
    return results


def get_mask_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the mask cache
    
    Returns:
        Dictionary with cache statistics
    """
    return mask_manager.get_stats()


def clear_mask_cache() -> None:
    """
    Clear the mask cache to free memory
    """
    mask_manager.clear_cache()
    print("Mask cache cleared")