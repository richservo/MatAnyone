"""
# video_processor.py - v1.1734584254
# Updated: Wednesday, May 22, 2025
# Changes in this version:
# - Created new unified video processing pipeline
# - Implemented caching system for processed videos
# - Added operation chaining for multiple video transformations
# - Reduced disk I/O by keeping videos in memory where possible
# - Added automatic cleanup of temporary files
# - Improved temp directory management to only create when needed

Unified video processing pipeline for MatAnyone.
"""

import os
import cv2
import time
import numpy as np
import tempfile
import shutil
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from pathlib import Path


class VideoProcessor:
    """
    Unified video processing pipeline with caching capabilities
    """
    
    def __init__(self, temp_dir=None, cleanup_temp=True, cache_size_mb=1000, verbose=True):
        """
        Initialize the video processor
        
        Args:
            temp_dir: Directory for temporary files (default: system temp)
            cleanup_temp: Whether to clean up temporary files
            cache_size_mb: Maximum size of in-memory cache in MB
            verbose: Whether to print detailed information
        """
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="matanyone_")
        self.cleanup_temp = cleanup_temp
        self.cache_size_mb = cache_size_mb
        self.verbose = verbose
        self.temp_dir_created = False
        
        # Track temporary files for cleanup
        self.temp_files = []
        
        # Cache for video frames
        self.frame_cache = {}
        
        # Cache for video metadata
        self.metadata_cache = {}
        
        # Statistics
        self.stats = {
            "operations": {},
            "cache_hits": 0,
            "cache_misses": 0,
            "disk_reads": 0,
            "disk_writes": 0,
            "processing_time": 0
        }
        
        if self.verbose:
            print(f"Video processor initialized with temp directory: {self.temp_dir}")
    
    def __del__(self):
        """
        Cleanup on deletion
        """
        self.cleanup()
    
    def cleanup(self):
        """
        Clean up temporary files
        """
        if self.cleanup_temp:
            for path in self.temp_files:
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    elif os.path.exists(path):
                        os.remove(path)
                except Exception as e:
                    if self.verbose:
                        print(f"Error cleaning up {path}: {str(e)}")
            
            # Only try to remove the temp directory if it was actually created
            if self.temp_dir_created:
                try:
                    if os.path.exists(self.temp_dir):
                        shutil.rmtree(self.temp_dir, ignore_errors=True)
                except Exception as e:
                    if self.verbose:
                        print(f"Error cleaning up temp directory {self.temp_dir}: {str(e)}")
            
            self.temp_files = []
            self.temp_dir_created = False
            
            if self.verbose and self.temp_files:
                print("Temporary files cleaned up")
    
    def clear_cache(self):
        """
        Clear in-memory caches
        """
        self.frame_cache = {}
        self.metadata_cache = {}
        
        if self.verbose:
            print("Video cache cleared")
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get information about a video
        
        Args:
            video_path: Path to the video
            
        Returns:
            Dictionary with video information
        """
        # Check cache first
        if video_path in self.metadata_cache:
            self.stats["cache_hits"] += 1
            return self.metadata_cache[video_path].copy()
        
        self.stats["cache_misses"] += 1
        self.stats["disk_reads"] += 1
        
        try:
            # Open the video
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate duration
            duration = frame_count / fps if fps > 0 else 0
            
            # Close the video
            cap.release()
            
            # Create video info
            info = {
                "path": video_path,
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "aspect_ratio": width / height if height > 0 else 0
            }
            
            # Cache the info
            self.metadata_cache[video_path] = info.copy()
            
            return info
        
        except Exception as e:
            print(f"Error getting video info for {video_path}: {str(e)}")
            return {
                "path": video_path,
                "error": str(e)
            }
    
    def extract_frames(self, video_path: str, start_frame: int = 0, 
                      end_frame: Optional[int] = None, step: int = 1) -> List[np.ndarray]:
        """
        Extract frames from a video
        
        Args:
            video_path: Path to the video
            start_frame: Starting frame index
            end_frame: Ending frame index (exclusive, None means all frames)
            step: Frame step (1 means every frame, 2 means every other frame, etc.)
            
        Returns:
            List of frames as numpy arrays
        """
        self._log_operation("extract_frames")
        
        try:
            # Get video info
            info = self.get_video_info(video_path)
            
            # Determine frame range
            frame_count = info["frame_count"]
            
            if end_frame is None:
                end_frame = frame_count
            
            end_frame = min(end_frame, frame_count)
            
            if start_frame >= end_frame:
                print(f"Invalid frame range: {start_frame}-{end_frame}")
                return []
            
            # Create cache key
            cache_key = f"{video_path}:{start_frame}-{end_frame}:{step}"
            
            # Check cache
            if cache_key in self.frame_cache:
                self.stats["cache_hits"] += 1
                return self.frame_cache[cache_key].copy()
            
            self.stats["cache_misses"] += 1
            self.stats["disk_reads"] += 1
            
            # Open the video
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Extract frames
            frames = []
            current_frame = start_frame
            
            if self.verbose:
                print(f"Extracting frames {start_frame}-{end_frame} from {video_path} (step={step})")
            
            while current_frame < end_frame:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if (current_frame - start_frame) % step == 0:
                    frames.append(frame)
                
                current_frame += 1
            
            # Close the video
            cap.release()
            
            # Cache the frames
            self._add_to_frame_cache(cache_key, frames)
            
            if self.verbose:
                print(f"Extracted {len(frames)} frames from {video_path}")
            
            return frames
        
        except Exception as e:
            print(f"Error extracting frames from {video_path}: {str(e)}")
            return []
    
    def extract_frame(self, video_path: str, frame_index: int) -> Optional[np.ndarray]:
        """
        Extract a single frame from a video
        
        Args:
            video_path: Path to the video
            frame_index: Index of the frame to extract
            
        Returns:
            Frame as numpy array or None if not found
        """
        self._log_operation("extract_frame")
        
        # Create cache key
        cache_key = f"{video_path}:{frame_index}"
        
        # Check cache
        if cache_key in self.frame_cache:
            self.stats["cache_hits"] += 1
            return self.frame_cache[cache_key][0].copy()
        
        # Extract frames
        frames = self.extract_frames(video_path, frame_index, frame_index + 1)
        
        if not frames:
            return None
        
        return frames[0]
    
    def create_video(self, frames: List[np.ndarray], output_path: str, fps: float = 30.0, 
                    fourcc: str = 'mp4v') -> str:
        """
        Create a video from frames
        
        Args:
            frames: List of frames as numpy arrays
            output_path: Path to save the video
            fps: Frames per second
            fourcc: FourCC code for video codec
            
        Returns:
            Path to the created video
        """
        self._log_operation("create_video")
        
        if not frames:
            print("No frames provided to create video")
            return None
        
        try:
            # Ensure output directory exists
            if output_path.startswith(self.temp_dir):
                self._ensure_temp_dir()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Get frame dimensions
            height, width = frames[0].shape[:2]
            
            # Create video writer
            fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
            out = cv2.VideoWriter(output_path, fourcc_code, fps, (width, height))
            
            if not out.isOpened():
                raise ValueError(f"Could not create video writer for {output_path}")
            
            # Write frames
            for frame in frames:
                out.write(frame)
            
            # Close the video writer
            out.release()
            
            # Track as temporary file for cleanup
            self._add_temp_file(output_path)
            
            # Update stats
            self.stats["disk_writes"] += 1
            
            if self.verbose:
                print(f"Created video with {len(frames)} frames at {output_path}")
            
            return output_path
        
        except Exception as e:
            print(f"Error creating video: {str(e)}")
            return None
    
    def resize_video(self, input_path: str, output_path: str, width: int, height: int, 
                    maintain_aspect_ratio: bool = True) -> str:
        """
        Resize a video
        
        Args:
            input_path: Path to the input video
            output_path: Path to save the resized video
            width: Target width
            height: Target height
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Path to the resized video
        """
        self._log_operation("resize_video")
        
        try:
            # Get video info
            info = self.get_video_info(input_path)
            
            # Calculate dimensions if maintaining aspect ratio
            if maintain_aspect_ratio:
                original_width = info["width"]
                original_height = info["height"]
                aspect_ratio = original_width / original_height
                
                if width / height > aspect_ratio:
                    # Height is limiting factor
                    new_height = height
                    new_width = int(height * aspect_ratio)
                else:
                    # Width is limiting factor
                    new_width = width
                    new_height = int(width / aspect_ratio)
            else:
                new_width = width
                new_height = height
            
            # Extract frames
            frames = self.extract_frames(input_path)
            
            if not frames:
                print(f"No frames extracted from {input_path}")
                return None
            
            # Resize frames
            resized_frames = []
            
            for frame in frames:
                resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                resized_frames.append(resized_frame)
            
            # Create resized video
            return self.create_video(resized_frames, output_path, info["fps"])
        
        except Exception as e:
            print(f"Error resizing video: {str(e)}")
            return None
    
    def reverse_video(self, input_path: str, output_path: str) -> str:
        """
        Reverse a video
        
        Args:
            input_path: Path to the input video
            output_path: Path to save the reversed video
            
        Returns:
            Path to the reversed video
        """
        self._log_operation("reverse_video")
        
        try:
            # Get video info
            info = self.get_video_info(input_path)
            
            # Extract frames
            frames = self.extract_frames(input_path)
            
            if not frames:
                print(f"No frames extracted from {input_path}")
                return None
            
            # Reverse frames
            reversed_frames = frames[::-1]
            
            # Create reversed video
            return self.create_video(reversed_frames, output_path, info["fps"])
        
        except Exception as e:
            print(f"Error reversing video: {str(e)}")
            return None
    
    def extract_chunk(self, input_path: str, output_path: str, 
                    start_frame: int, end_frame: int) -> str:
        """
        Extract a chunk from a video
        
        Args:
            input_path: Path to the input video
            output_path: Path to save the chunk
            start_frame: Starting frame index
            end_frame: Ending frame index (exclusive)
            
        Returns:
            Path to the extracted chunk
        """
        self._log_operation("extract_chunk")
        
        try:
            # Get video info
            info = self.get_video_info(input_path)
            
            # Extract frames
            frames = self.extract_frames(input_path, start_frame, end_frame)
            
            if not frames:
                print(f"No frames extracted from {input_path} for range {start_frame}-{end_frame}")
                return None
            
            # Create chunk video
            return self.create_video(frames, output_path, info["fps"])
        
        except Exception as e:
            print(f"Error extracting chunk: {str(e)}")
            return None
    
    def process_video(self, input_path: str, operations: List[Dict[str, Any]], 
                     output_path: Optional[str] = None) -> str:
        """
        Process a video through a series of operations
        
        Args:
            input_path: Path to the input video
            operations: List of operations as dictionaries
            output_path: Path to save the final output (if None, use temp file)
            
        Returns:
            Path to the processed video
        """
        self._log_operation("process_video")
        
        # Start timing
        start_time = time.time()
        
        try:
            current_path = input_path
            
            # If we have operations to perform, ensure temp directory exists
            if operations:
                self._ensure_temp_dir()
            
            # Create a unique output path if not provided
            if output_path is None:
                output_path = os.path.join(
                    self.temp_dir, 
                    f"processed_{os.path.basename(input_path)}"
                )
            
            # Process each operation
            for i, op in enumerate(operations):
                op_type = op.get("type")
                
                if not op_type:
                    print(f"Missing operation type in operation {i}")
                    continue
                
                # Determine intermediate output path
                if i == len(operations) - 1:
                    # Last operation, use final output path
                    current_output_path = output_path
                else:
                    # Intermediate operation, use temp file
                    current_output_path = os.path.join(
                        self.temp_dir, 
                        f"op{i}_{os.path.basename(input_path)}"
                    )
                
                # Execute operation
                if op_type == "resize":
                    width = op.get("width")
                    height = op.get("height")
                    maintain_aspect = op.get("maintain_aspect_ratio", True)
                    
                    if not width or not height:
                        print(f"Missing dimensions for resize operation {i}")
                        continue
                    
                    current_path = self.resize_video(
                        current_path, 
                        current_output_path, 
                        width, 
                        height, 
                        maintain_aspect
                    )
                
                elif op_type == "reverse":
                    current_path = self.reverse_video(current_path, current_output_path)
                
                elif op_type == "extract":
                    start_frame = op.get("start_frame", 0)
                    end_frame = op.get("end_frame")
                    
                    current_path = self.extract_chunk(
                        current_path, 
                        current_output_path, 
                        start_frame, 
                        end_frame
                    )
                
                else:
                    print(f"Unknown operation type: {op_type}")
                    continue
                
                # Check if operation succeeded
                if not current_path:
                    print(f"Operation {i} ({op_type}) failed")
                    return None
            
            # Update stats
            end_time = time.time()
            processing_time = end_time - start_time
            self.stats["processing_time"] += processing_time
            
            if self.verbose:
                print(f"Video processing completed in {processing_time:.2f} seconds")
            
            return current_path
        
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return None
    
    def blend_videos(self, foreground_path: str, alpha_path: str, output_path: str) -> str:
        """
        Blend a foreground video with an alpha mask
        
        Args:
            foreground_path: Path to the foreground video
            alpha_path: Path to the alpha mask video
            output_path: Path to save the blended video
            
        Returns:
            Path to the blended video
        """
        self._log_operation("blend_videos")
        
        try:
            # Get video info
            fg_info = self.get_video_info(foreground_path)
            alpha_info = self.get_video_info(alpha_path)
            
            # Check dimensions match
            if fg_info["width"] != alpha_info["width"] or fg_info["height"] != alpha_info["height"]:
                print(f"Dimension mismatch: foreground {fg_info['width']}x{fg_info['height']}, "
                      f"alpha {alpha_info['width']}x{alpha_info['height']}")
                return None
            
            # Extract frames
            fg_frames = self.extract_frames(foreground_path)
            alpha_frames = self.extract_frames(alpha_path)
            
            # Check frame count match
            if len(fg_frames) != len(alpha_frames):
                print(f"Frame count mismatch: foreground {len(fg_frames)}, alpha {len(alpha_frames)}")
                return None
            
            # Blend frames
            blended_frames = []
            
            for fg, alpha in zip(fg_frames, alpha_frames):
                # Convert alpha to single channel if needed
                if len(alpha.shape) == 3:
                    alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
                
                # Ensure alpha is float in range [0, 1]
                alpha_float = alpha.astype(np.float32) / 255.0
                
                # Blend using alpha
                blended = (fg.astype(np.float32) * alpha_float[:, :, np.newaxis]).astype(np.uint8)
                blended_frames.append(blended)
            
            # Create blended video
            return self.create_video(blended_frames, output_path, fg_info["fps"])
        
        except Exception as e:
            print(f"Error blending videos: {str(e)}")
            return None
    
    def _ensure_temp_dir(self) -> None:
        """
        Ensure the temporary directory exists, creating it if necessary
        """
        if not self.temp_dir_created:
            os.makedirs(self.temp_dir, exist_ok=True)
            self.temp_dir_created = True
            if self.verbose:
                print(f"Created temp directory: {self.temp_dir}")
    
    def _add_temp_file(self, file_path: str) -> None:
        """
        Add a file to the list of temporary files
        
        Args:
            file_path: Path to the file
        """
        if file_path and file_path not in self.temp_files:
            self.temp_files.append(file_path)
    
    def _add_to_frame_cache(self, key: str, frames: List[np.ndarray]) -> None:
        """
        Add frames to the cache
        
        Args:
            key: Cache key
            frames: Frames to cache
        """
        # Check cache size
        # TODO: Implement cache size management
        self.frame_cache[key] = frames.copy()
    
    def _log_operation(self, operation: str) -> None:
        """
        Log an operation for statistics
        
        Args:
            operation: Name of the operation
        """
        if operation not in self.stats["operations"]:
            self.stats["operations"][operation] = 0
        self.stats["operations"][operation] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics
        
        Returns:
            Dictionary with statistics
        """
        return {
            "operations": self.stats["operations"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "hit_rate": self.stats["cache_hits"] / (self.stats["cache_hits"] + self.stats["cache_misses"]) if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0 else 0,
            "disk_reads": self.stats["disk_reads"],
            "disk_writes": self.stats["disk_writes"],
            "processing_time": self.stats["processing_time"],
            "temp_files": len(self.temp_files)
        }


# Example usage:
"""
# Create processor
processor = VideoProcessor()

# Resize a video
resized_path = processor.resize_video('input.mp4', 'resized.mp4', 640, 360)

# Reverse a video
reversed_path = processor.reverse_video('input.mp4', 'reversed.mp4')

# Chain operations
processed_path = processor.process_video('input.mp4', [
    {"type": "resize", "width": 640, "height": 360},
    {"type": "reverse"}
])

# Clean up temporary files
processor.cleanup()
"""