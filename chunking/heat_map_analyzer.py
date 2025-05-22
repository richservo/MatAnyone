"""
# heat_map_analyzer.py - v1.000000000
# Created: Wednesday, December 18, 2024
# Analyzes mask sequences to create heat maps for intelligent chunk placement

This module creates heat maps from mask sequences to identify areas of activity
across an entire video sequence, with priority given to facial regions.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import time

# Import face detection utilities
try:
    from mask.mask_analysis import detect_faces_in_frame
    FACE_DETECTION_AVAILABLE = True
except ImportError:
    print("Warning: Face detection not available for heat map analysis")
    FACE_DETECTION_AVAILABLE = False
    def detect_faces_in_frame(frame):
        return []


class HeatMapAnalyzer:
    """
    Analyzes mask sequences to create heat maps for optimal chunk placement.
    """
    
    def __init__(self, face_priority_weight: float = 3.0):
        """
        Initialize the heat map analyzer.
        
        Args:
            face_priority_weight: Multiplier for face regions in heat map (default 3.0)
        """
        self.face_priority_weight = face_priority_weight
        self.heat_map = None
        self.face_heat_map = None
        self.combined_heat_map = None
        self.frame_count = 0
        self.width = 0
        self.height = 0
        
    def analyze_mask_sequence(self, mask_dir: str, original_frames_dir: Optional[str] = None) -> np.ndarray:
        """
        Analyze a sequence of masks to create a combined heat map.
        
        Args:
            mask_dir: Directory containing mask frames
            original_frames_dir: Optional directory with original frames for face detection
            
        Returns:
            Combined heat map as numpy array
        """
        print(f"Analyzing mask sequence in: {mask_dir}")
        
        # Get all mask files
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        if not mask_files:
            raise ValueError(f"No mask files found in {mask_dir}")
        
        print(f"Found {len(mask_files)} mask frames to analyze")
        
        # Initialize heat maps with first mask
        first_mask_path = os.path.join(mask_dir, mask_files[0])
        first_mask = cv2.imread(first_mask_path, cv2.IMREAD_GRAYSCALE)
        if first_mask is None:
            raise ValueError(f"Could not read first mask: {first_mask_path}")
        
        self.height, self.width = first_mask.shape
        self.frame_count = len(mask_files)
        
        # Initialize heat maps
        self.heat_map = np.zeros((self.height, self.width), dtype=np.float32)
        self.face_heat_map = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Process each mask frame
        print("Creating base activity heat map...")
        for i, mask_file in enumerate(mask_files):
            if i % 100 == 0:
                print(f"Processing frame {i}/{len(mask_files)}")
            
            mask_path = os.path.join(mask_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"Warning: Could not read mask {mask_file}")
                continue
            
            # Add mask to heat map (normalize to 0-1 range)
            self.heat_map += mask.astype(np.float32) / 255.0
        
        # Normalize base heat map by frame count
        self.heat_map = self.heat_map / self.frame_count
        
        # Process face detection if original frames are available
        if original_frames_dir and os.path.exists(original_frames_dir) and FACE_DETECTION_AVAILABLE:
            print("Detecting faces in original frames...")
            self._create_face_heat_map(original_frames_dir, mask_files)
        elif original_frames_dir and not FACE_DETECTION_AVAILABLE:
            print("Skipping face detection (not available)")
        
        # Combine heat maps
        self._combine_heat_maps()
        
        print(f"Heat map analysis complete. Shape: {self.combined_heat_map.shape}")
        return self.combined_heat_map
    
    def _create_face_heat_map(self, frames_dir: str, mask_files: List[str]):
        """
        Create a heat map based on face detections in original frames.
        
        Args:
            frames_dir: Directory containing original frames
            mask_files: List of mask file names (to match frame numbers)
        """
        face_count = 0
        
        for i, mask_file in enumerate(mask_files):
            if i % 100 == 0 and i > 0:
                print(f"Face detection progress: {i}/{len(mask_files)} frames, {face_count} faces found")
            
            # Extract frame number from mask filename
            frame_num = int(mask_file.split('_')[-1].split('.')[0])
            
            # Look for corresponding original frame
            frame_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_path = os.path.join(frames_dir, f"frame_{frame_num:06d}{ext}")
                if os.path.exists(potential_path):
                    frame_path = potential_path
                    break
            
            if not frame_path:
                continue
            
            # Read frame
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            
            # Detect faces
            faces = detect_faces_in_frame(frame)
            
            # Add face regions to heat map
            for (x, y, w, h) in faces:
                # Add a Gaussian-like weight centered on face
                face_center_x = x + w // 2
                face_center_y = y + h // 2
                
                # Create a mask for this face region with soft edges
                Y, X = np.ogrid[:self.height, :self.width]
                dist_from_center = np.sqrt((X - face_center_x)**2 + (Y - face_center_y)**2)
                
                # Use face size to determine spread
                face_radius = max(w, h) // 2
                face_mask = np.exp(-(dist_from_center**2) / (2 * (face_radius**2)))
                
                # Add to face heat map
                self.face_heat_map += face_mask
                face_count += 1
        
        # Normalize face heat map
        if face_count > 0:
            self.face_heat_map = self.face_heat_map / face_count
            print(f"Total faces detected: {face_count}")
        else:
            print("No faces detected in sequence")
    
    def _combine_heat_maps(self):
        """
        Combine base and face heat maps into final heat map.
        """
        if np.max(self.face_heat_map) > 0:
            # Normalize face heat map to 0-1 range
            face_normalized = self.face_heat_map / np.max(self.face_heat_map)
            
            # Combine with base heat map using face priority weight
            self.combined_heat_map = self.heat_map + (face_normalized * self.face_priority_weight)
            
            # Normalize combined heat map
            self.combined_heat_map = self.combined_heat_map / np.max(self.combined_heat_map)
            
            print(f"Combined heat map with face priority (weight: {self.face_priority_weight})")
        else:
            # No faces detected, use base heat map only
            self.combined_heat_map = self.heat_map
            print("Using base heat map only (no faces detected)")
    
    def save_heat_map_visualization(self, output_path: str):
        """
        Save a visualization of the heat map.
        
        Args:
            output_path: Path to save the visualization
        """
        if self.combined_heat_map is None:
            print("No heat map to visualize")
            return
        
        # Convert to colormap for visualization
        heat_map_vis = (self.combined_heat_map * 255).astype(np.uint8)
        heat_map_colored = cv2.applyColorMap(heat_map_vis, cv2.COLORMAP_JET)
        
        # Save visualization
        cv2.imwrite(output_path, heat_map_colored)
        print(f"Heat map visualization saved to: {output_path}")
    
    def get_activity_stats(self) -> Dict[str, float]:
        """
        Get statistics about the heat map.
        
        Returns:
            Dictionary with activity statistics
        """
        if self.combined_heat_map is None:
            return {}
        
        stats = {
            'mean_activity': float(np.mean(self.combined_heat_map)),
            'max_activity': float(np.max(self.combined_heat_map)),
            'min_activity': float(np.min(self.combined_heat_map)),
            'std_activity': float(np.std(self.combined_heat_map)),
            'active_pixels': int(np.sum(self.combined_heat_map > 0.1)),
            'total_pixels': int(self.width * self.height),
            'activity_ratio': float(np.sum(self.combined_heat_map > 0.1) / (self.width * self.height))
        }
        
        return stats


def create_heat_map_from_masks(mask_dir: str, original_frames_dir: Optional[str] = None, 
                              face_priority: float = 3.0) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Convenience function to create a heat map from a mask directory.
    
    Args:
        mask_dir: Directory containing mask frames
        original_frames_dir: Optional directory with original frames for face detection
        face_priority: Face region priority weight
        
    Returns:
        Tuple of (heat_map, stats_dict)
    """
    analyzer = HeatMapAnalyzer(face_priority_weight=face_priority)
    heat_map = analyzer.analyze_mask_sequence(mask_dir, original_frames_dir)
    stats = analyzer.get_activity_stats()
    
    return heat_map, stats