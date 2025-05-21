"""
# mask_analysis.py - v1.1716160381
# Updated: Sunday, May 19, 2025
# Changes in this version:
# - Modified face detection to only detect faces within mask regions
# - Enhanced optimal frame selection to prefer frames with both faces and good mask coverage
# - Improved mask analysis for optimal keyframe selection in bidirectional processing
# - Added option to constrain face detection to areas within mask content
# - Fixed bug in frame range analysis when processing with multiple chunks

Mask analysis and manipulation utilities for MatAnyone video processing.
Contains functions for analyzing and manipulating video masks.
"""

import os
import cv2
import numpy as np
import traceback


def upscale_and_binarize_mask_video(mask_video_path, output_dir, width, height, threshold=128):
    """
    Extract frames from a mask video, upscale them, and apply binary thresholding
    
    Args:
        mask_video_path: Path to the input mask video
        output_dir: Directory to save upscaled frame masks
        width: Target width for upscaled masks
        height: Target height for upscaled masks
        threshold: Threshold value for binary mask (0-255)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        cap = cv2.VideoCapture(mask_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {mask_video_path}")
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process frames
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # For mask videos, we often just need one channel
            if frame.shape[2] == 3:
                # Use first channel for simplicity (masks should be grayscale)
                mask = frame[:, :, 0]
            else:
                mask = frame
            
            # Upscale mask using nearest-neighbor to preserve edges
            upscaled_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
            
            # Apply binary thresholding to ensure mask is either 0 or 255
            _, binary_mask = cv2.threshold(upscaled_mask, threshold, 255, cv2.THRESH_BINARY)
            
            # Save mask frame with a sequential number (8 digits for proper sorting)
            frame_path = os.path.join(output_dir, f"{frame_idx:08d}.png")
            cv2.imwrite(frame_path, binary_mask)
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Upscaled and binarized {frame_idx}/{frame_count} mask frames")
        
        # Release resources
        cap.release()
        
        print(f"Upscaled and binarized {frame_idx} mask frames to {width}x{height}")
        return True
        
    except Exception as e:
        print(f"Error upscaling mask video: {str(e)}")
        traceback.print_exc()
        return False


def detect_faces_in_frame(frame, face_cascade=None, min_size=(60, 60), mask=None):
    """
    Detect faces in a frame using OpenCV's face detection
    
    Args:
        frame: Input frame (numpy array)
        face_cascade: Pre-loaded face cascade classifier (will load default if None)
        min_size: Minimum face size to detect
        mask: Optional binary mask to constrain face detection to masked areas only
        
    Returns:
        Number of faces detected and a confidence score
    """
    try:
        # If no face cascade is provided, load the default
        if face_cascade is None:
            # Try to load the face cascade from OpenCV's path
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(face_cascade_path):
                # Fallback to looking in the current directory
                face_cascade_path = 'haarcascade_frontalface_default.xml'
                if not os.path.exists(face_cascade_path):
                    print("Warning: Face cascade file not found. Face detection disabled.")
                    return 0, 0.0
            
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Convert to grayscale for face detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # If a mask is provided, apply it to only detect faces in masked areas
        if mask is not None:
            # Make sure mask is grayscale
            if len(mask.shape) == 3:
                mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            else:
                mask_gray = mask
            
            # Ensure the mask is binary (0 or 255)
            _, binary_mask = cv2.threshold(mask_gray, 128, 255, cv2.THRESH_BINARY)
            
            # Apply mask to the grayscale image
            # Faces will only be detected in the masked areas
            masked_gray = cv2.bitwise_and(gray, gray, mask=binary_mask)
            
            # Use masked image for face detection
            detection_image = masked_gray
        else:
            # Use original grayscale image for face detection
            detection_image = gray
            
        # Detect faces
        faces = face_cascade.detectMultiScale(
            detection_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=min_size
        )
        
        # Calculate confidence based on number and size of faces
        num_faces = len(faces)
        confidence = 0.0
        
        if num_faces > 0:
            # Calculate confidence based on face size relative to frame
            frame_area = gray.shape[0] * gray.shape[1]
            face_areas = [w * h for (x, y, w, h) in faces]
            max_face_area = max(face_areas) if face_areas else 0
            
            # Normalized confidence (0-1), higher for larger faces
            confidence = min(1.0, (max_face_area / frame_area) * 10)
            
        return num_faces, confidence
        
    except Exception as e:
        print(f"Error in face detection: {str(e)}")
        return 0, 0.0


def analyze_masks_for_optimal_ranges(mask_dir, frame_count, start_x, end_x, start_y, end_y, threshold=5, 
                                   prioritize_faces=True, use_original_frames=True, original_video_path=None,
                                   only_detect_faces_in_mask=True):
    """
    Analyze masks to find optimal frame ranges with best mask coverage
    
    Args:
        mask_dir: Directory containing mask frames
        frame_count: Total number of frames in the video
        start_x: Start X coordinate of the chunk
        end_x: End X coordinate of the chunk
        start_y: Start Y coordinate of the chunk
        end_y: End Y coordinate of the chunk
        threshold: Percentage threshold of non-zero pixels to consider a mask chunk worth processing
        prioritize_faces: Whether to prioritize frames with faces as keyframes
        use_original_frames: Whether to use original video frames for face detection
        original_video_path: Path to the original video (required if use_original_frames is True)
        only_detect_faces_in_mask: Whether to only detect faces within the mask area
    
    Returns:
        List of tuples (start_frame, end_frame, keyframe) where keyframe has optimal mask coverage
    """
    try:
        # Create list to store which frames have mask content
        frame_mask_coverage = []
        face_detections = []
        
        # Initialize face detection if needed
        face_cascade = None
        if prioritize_faces:
            try:
                # Try to load the face cascade from OpenCV's data path
                face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if not os.path.exists(face_cascade_path):
                    # Fallback to looking in the current directory
                    face_cascade_path = 'haarcascade_frontalface_default.xml'
                    if not os.path.exists(face_cascade_path):
                        print("Warning: Face cascade file not found. Face detection disabled.")
                        prioritize_faces = False
                    
                if prioritize_faces:
                    face_cascade = cv2.CascadeClassifier(face_cascade_path)
                    print("Face detection enabled for optimal keyframe selection")
            except:
                print("Error initializing face detection. Falling back to mask coverage only.")
                prioritize_faces = False
        
        # Open the original video if needed for face detection
        original_cap = None
        if prioritize_faces and use_original_frames and original_video_path:
            try:
                original_cap = cv2.VideoCapture(original_video_path)
                if not original_cap.isOpened():
                    print(f"Warning: Could not open original video for face detection: {original_video_path}")
                    original_cap = None
            except:
                print("Error opening original video for face detection")
                original_cap = None
        
        # Check each mask frame
        for frame_idx in range(frame_count):
            # Construct mask frame path
            frame_path = os.path.join(mask_dir, f"{frame_idx:08d}.png")
            
            # Skip if mask frame doesn't exist
            if not os.path.exists(frame_path):
                frame_mask_coverage.append(0.0)
                face_detections.append((0, 0.0))
                continue
            
            # Load mask frame
            mask_frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            
            # Skip if loading fails
            if mask_frame is None:
                frame_mask_coverage.append(0.0)
                face_detections.append((0, 0.0))
                continue
            
            # Extract chunk portion of the mask
            if start_y < mask_frame.shape[0] and start_x < mask_frame.shape[1]:
                # Handle edge cases where mask might be smaller than expected
                actual_end_y = min(end_y, mask_frame.shape[0])
                actual_end_x = min(end_x, mask_frame.shape[1])
                
                chunk_mask = mask_frame[start_y:actual_end_y, start_x:actual_end_x]
                
                # Calculate percentage of non-zero pixels
                non_zero_count = np.count_nonzero(chunk_mask)
                total_pixels = chunk_mask.size
                non_zero_percentage = (non_zero_count / total_pixels) * 100
            else:
                non_zero_percentage = 0.0
            
            # Store coverage percentage
            frame_mask_coverage.append(non_zero_percentage)
            
            # Detect faces if enabled and we have mask content
            if prioritize_faces and non_zero_percentage >= threshold:
                if original_cap is not None:
                    # Use original video frame for better face detection
                    original_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, original_frame = original_cap.read()
                    
                    if ret:
                        # Crop to the current chunk
                        if start_y < original_frame.shape[0] and start_x < original_frame.shape[1]:
                            actual_end_y = min(end_y, original_frame.shape[0])
                            actual_end_x = min(end_x, original_frame.shape[1])
                            
                            chunk_frame = original_frame[start_y:actual_end_y, start_x:actual_end_x]
                            
                            # Detect faces either within the mask or in the entire chunk
                            if only_detect_faces_in_mask:
                                # Use chunk mask to constrain face detection to mask area
                                num_faces, face_confidence = detect_faces_in_frame(
                                    chunk_frame, 
                                    face_cascade, 
                                    mask=chunk_mask if non_zero_percentage >= threshold else None
                                )
                            else:
                                # Detect faces in the entire chunk
                                num_faces, face_confidence = detect_faces_in_frame(chunk_frame, face_cascade)
                                
                            face_detections.append((num_faces, face_confidence))
                        else:
                            face_detections.append((0, 0.0))
                    else:
                        face_detections.append((0, 0.0))
                else:
                    # Use the mask for face detection (less accurate but can still work)
                    # Convert binary mask to 3-channel for face detection
                    mask_rgb = cv2.cvtColor(chunk_mask, cv2.COLOR_GRAY2BGR)
                    
                    # Detect faces either within the mask or in the entire chunk
                    if only_detect_faces_in_mask:
                        # For mask-only detection with just the mask,
                        # we don't need to provide the mask separately
                        num_faces, face_confidence = detect_faces_in_frame(mask_rgb, face_cascade)
                    else:
                        # Same as above since we're already using just the mask
                        num_faces, face_confidence = detect_faces_in_frame(mask_rgb, face_cascade)
                        
                    face_detections.append((num_faces, face_confidence))
            else:
                # No face detection possible or needed
                face_detections.append((0, 0.0))
            
            # Print progress periodically
            if frame_idx % 100 == 0 or frame_idx == frame_count - 1:
                print(f"Analyzed {frame_idx+1}/{frame_count} mask frames for chunk ({start_x},{start_y})-({end_x},{end_y})")
        
        # Clean up video capture
        if original_cap is not None:
            original_cap.release()
        
        # Find continuous ranges where coverage is above threshold
        ranges = []
        start_range = None
        
        for i in range(frame_count):
            if frame_mask_coverage[i] >= threshold and start_range is None:
                # Start a new range
                start_range = i
            elif (frame_mask_coverage[i] < threshold or i == frame_count - 1) and start_range is not None:
                # End of range or last frame
                end_range = i if frame_mask_coverage[i] < threshold else i
                
                # Only include ranges with at least 2 frames
                if end_range - start_range >= 1:
                    # Find optimal keyframe within this range
                    range_coverages = frame_mask_coverage[start_range:end_range+1]
                    range_faces = face_detections[start_range:end_range+1]
                    
                    if prioritize_faces:
                        # Check if any frames have faces
                        face_frames = [(start_range + i, count, conf, range_coverages[i]) 
                                     for i, (count, conf) in enumerate(range_faces) if count > 0]
                        
                        if face_frames:
                            # Consider both face confidence AND mask coverage
                            # First, check if any of the face frames have decent mask coverage
                            # (at least 25% of the maximum coverage in this range)
                            max_coverage = max(range_coverages)
                            min_acceptable_coverage = max(threshold, max_coverage * 0.25)
                            
                            # Filter face frames by minimum acceptable coverage
                            good_face_frames = [f for f in face_frames if f[3] >= min_acceptable_coverage]
                            
                            if good_face_frames:
                                # Sort by a combined score that considers both face confidence and coverage
                                # This way we get frames with good faces AND good mask coverage
                                # Normalize both scores to 0-1 range
                                max_face_conf = max(f[2] for f in good_face_frames)
                                max_mask_cov = max(f[3] for f in good_face_frames)
                                
                                # Calculate combined scores
                                scored_frames = []
                                for frame_idx, face_count, face_conf, coverage in good_face_frames:
                                    # Normalize scores
                                    norm_face_conf = face_conf / max(max_face_conf, 0.001)
                                    norm_coverage = coverage / max(max_mask_cov, 0.001)
                                    
                                    # Combined score with slightly higher weight on face confidence
                                    combined_score = (norm_face_conf * 0.6) + (norm_coverage * 0.4)
                                    scored_frames.append((frame_idx, face_count, face_conf, coverage, combined_score))
                                
                                # Sort by combined score, descending
                                scored_frames.sort(key=lambda x: x[4], reverse=True)
                                keyframe = scored_frames[0][0]
                                face_count = scored_frames[0][1]
                                face_conf = scored_frames[0][2]
                                coverage = scored_frames[0][3]
                                print(f"Selected keyframe {keyframe} with {face_count} faces (conf: {face_conf:.2f}) and {coverage:.2f}% coverage")
                            else:
                                # No faces with good coverage, fall back to max coverage
                                max_coverage_offset = np.argmax(range_coverages)
                                keyframe = start_range + max_coverage_offset
                                coverage = range_coverages[max_coverage_offset]
                                print(f"No faces with adequate coverage. Using max coverage keyframe {keyframe} ({coverage:.2f}%)")
                        else:
                            # No faces detected, fall back to mask coverage
                            max_coverage_offset = np.argmax(range_coverages)
                            keyframe = start_range + max_coverage_offset
                            coverage = range_coverages[max_coverage_offset]
                            print(f"No faces detected. Using max coverage keyframe {keyframe} ({coverage:.2f}%)")
                    else:
                        # Just use mask coverage
                        max_coverage_offset = np.argmax(range_coverages)
                        keyframe = start_range + max_coverage_offset
                    
                    # Add to ranges list
                    ranges.append((start_range, end_range, keyframe))
                
                start_range = None
        
        # Optimize ranges: combine close ranges to reduce overhead
        if len(ranges) > 1:
            optimized_ranges = [ranges[0]]
            
            for i in range(1, len(ranges)):
                prev_start, prev_end, prev_key = optimized_ranges[-1]
                curr_start, curr_end, curr_key = ranges[i]
                
                # If ranges are close (less than 10 frames apart), combine them
                if curr_start - prev_end <= 10:
                    # Determine which keyframe to use (face priority or coverage)
                    if prioritize_faces:
                        prev_face_count, prev_face_conf = face_detections[prev_key]
                        curr_face_count, curr_face_conf = face_detections[curr_key]
                        prev_coverage = frame_mask_coverage[prev_key]
                        curr_coverage = frame_mask_coverage[curr_key]
                        
                        # Check if both have faces
                        if prev_face_count > 0 and curr_face_count > 0:
                            # Both have faces, calculate combined score
                            prev_score = (prev_face_conf * 0.6) + (prev_coverage/100 * 0.4)
                            curr_score = (curr_face_conf * 0.6) + (curr_coverage/100 * 0.4)
                            
                            best_key = prev_key if prev_score >= curr_score else curr_key
                        elif prev_face_count > 0:
                            # Check if prev face keyframe has decent coverage
                            if prev_coverage >= threshold * 2:  # Stricter requirement
                                best_key = prev_key
                            else:
                                # Use coverage as fallback
                                best_key = prev_key if prev_coverage >= curr_coverage else curr_key
                        elif curr_face_count > 0:
                            # Check if curr face keyframe has decent coverage
                            if curr_coverage >= threshold * 2:  # Stricter requirement
                                best_key = curr_key
                            else:
                                # Use coverage as fallback
                                best_key = prev_key if prev_coverage >= curr_coverage else curr_key
                        else:
                            # No faces, use highest coverage
                            best_key = prev_key if prev_coverage >= curr_coverage else curr_key
                    else:
                        # Determine which keyframe has better coverage
                        prev_coverage = frame_mask_coverage[prev_key]
                        curr_coverage = frame_mask_coverage[curr_key]
                        
                        best_key = prev_key if prev_coverage >= curr_coverage else curr_key
                    
                    # Update the last range instead of adding a new one
                    optimized_ranges[-1] = (prev_start, curr_end, best_key)
                else:
                    # Add as a separate range
                    optimized_ranges.append(ranges[i])
            
            ranges = optimized_ranges
        
        # Print identified ranges
        if ranges:
            print(f"Identified {len(ranges)} optimal frame ranges for chunk ({start_x},{start_y})-({end_x},{end_y}):")
            for i, (start, end, key) in enumerate(ranges):
                coverage = frame_mask_coverage[key]
                num_faces, face_conf = face_detections[key]
                face_info = f", {num_faces} faces detected" if num_faces > 0 else ""
                print(f"  Range {i+1}: Frames {start}-{end} with keyframe at {key} ({coverage:.2f}% coverage{face_info})")
        else:
            print(f"No frame ranges with significant mask content found for chunk ({start_x},{start_y})-({end_x},{end_y})")
            
        return ranges
        
    except Exception as e:
        print(f"Error analyzing mask frames: {str(e)}")
        traceback.print_exc()
        return []


def create_optimal_mask_for_range(original_mask, mask_dir, keyframe, start_x, end_x, start_y, end_y, output_path):
    """
    Create an optimal mask for a specific range using either the original mask or a specific keyframe mask
    
    Args:
        original_mask: Original mask image (numpy array)
        mask_dir: Directory containing mask frames
        keyframe: Keyframe index for the range
        start_x: Start X coordinate of the chunk
        end_x: End X coordinate of the chunk
        start_y: Start Y coordinate of the chunk
        end_y: End Y coordinate of the chunk
        output_path: Path to save the created mask
        
    Returns:
        Path to the created mask
    """
    try:
        # First, try to load the specific keyframe mask from the full_res_mask_dir
        keyframe_path = os.path.join(mask_dir, f"{keyframe:08d}.png")
        
        if os.path.exists(keyframe_path):
            # Load keyframe mask
            keyframe_mask = cv2.imread(keyframe_path, cv2.IMREAD_GRAYSCALE)
            
            if keyframe_mask is not None:
                # Extract chunk portion
                chunk_height = end_y - start_y
                chunk_width = end_x - start_x
                
                if start_y < keyframe_mask.shape[0] and start_x < keyframe_mask.shape[1]:
                    # Handle edge cases where mask might be smaller than expected
                    actual_end_y = min(end_y, keyframe_mask.shape[0])
                    actual_end_x = min(end_x, keyframe_mask.shape[1])
                    
                    # Extract the portion of the mask for this chunk
                    chunk_mask = keyframe_mask[start_y:actual_end_y, start_x:actual_end_x]
                    
                    # If dimensions don't match expected, resize or pad
                    if chunk_mask.shape[0] != chunk_height or chunk_mask.shape[1] != chunk_width:
                        # Create a blank mask of expected size
                        padded_mask = np.zeros((chunk_height, chunk_width), dtype=np.uint8)
                        
                        # Copy what we have
                        available_height = min(chunk_mask.shape[0], chunk_height)
                        available_width = min(chunk_mask.shape[1], chunk_width)
                        padded_mask[:available_height, :available_width] = chunk_mask[:available_height, :available_width]
                        
                        # Use the padded mask
                        chunk_mask = padded_mask
                    
                    # Ensure binary mask (0 or 255)
                    _, chunk_mask = cv2.threshold(chunk_mask, 128, 255, cv2.THRESH_BINARY)
                    
                    # Save the mask
                    cv2.imwrite(output_path, chunk_mask)
                    print(f"Created mask for range using keyframe {keyframe} from full-res mask directory")
                    return output_path
                
        # If keyframe mask doesn't exist or couldn't be loaded, use the original mask
        if original_mask is not None:
            # Extract chunk portion
            chunk_height = end_y - start_y
            chunk_width = end_x - start_x
            
            if start_y < original_mask.shape[0] and start_x < original_mask.shape[1]:
                # Handle edge cases where mask might be smaller than expected
                actual_end_y = min(end_y, original_mask.shape[0])
                actual_end_x = min(end_x, original_mask.shape[1])
                
                # Extract the portion of the mask for this chunk
                chunk_mask = original_mask[start_y:actual_end_y, start_x:actual_end_x]
                
                # If dimensions don't match expected, resize or pad
                if chunk_mask.shape[0] != chunk_height or chunk_mask.shape[1] != chunk_width:
                    # Create a blank mask of expected size
                    padded_mask = np.zeros((chunk_height, chunk_width), dtype=np.uint8)
                    
                    # Copy what we have
                    available_height = min(chunk_mask.shape[0], chunk_height)
                    available_width = min(chunk_mask.shape[1], chunk_width)
                    padded_mask[:available_height, :available_width] = chunk_mask[:available_height, :available_width]
                    
                    # Use the padded mask
                    chunk_mask = padded_mask
                
                # Ensure binary mask (0 or 255)
                _, chunk_mask = cv2.threshold(chunk_mask, 128, 255, cv2.THRESH_BINARY)
                
                # Save the mask
                cv2.imwrite(output_path, chunk_mask)
                print(f"Created mask for range using original mask (keyframe {keyframe} mask not found)")
                return output_path
        
        # If neither mask could be used, create an empty mask
        chunk_height = end_y - start_y
        chunk_width = end_x - start_x
        empty_mask = np.zeros((chunk_height, chunk_width), dtype=np.uint8)
        cv2.imwrite(output_path, empty_mask)
        print(f"Created empty mask for range (no masks available)")
        return output_path
        
    except Exception as e:
        print(f"Error creating optimal mask for range: {str(e)}")
        traceback.print_exc()
        
        # Create an empty mask as fallback
        try:
            chunk_height = end_y - start_y
            chunk_width = end_x - start_x
            empty_mask = np.zeros((chunk_height, chunk_width), dtype=np.uint8)
            cv2.imwrite(output_path, empty_mask)
            return output_path
        except:
            return None
