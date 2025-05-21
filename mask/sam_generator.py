"""
Segment Anything Model (SAM) mask generator for MatAnyone.
"""

import os
import numpy as np
import cv2
import tempfile
import platform
from pathlib import Path
import traceback


class SAMMaskGenerator:
    """Class to generate masks using Segment Anything Model (SAM)"""
    
    def __init__(self, model_type="vit_b", device=None):
        """
        Initialize SAM mask generator
        
        Args:
            model_type: Type of SAM model to use (vit_h, vit_l, vit_b)
            device: Device to run the model on (None for auto-detection)
        """
        self.model_type = model_type
        self.device = device
        self.model = None
        self.predictor = None
        
    def load_model(self):
        """Load and initialize the SAM model"""
        # Import SAM here to avoid dependency issues if not used
        try:
            import torch
            from segment_anything import sam_model_registry, SamPredictor
            
            # Determine device if not specified
            if self.device is None:
                try:
                    if torch.backends.mps.is_available():
                        self.device = torch.device("mps")
                        print("Using MPS (Apple Metal) for SAM")
                    elif torch.cuda.is_available():
                        self.device = torch.device("cuda")
                        print("Using CUDA for SAM")
                    else:
                        self.device = torch.device("cpu")
                        print("Using CPU for SAM")
                except:
                    # Fallback to CPU if device detection fails
                    self.device = torch.device("cpu")
                    print("Falling back to CPU for SAM")
            
            # Define model paths
            checkpoint_path = {
                "vit_h": "sam_vit_h_4b8939.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
                "vit_b": "sam_vit_b_01ec64.pth"
            }
            
            # Download the model if it doesn't exist
            model_path = self._download_model(self.model_type, checkpoint_path[self.model_type])
            
            try:
                # Load the model with the detected device
                print(f"Loading SAM model on {self.device}...")
                sam = sam_model_registry[self.model_type](checkpoint=model_path)
                sam.to(self.device)
                self.predictor = SamPredictor(sam)
                print("SAM model loaded successfully")
                return True
            except Exception as e:
                # If loading on MPS fails, fallback to CPU
                if str(self.device) == "mps":
                    print(f"Failed to load SAM on MPS: {str(e)}. Falling back to CPU.")
                    self.device = torch.device("cpu")
                    sam = sam_model_registry[self.model_type](checkpoint=model_path)
                    sam.to(self.device)
                    self.predictor = SamPredictor(sam)
                    print("SAM model loaded successfully on CPU")
                    return True
                else:
                    # Re-raise the exception if not on MPS
                    raise
                
        except ImportError:
            print("Error: segment-anything package not found. Please install it with:")
            print("pip install git+https://github.com/facebookresearch/segment-anything.git")
            raise
    
    def _download_model(self, model_type, checkpoint_name):
        """
        Download the SAM model if it doesn't exist
        
        Args:
            model_type: Type of SAM model
            checkpoint_name: Name of the checkpoint file
        
        Returns:
            Path to the downloaded model
        """
        import os
        
        # Define the model directory
        model_dir = os.path.expanduser("~/.cache/sam")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, checkpoint_name)
        
        # Also check in the current directory
        local_model_path = os.path.join(os.getcwd(), checkpoint_name)
        
        # Check if model exists locally first
        if os.path.exists(local_model_path):
            print(f"Using local SAM model: {local_model_path}")
            return local_model_path
        
        # Check if model exists in cache
        if os.path.exists(model_path):
            print(f"Using cached SAM model: {model_path}")
            return model_path
        
        # Model doesn't exist, need to download
        print(f"Downloading SAM {model_type} model...")
        # Define URLs for the models
        urls = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        }
        
        try:
            import urllib.request
            print(f"Downloading from {urls[model_type]}...")
            urllib.request.urlretrieve(urls[model_type], model_path)
            print(f"Model downloaded to {model_path}")
            return model_path
        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            print(f"Please download the model manually from {urls[model_type]}")
            print(f"And place it in {model_path} or in the application directory")
            raise
    
    def generate_mask_from_image(self, image, points=None, box=None, multimask_output=True):
        """
        Generate a mask from an image using SAM
        
        Args:
            image: Input image (numpy array, HWC format, RGB)
            points: Points to consider for mask generation [x, y, label], label: 1 for foreground, 0 for background
            box: Bounding box for mask generation [x1, y1, x2, y2]
            multimask_output: Whether to return multiple masks
        
        Returns:
            Generated mask and confidence score
        """
        import numpy as np
        
        if self.predictor is None:
            self.load_model()
        
        # Set the image for the predictor
        self.predictor.set_image(image)
        
        # Generate masks based on provided input
        if points is not None and len(points) > 0:
            # Convert points to numpy arrays with labels
            input_points = np.array(points[:, :2])
            input_labels = np.array(points[:, 2])
            
            # Ensure we have at least one foreground point for proper background handling
            has_fg = np.any(input_labels == 1)
            if not has_fg:
                print("Warning: No foreground points specified, adding automatic foreground point in center")
                # Add a foreground point in the center as fallback
                h, w = image.shape[:2]
                center_point = np.array([[w//2, h//2]])
                center_label = np.array([1])  # Foreground
                
                # Combine with existing points
                input_points = np.vstack([input_points, center_point]) 
                input_labels = np.append(input_labels, center_label)
            
            # If we also have a box, use it together with the points
            if box is not None:
                masks, scores, logits = self.predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    box=np.array(box),
                    multimask_output=multimask_output
                )
            else:
                masks, scores, logits = self.predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=multimask_output
                )
        elif box is not None:
            masks, scores, logits = self.predictor.predict(
                box=np.array(box),
                multimask_output=multimask_output
            )
        else:
            raise ValueError("Either points or box must be provided")
        
        # Return the mask with the highest score
        if multimask_output:
            mask_idx = np.argmax(scores)
            return masks[mask_idx], scores[mask_idx]
        else:
            return masks[0], scores[0]

    def extract_frame(self, video_path, frame_index=0):
        """
        Extract a specific frame from a video
        
        Args:
            video_path: Path to the video file
            frame_index: Index of the frame to extract (0-based)
        
        Returns:
            Frame from the video (numpy array, RGB)
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Ensure frame index is valid
        frame_index = max(0, min(frame_index, total_frames - 1))
        
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        # Read the frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame {frame_index} from video: {video_path}")
        
        # Convert BGR (OpenCV) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
    
    def get_video_info(self, video_path):
        """
        Get information about a video file
        
        Args:
            video_path: Path to the video file
        
        Returns:
            Dictionary with video information (fps, frame_count, width, height)
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height
        }

    def save_mask(self, mask, output_path):
        """
        Save the generated mask to a file
        
        Args:
            mask: Generated mask (numpy array, boolean or uint8)
            output_path: Path to save the mask
        
        Returns:
            Path to the saved mask
        """
        import cv2
        import numpy as np
        
        # Convert boolean mask to uint8
        if mask.dtype == bool:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = mask.astype(np.uint8)
        
        # Save the mask
        cv2.imwrite(output_path, mask_uint8)
        return output_path
