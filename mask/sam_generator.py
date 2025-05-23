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
        self.model_type_loaded = None
        
    def load_model(self):
        """Load and initialize the SAM model (tries SAM2 first, falls back to SAM)"""
        # Try SAM2 first (better quality), then fall back to original SAM
        if self._try_load_sam2():
            return True
        else:
            return self._try_load_sam()
    
    def _try_load_sam2(self):
        """Try to load SAM2 model"""
        try:
            import torch
            import os
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            # Determine device if not specified
            if self.device is None:
                try:
                    if torch.backends.mps.is_available():
                        # SAM2 has issues with MPS, so use CPU instead
                        self.device = torch.device("cpu")
                        print("MPS detected but SAM2 has compatibility issues with MPS.")
                        print("Using CPU for SAM2 (still fast enough for real-time use).")
                    elif torch.cuda.is_available():
                        self.device = torch.device("cuda")
                        print("Using CUDA for SAM2")
                    else:
                        self.device = torch.device("cpu")
                        print("Using CPU for SAM2")
                except:
                    # Fallback to CPU if device detection fails
                    self.device = torch.device("cpu")
                    print("Falling back to CPU for SAM2")
            
            # Define SAM2 model paths
            sam2_checkpoint_path = {
                "vit_h": "sam2_hiera_l.pth",  # Large model (best quality)
                "vit_l": "sam2_hiera_l.pth",  # Use large for both
                "vit_b": "sam2_hiera_b+.pth"  # Base+ model
            }
            
            # Use the large model by default for best quality
            model_name = "sam2_hiera_l.pth"
            # Note: SAM2 config files are embedded in the package, not separate files
            config_name = "sam2_hiera_l.yaml"
            
            # Download the model if it doesn't exist
            model_path = self._download_sam2_model(model_name)
            
            try:
                # Load SAM2 model
                print(f"Loading SAM2 model on {self.device}...")
                # Check if model file exists and is valid
                if not os.path.exists(model_path):
                    print(f"Model file not found at: {model_path}")
                    return False
                
                # Load and analyze checkpoint
                checkpoint = None
                has_model_key = False
                
                try:
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    if isinstance(checkpoint, dict):
                        print(f"Checkpoint keys: {list(checkpoint.keys())[:5]}...")  # Show first 5 keys
                        
                        # If checkpoint has 'model' key, we need to handle it differently
                        if 'model' in checkpoint:
                            has_model_key = True
                            print("Detected wrapped checkpoint format...")
                            # Check what's inside the 'model' key
                            model_data = checkpoint['model']
                            if isinstance(model_data, dict):
                                print(f"Model data keys: {list(model_data.keys())[:5]}...")
                except Exception as e:
                    print(f"Error loading checkpoint file: {str(e)}")
                    return False
                
                try:
                    # If checkpoint has 'model' key, we need to extract and save just the weights
                    if has_model_key:
                        print("Extracting model weights from checkpoint...")
                        # Create a new checkpoint file with just the state dict
                        weights_only_path = model_path.replace('.pth', '_weights.pth')
                        
                        # Check if we already extracted it
                        if not os.path.exists(weights_only_path):
                            # Save just the model state dict
                            torch.save(checkpoint['model'], weights_only_path)
                            print(f"Saved extracted weights to: {weights_only_path}")
                        
                        # Use the extracted weights file
                        model_path = weights_only_path
                    
                    # Try loading with the (possibly extracted) checkpoint
                    print("Attempting to load SAM2 model...")
                    sam2_model = build_sam2(config_name, model_path, device=self.device)
                    
                    self.predictor = SAM2ImagePredictor(sam2_model)
                    print("SAM2 model loaded successfully")
                    self.model_type_loaded = "SAM2"
                    return True
                except Exception as build_error:
                    import traceback
                    print(f"Failed to load SAM2: {str(build_error)}")
                    print("Full error traceback:")
                    traceback.print_exc()
                    
                    # Handle specific errors
                    if "load_state_dict" in str(build_error) or "Missing key" in str(build_error):
                        print("\nNote: SAM2 checkpoint format issue detected.")
                        print("This may be due to a version mismatch between SAM2 code and model.")
                        
                    return False
            except Exception as e:
                # If loading on MPS fails, fallback to CPU
                if str(self.device) == "mps":
                    print(f"Failed to load SAM2 on MPS: {str(e)}. Falling back to CPU.")
                    self.device = torch.device("cpu")
                    sam2_model = build_sam2(config_name, model_path, device=self.device)
                    self.predictor = SAM2ImagePredictor(sam2_model)
                    print("SAM2 model loaded successfully on CPU")
                    self.model_type_loaded = "SAM2"
                    return True
                else:
                    print(f"Failed to load SAM2: {str(e)}")
                    return False
                
        except ImportError:
            print("SAM2 not available, will try original SAM...")
            return False
        except Exception as e:
            print(f"Error loading SAM2: {str(e)}, will try original SAM...")
            return False
    
    def _try_load_sam(self):
        """Try to load original SAM model"""
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
                    self.model_type_loaded = "SAM"
                    return True
                else:
                    print(f"Failed to load SAM: {str(e)}")
                    return False
                
        except ImportError:
            print("Error: segment-anything package not found. Please install it with:")
            print("pip install git+https://github.com/facebookresearch/segment-anything.git")
            return False
    
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
    
    def _download_sam2_model(self, checkpoint_name):
        """
        Download the SAM2 model if it doesn't exist
        
        Args:
            checkpoint_name: Name of the SAM2 checkpoint file
        
        Returns:
            Path to the downloaded model
        """
        import os
        
        # Define the model directory (same as SAM)
        model_dir = os.path.expanduser("~/.cache/sam")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, checkpoint_name)
        
        # Check if model already exists
        if os.path.exists(model_path):
            return model_path
        
        # Also check in the application directory (for compatibility)
        local_model_path = os.path.join("pretrained_models/sam", checkpoint_name)
        if os.path.exists(local_model_path):
            return local_model_path
        
        # Define URLs for SAM2 models
        urls = {
            "sam2_hiera_l.pth": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
            "sam2_hiera_b+.pth": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"
        }
        
        if checkpoint_name not in urls:
            print(f"Unknown SAM2 model: {checkpoint_name}")
            print(f"Available models: {list(urls.keys())}")
            raise ValueError(f"Unknown SAM2 model: {checkpoint_name}")
        
        try:
            import urllib.request
            print(f"Downloading SAM2 model from {urls[checkpoint_name]}...")
            urllib.request.urlretrieve(urls[checkpoint_name], model_path)
            print(f"SAM2 model downloaded to {model_path}")
            return model_path
        except Exception as e:
            print(f"Error downloading SAM2 model: {str(e)}")
            print(f"Please download the model manually from {urls[checkpoint_name]}")
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
        import torch
        
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

    def save_mask(self, mask, output_path, keyframe=None):
        """
        Save the generated mask to a file
        
        Args:
            mask: Generated mask (numpy array, boolean or uint8)
            output_path: Path to save the mask
            keyframe: Frame number to store as keyframe metadata (optional)
        
        Returns:
            Path to the saved mask
        """
        import cv2
        import numpy as np
        from PIL import Image
        
        # Convert boolean mask to uint8
        if mask.dtype == bool:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = mask.astype(np.uint8)
        
        # If keyframe metadata should be added (non-zero frame)
        if keyframe is not None and keyframe != 0:
            # Use PIL to save with metadata
            from mask.mask_utils import add_keyframe_metadata_to_mask
            
            # First save with cv2 to a temporary path
            import tempfile
            import os
            temp_path = tempfile.mktemp(suffix='.png')
            cv2.imwrite(temp_path, mask_uint8)
            
            # Then add metadata and save to final path
            final_path = add_keyframe_metadata_to_mask(temp_path, keyframe, output_path)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return final_path
        else:
            # Standard save without metadata (frame 0 or no keyframe specified)
            cv2.imwrite(output_path, mask_uint8)
            return output_path
