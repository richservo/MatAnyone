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
        
        # Threshold settings for better control
        self.mask_threshold = 0.0  # Default SAM threshold for converting logits to binary mask
        self.stability_score_threshold = 0.95  # Higher = more stable masks only
        self.stability_score_offset = 1.0  # Offset for calculating stability score
        self.box_nms_thresh = 0.7  # Non-max suppression threshold for boxes
        self.crop_nms_thresh = 0.7  # Non-max suppression threshold for crops
        
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
            # Try different config names - sam2.1 uses different naming
            config_name = "sam2.1_hiera_l.yaml"
            
            # First check if new config exists, otherwise use old one
            try:
                from sam2 import sam2_configs
                # This will raise an error if config doesn't exist
                test_cfg = sam2_configs._CONFIGS.get(config_name)
                if test_cfg is None:
                    config_name = "sam2_hiera_l.yaml"
                    print(f"Using config: {config_name}")
            except:
                config_name = "sam2_hiera_l.yaml"
                print(f"Using fallback config: {config_name}")
            
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
                weights_only_path = None
                
                try:
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    # Check if it's a wrapped checkpoint
                    has_model_key = 'model' in checkpoint if isinstance(checkpoint, dict) else False
                except Exception as e:
                    print(f"Error loading checkpoint file: {str(e)}")
                    return False
                
                try:
                    # Don't extract weights for normal loading - build_sam2 expects the full checkpoint
                    weights_only_path = None
                    
                    # Only extract for manual loading attempt
                    if has_model_key:
                        weights_only_path = model_path.replace('.pth', '_weights.pth')
                        if not os.path.exists(weights_only_path):
                            print("Preparing weights for manual loading...")
                            torch.save(checkpoint['model'], weights_only_path)
                    
                    # Try loading SAM2 with the original checkpoint
                    print("Attempting to load SAM2 model...")
                    
                    # For SAM2.1, we need to handle the new checkpoint format
                    # Check if this is a SAM2.1 checkpoint by looking for new keys
                    if has_model_key and any(key in checkpoint['model'] for key in ['no_obj_embed_spatial', 'obj_ptr_tpos_proj.weight']):
                        print("Detected SAM2.1 checkpoint format, using compatibility mode...")
                        # Build the model without loading weights
                        sam2_model = build_sam2(config_name, None, device=self.device)
                        
                        # Load the state dict manually with strict=False to ignore unexpected keys
                        state_dict = checkpoint['model'] if has_model_key else checkpoint
                        # Filter out keys that don't exist in the model
                        model_state = sam2_model.state_dict()
                        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state}
                        
                        # Check for missing keys
                        missing_keys = set(model_state.keys()) - set(filtered_state_dict.keys())
                        unexpected_keys = set(state_dict.keys()) - set(model_state.keys())
                        
                        
                        sam2_model.load_state_dict(filtered_state_dict, strict=False)
                    else:
                        # Normal loading for older checkpoints
                        sam2_model = build_sam2(config_name, model_path, device=self.device)
                    
                    self.predictor = SAM2ImagePredictor(sam2_model)
                    print("SAM2 model loaded successfully")
                    self.model_type_loaded = "SAM2"
                    return True
                except Exception as build_error:
                    import traceback
                    print(f"Failed to load SAM2: {str(build_error)}")
                    
                    print("Full error traceback:")
                    import sys
                    traceback.print_exc(file=sys.stdout)
                    
                    # Let's check the actual error location
                    if "_load_checkpoint" in str(traceback.format_exc()) or "KeyError: 'model'" in str(traceback.format_exc()):
                        print("\nCheckpoint format issue detected:")
                        print(f"Model path: {model_path}")
                        print(f"Config: {config_name}")
                        
                        # The error shows SAM2 expects the checkpoint to work with weights_only=True
                        print("\nThe checkpoint format doesn't match what SAM2 expects.")
                        print("This might be an old checkpoint format.")
                        print("\nTo fix this:")
                        print(f"1. Delete the old checkpoint: {model_path}")
                        print("2. The new SAM2.1 checkpoint will be downloaded automatically")
                        print("\nOr manually delete and re-run:")
                        if platform.system() == "Windows":
                            print(f"   del \"{model_path}\"")
                        else:
                            print(f"   rm \"{model_path}\"")
                        
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
        # Updated URLs based on the official SAM2 repository
        urls = {
            "sam2_hiera_l.pth": "https://dl.fbaipublicfiles.com/segment_anything_2/10312024/sam2.1_hiera_large.pt",
            "sam2_hiera_b+.pth": "https://dl.fbaipublicfiles.com/segment_anything_2/10312024/sam2.1_hiera_base_plus.pt",
            # Alternative URLs
            "sam2_hiera_l_alt.pth": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_large.pt",
        }
        
        if checkpoint_name not in urls:
            print(f"Unknown SAM2 model: {checkpoint_name}")
            print(f"Available models: {list(urls.keys())}")
            raise ValueError(f"Unknown SAM2 model: {checkpoint_name}")
        
        # Try multiple URLs in case some are not accessible
        url_list = [
            urls[checkpoint_name],
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_large.pt",
            "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
        ]
        
        import urllib.request
        for url in url_list:
            try:
                print(f"Trying to download SAM2 model from {url}...")
                urllib.request.urlretrieve(url, model_path)
                print(f"SAM2 model downloaded successfully to {model_path}")
                return model_path
            except Exception as e:
                print(f"Failed with {url}: {str(e)}")
                continue
        
        # If all URLs fail
        print("\nAll download attempts failed. Please download the model manually.")
        print("Try one of these URLs:")
        for url in url_list:
            print(f"  - {url}")
        print(f"\nSave the file as: {model_path}")
        raise Exception("Failed to download SAM2 model from all URLs")
    
    def generate_mask_from_image(self, image, points=None, box=None, multimask_output=True, return_logits=False):
        """
        Generate a mask from an image using SAM
        
        Args:
            image: Input image (numpy array, HWC format, RGB)
            points: Points to consider for mask generation [x, y, label], label: 1 for foreground, 0 for background
            box: Bounding box for mask generation [x1, y1, x2, y2]
            multimask_output: Whether to return multiple masks
            return_logits: Whether to return raw logits for custom thresholding
        
        Returns:
            Generated mask, confidence score, and optionally logits
        """
        import numpy as np
        import torch
        
        if self.predictor is None:
            self.load_model()
        
        # Set the image for the predictor
        self.predictor.set_image(image)
        
        # Apply custom threshold if set (SAM uses 0.0 by default)
        if hasattr(self.predictor, 'model') and hasattr(self.predictor.model, 'mask_threshold'):
            self.predictor.model.mask_threshold = self.mask_threshold
        
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
            if return_logits:
                return masks[mask_idx], scores[mask_idx], logits[mask_idx]
            return masks[mask_idx], scores[mask_idx]
        else:
            if return_logits:
                return masks[0], scores[0], logits[0]
            return masks[0], scores[0]
    
    def apply_custom_threshold(self, logits, threshold=None):
        """
        Apply custom threshold to logits to generate binary mask
        
        Args:
            logits: Raw logits from SAM prediction
            threshold: Custom threshold (uses self.mask_threshold if None)
        
        Returns:
            Binary mask
        """
        import numpy as np
        
        if threshold is None:
            threshold = self.mask_threshold
        
        # Convert logits to binary mask using threshold
        mask = logits > threshold
        return mask.astype(np.uint8)
    
    def filter_mask_by_stability(self, mask, logits, stability_threshold=None):
        """
        Filter mask based on stability score to reduce shuttering
        
        Args:
            mask: Binary mask
            logits: Raw logits from SAM
            stability_threshold: Minimum stability score (uses self.stability_score_threshold if None)
        
        Returns:
            Filtered mask
        """
        import numpy as np
        import cv2
        
        if stability_threshold is None:
            stability_threshold = self.stability_score_threshold
        
        # Ensure mask is boolean type
        mask_bool = mask.astype(bool)
        
        # Check if logits need to be resized to match mask dimensions
        if logits.shape != mask.shape:
            # Resize logits to match mask size
            # Use bilinear interpolation for smooth transitions
            logits_resized = cv2.resize(logits.astype(np.float32), 
                                       (mask.shape[1], mask.shape[0]), 
                                       interpolation=cv2.INTER_LINEAR)
        else:
            logits_resized = logits
        
        # Calculate stability score (how confident the model is)
        # Areas where logits are close to threshold are less stable
        stability = np.abs(logits_resized - self.mask_threshold)
        
        # Create stability mask
        stable_mask = stability > (stability_threshold * self.stability_score_offset)
        stable_mask = stable_mask.astype(bool)
        
        # Apply stability filter to original mask using logical AND
        filtered_mask = np.logical_and(mask_bool, stable_mask)
        
        return filtered_mask.astype(np.uint8)
    
    def set_thresholds(self, mask_threshold=None, stability_threshold=None, stability_offset=None):
        """
        Set threshold values for mask generation
        
        Args:
            mask_threshold: Threshold for converting logits to binary mask (default: 0.0)
            stability_threshold: Minimum stability score (default: 0.95)
            stability_offset: Offset for stability calculation (default: 1.0)
        """
        if mask_threshold is not None:
            self.mask_threshold = mask_threshold
        if stability_threshold is not None:
            self.stability_score_threshold = stability_threshold
        if stability_offset is not None:
            self.stability_score_offset = stability_offset

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
