import sys
from pathlib import Path

# Add paths for imports
plugin_dir = Path(__file__).parent
sys.path.insert(0, str(plugin_dir.parent.parent))  # MatAnyone main directory
sys.path.insert(0, str(plugin_dir))  # Plugin directory (cloned MatAnyone)

from adapters.base_adapter import BaseModelAdapter
from typing import Tuple, Optional, Callable, Dict, Any

class MatanyoneAdapter(BaseModelAdapter):
    def __init__(self, model_path: str = "", **kwargs):
        super().__init__(model_path, **kwargs)
        
        try:
            # Import the MatAnyone inference core from the cloned repository
            from matanyone.inference.inference_core import InferenceCore
            import hydra
            from omegaconf import DictConfig
            import torch
            
            # Clear any existing Hydra instance
            if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
                hydra.core.global_hydra.GlobalHydra.instance().clear()
            
            # Initialize with default config and use local weights
            self.inference_core = InferenceCore()
            
            # Look for weights in plugin directory first, fallback to HF if not found
            plugin_weights = plugin_dir / "matanyone" / "pretrained_models" / "matanyone.pth"
            if plugin_weights.exists():
                model_to_load = str(plugin_weights)
            else:
                model_to_load = model_path or "PeiqingYang/MatAnyone"
            
            self.inference_core.load_model(model_to_load)
            print("MatAnyone adapter initialized successfully")
        except Exception as e:
            print(f"Failed to initialize MatAnyone: {e}")
            import traceback
            traceback.print_exc()
            self.inference_core = None
    
    def process_video(self, input_path: str, mask_path: str, output_path: str, **kwargs) -> Tuple[str, str]:
        if self.inference_core is None:
            raise RuntimeError("MatAnyone not initialized")
        
        # Use the built-in inference core
        return self.inference_core.process_video(
            input_path=input_path,
            mask_path=mask_path,
            output_path=output_path,
            **kwargs
        )
    
    def clear_memory(self):
        if hasattr(self.inference_core, 'clear_memory'):
            self.inference_core.clear_memory()
    
    @classmethod
    def get_model_info(cls) -> Dict[str, Any]:
        return {
            'name': 'MatAnyone',
            'version': '1.0',
            'description': 'High-quality video matting with temporal consistency',
            'requires_gpu': False,
            'min_gpu_memory': 4,
            'devices': ['cuda', 'mps', 'cpu']
        }
