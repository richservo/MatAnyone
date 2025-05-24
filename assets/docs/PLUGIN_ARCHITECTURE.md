# Plugin Architecture and Model Adaptation Guide

MatAnyone now features a clean, modular plugin architecture that makes it easy to adapt other video processing models while maintaining the core MatAnyone functionality and enhanced chunking system.

## Overview

The codebase is organized with MatAnyone as a plugin in `/plugins/MatAnyone/`, keeping the core functionality separate from the main application. This modular approach provides several benefits:

- **Clean separation of concerns**: Core processing logic is separate from model-specific code
- **Easy updates**: The MatAnyone plugin can be updated independently
- **Extensibility**: New models can be added as plugins without modifying core code
- **Maintainability**: Each model is self-contained with its own dependencies

## Current Structure

```
MatAnyone/
├── core/                    # Core processing engine
│   ├── inference_core.py    # Main inference wrapper
│   ├── enhanced_chunk_processor.py  # Advanced chunking system
│   └── checkpoint_processor.py
├── plugins/
│   └── MatAnyone/          # MatAnyone model plugin
│       ├── adapter.py      # Model adapter implementation
│       ├── adapter_metadata.json
│       ├── matanyone/      # Original MatAnyone code
│       └── hugging_face/   # HuggingFace integration
├── ui/                     # GUI components
├── chunking/              # Chunking utilities
├── mask/                  # Mask processing
└── utils/                 # Shared utilities
```

## How to Adapt a New Model

If you want to experiment with adapting other video processing models, you can create a new plugin following this structure:

### 1. Create Plugin Directory

```bash
mkdir -p plugins/YourModel
cd plugins/YourModel
```

### 2. Create Adapter Metadata

Create `adapter_metadata.json`:

```json
{
    "name": "YourModel",
    "version": "1.0.0",
    "description": "Description of your model",
    "author": "Your Name",
    "model_type": "video_processing",
    "category": "matting",
    "github_url": "https://github.com/your/repo",
    "requirements": [
        "torch>=1.8.0",
        "torchvision"
    ]
}
```

### 3. Implement Base Adapter

Create `adapter.py` implementing the required interface:

```python
from pathlib import Path
import sys
import os

class YourModelAdapter:
    """Adapter for YourModel integration with MatAnyone"""
    
    def __init__(self, model_path="", **kwargs):
        """Initialize your model"""
        # Add the plugin directory to path for imports
        plugin_dir = Path(__file__).parent
        if str(plugin_dir) not in sys.path:
            sys.path.insert(0, str(plugin_dir))
        
        # Initialize your model here
        # from your_model import YourModel
        # self.model = YourModel(model_path)
        pass
    
    @staticmethod
    def get_model_info():
        """Return model information"""
        return {
            'name': 'YourModel',
            'version': '1.0.0',
            'description': 'Your model description',
            'supported_formats': ['mp4', 'avi', 'mov'],
            'requires_mask': True
        }
    
    def process_video_with_enhanced_chunking(self, input_path, mask_path, output_dir, **kwargs):
        """
        Main processing method that integrates with MatAnyone's enhanced chunking system.
        
        This method should handle:
        - Video input/output
        - Mask processing
        - Integration with chunking parameters
        - Progress reporting
        
        Args:
            input_path: Path to input video
            mask_path: Path to mask image/video
            output_dir: Output directory
            **kwargs: Additional processing parameters
            
        Returns:
            tuple: (foreground_path, alpha_path) or (output_path, output_path) for inpainting
        """
        
        # Your model processing logic here
        # This is where you integrate your model with the MatAnyone pipeline
        
        # Example structure:
        # 1. Load and preprocess input
        # 2. Process with your model
        # 3. Save results in MatAnyone format
        # 4. Return output paths
        
        foreground_path = os.path.join(output_dir, "output_foreground.mp4")
        alpha_path = os.path.join(output_dir, "output_alpha.mp4")
        
        # Implement your processing here
        
        return foreground_path, alpha_path
    
    def process_frame(self, frame, mask=None, **kwargs):
        """
        Process a single frame (used by chunking system)
        
        Args:
            frame: Input frame (numpy array)
            mask: Mask for the frame (numpy array, optional)
            **kwargs: Additional parameters
            
        Returns:
            Processed frame or tuple of (foreground, alpha)
        """
        # Implement single frame processing
        pass
    
    def cleanup(self):
        """Clean up resources"""
        pass
```

### 4. Organize Model Code

Place your model's code in subdirectories:

```
plugins/YourModel/
├── adapter.py
├── adapter_metadata.json
├── your_model/           # Your model's code
│   ├── __init__.py
│   ├── model.py
│   └── utils.py
├── weights/              # Model weights (gitignored)
└── requirements.txt      # Model-specific requirements
```

### 5. Key Integration Points

When adapting a model, focus on these integration points:

#### Enhanced Chunking Integration
Your adapter should work with MatAnyone's enhanced chunking system:
- The system will call your `process_video_with_enhanced_chunking` method
- You'll receive chunk parameters and should respect them
- Return results in the expected format

#### Mask Handling
- Input masks are typically binary (0/255) images
- Your model should handle mask preprocessing as needed
- Some models expect inverted masks - handle this in your adapter

#### Progress Reporting
Integrate with the progress system by accepting callback functions:
```python
def process_video_with_enhanced_chunking(self, input_path, mask_path, output_dir, 
                                       progress_callback=None, **kwargs):
    if progress_callback:
        progress_callback(25, "Loading model...")
    # ... processing ...
    if progress_callback:
        progress_callback(100, "Complete")
```

#### File Format Compatibility
Ensure your outputs are compatible with MatAnyone's expected formats:
- Foreground videos with alpha transparency
- Alpha matte videos as grayscale
- High-quality MP4 encoding

## Example: Inpainting Model Adaptation

For video inpainting models (like ProPainter), the adapter pattern would be:

```python
def process_video_with_enhanced_chunking(self, input_path, mask_path, output_dir, **kwargs):
    # Inpainting models typically:
    # 1. Take video + mask -> output inpainted video
    # 2. Return the same path for both foreground and alpha
    
    output_path = os.path.join(output_dir, "inpainted_video.mp4")
    
    # Process with your inpainting model
    your_inpainting_function(input_path, mask_path, output_path)
    
    # For inpainting, return the same path for both outputs
    return output_path, output_path
```

## Testing Your Adapter

1. **Unit Testing**: Test individual methods in isolation
2. **Integration Testing**: Test with the full MatAnyone pipeline
3. **Chunking Testing**: Verify it works with different chunk configurations
4. **Memory Testing**: Ensure it handles large videos without memory issues

## Best Practices

### Code Organization
- Keep your model code self-contained in the plugin directory
- Use relative imports within your plugin
- Don't modify core MatAnyone files

### Error Handling
- Implement robust error handling and cleanup
- Provide meaningful error messages
- Handle GPU/CPU fallback gracefully

### Performance
- Leverage MatAnyone's chunking system for large videos
- Implement efficient memory management
- Consider GPU memory constraints

### Documentation
- Document your adapter's specific requirements
- Provide example usage and parameters
- Include troubleshooting information

## Advanced Features

### Custom Chunking Strategies
You can implement custom chunking logic for your specific model:

```python
def get_optimal_chunk_size(self, video_resolution):
    """Return optimal chunk size for this model"""
    # Your logic here
    return chunk_width, chunk_height

def requires_special_chunking(self):
    """Return True if this model needs special chunking handling"""
    return False
```

### Model-Specific Parameters
Add model-specific GUI controls by extending the adapter interface:

```python
@staticmethod
def get_ui_parameters():
    """Return model-specific UI parameters"""
    return {
        'neighbor_length': {'type': 'int', 'default': 10, 'range': (5, 30)},
        'mask_dilation': {'type': 'int', 'default': 4, 'range': (0, 20)},
        'use_optical_flow': {'type': 'bool', 'default': True}
    }
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure your plugin directory is in the Python path
2. **Memory Issues**: Use the chunking system for large videos
3. **Format Issues**: Ensure output formats match expectations
4. **GPU Issues**: Handle CUDA availability gracefully

### Debug Mode
Enable debug output in your adapter:

```python
def __init__(self, model_path="", debug=False, **kwargs):
    self.debug = debug
    if self.debug:
        print(f"Initializing {self.__class__.__name__} with {kwargs}")
```

## Contributing

If you successfully adapt a new model and think it would benefit others:

1. Test thoroughly with various video types and sizes
2. Document the adaptation process
3. Consider contributing back to the community
4. Follow the established plugin structure and conventions

## Conclusion

The plugin architecture makes MatAnyone extensible while maintaining its core strengths. By following this guide, you can adapt other video processing models to work within the MatAnyone ecosystem, benefiting from its enhanced chunking system, GUI, and robust video processing pipeline.

Remember that this is primarily intended for experimentation and advanced users who want to explore different models. The core MatAnyone functionality remains the primary, well-tested path for video matting tasks.