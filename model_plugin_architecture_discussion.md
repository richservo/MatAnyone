# MatAnyone Model Plugin Architecture Discussion

## Overview
This document captures the discussion about evolving MatAnyone from a single-model system to a modular architecture that can automatically integrate any video matting model without changing the existing GUI.

## The Vision
Create a system where users can:
1. Copy/paste a GitHub URL into the GUI
2. The system automatically analyzes and adapts the new model
3. Switch between models via a simple dropdown
4. No code changes required to the existing frontend

## Key Design Principles
- **Zero Frontend Changes**: The GUI processing logic remains untouched
- **Automatic Adaptation**: New models conform to our interface, not the other way around
- **Local LLM Processing**: Privacy-focused, no API keys required
- **Seamless Integration**: Models appear in dropdown once installed

## Architecture Components

### 1. Base Model Adapter Interface
Every model must implement this interface to work with the system:

```python
class BaseModelAdapter:
    def __init__(self, model_path="", **kwargs):
        pass
    
    def process_video(self, input_path, mask_path, output_path, 
                     n_warmup=10, r_erode=10, r_dilate=15, 
                     save_image=True, max_size=512,
                     bidirectional=False, blend_method='weighted', 
                     reverse_dilate=15, cleanup_temp=True,
                     suffix=None, video_codec='Auto', video_quality='High',
                     custom_bitrate=None, progress_callback=None, **kwargs):
        # Returns: (foreground_video_path, alpha_video_path)
        raise NotImplementedError
    
    def process_video_in_chunks(self, input_path, mask_path, output_path, 
                               num_chunks=1, progress_callback=None, **kwargs):
        return self.process_video(input_path, mask_path, output_path, 
                                progress_callback=progress_callback, **kwargs)
    
    def process_video_with_enhanced_chunking(self, input_path, mask_path, output_path, 
                                            num_chunks=2, progress_callback=None, **kwargs):
        return self.process_video(input_path, mask_path, output_path, 
                                progress_callback=progress_callback, **kwargs)
    
    def clear_memory(self):
        pass
    
    def request_interrupt(self):
        self.interrupt_requested = True
    
    def check_interrupt(self):
        if hasattr(self, 'interrupt_requested') and self.interrupt_requested:
            raise KeyboardInterrupt("Processing interrupted by user")
```

### 2. Model Registry System
Automatically discovers and manages available models:

```python
class ModelRegistry:
    def __init__(self):
        self.models = {
            'matanyone': MatAnyoneAdapter,  # Built-in
        }
        self.scan_plugins()
    
    def scan_plugins(self):
        """Scan plugins/ directory for installed models"""
        plugin_dir = Path("plugins")
        for model_dir in plugin_dir.iterdir():
            if model_dir.is_dir():
                adapter_path = model_dir / "adapter.py"
                if adapter_path.exists():
                    # Dynamically load adapter
                    self.load_adapter(model_dir.name, adapter_path)
    
    def get_model(self, model_name):
        return self.models.get(model_name)
    
    def list_models(self):
        return list(self.models.keys())
```

### 3. Intelligent Model Installer
Uses local LLM to automatically generate adapters:

```python
class ModelInstaller:
    def __init__(self):
        # Use CodeLlama 7B for local, private analysis
        self.llm = Llama(
            model_path="models/codellama-7b-instruct.Q4_K_M.gguf",
            n_ctx=4096,
            n_gpu_layers=-1
        )
    
    def install_from_github(self, github_url, progress_callback=None):
        # 1. Clone repository to plugins/ModelName/
        model_name = self.extract_model_name(github_url)
        clone_path = f"plugins/{model_name}"
        
        progress_callback(10, "Cloning repository...")
        subprocess.run(["git", "clone", github_url, clone_path])
        
        # 2. Analyze repository structure
        progress_callback(30, "Analyzing model code...")
        readme = self.read_file(f"{clone_path}/README.md")
        main_code = self.find_main_script(clone_path)
        
        # 3. Generate adapter using LLM
        progress_callback(50, "Generating adapter...")
        adapter_code = self.generate_adapter(readme, main_code)
        
        # 4. Test adapter
        progress_callback(70, "Testing integration...")
        if self.test_adapter(adapter_code):
            # 5. Save adapter
            progress_callback(90, "Installing model...")
            self.save_adapter(clone_path, adapter_code)
            progress_callback(100, "Model installed successfully!")
        else:
            raise Exception("Adapter test failed")
    
    def generate_adapter(self, readme, code):
        prompt = f"""
        Generate a Python adapter class that implements BaseModelAdapter interface.
        
        The model's README:
        {readme}
        
        Main code structure:
        {code}
        
        Generate an adapter that:
        1. Initializes the model
        2. Implements process_video() to match our interface
        3. Handles format conversions as needed
        """
        
        response = self.llm(prompt, max_tokens=2000)
        return response['choices'][0]['text']
```

### 4. Modified InferenceCore (Minimal Change)
The only change to existing code - adding model type support:

```python
class InterruptibleInferenceCore:
    def __init__(self, model_path="", model_type="matanyone", **kwargs):
        if model_type == "matanyone":
            # Existing MatAnyone code path (unchanged)
            from matanyone.inference import InferenceCore
            self.model = InferenceCore(model_path, **kwargs)
        else:
            # New plugin system
            from adapters.model_registry import ModelRegistry
            registry = ModelRegistry()
            adapter_class = registry.get_model(model_type)
            if adapter_class:
                self.model = adapter_class(model_path, **kwargs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        # Rest of the class remains exactly the same...
```

### 5. GUI Integration (5-10 lines total)
The only GUI changes needed:

```python
# In gui_widgets.py - Add model dropdown
def create_model_frame(self):
    # Existing code...
    
    # Add model selection dropdown
    self.model_label = QLabel("Model:")
    self.model_dropdown = QComboBox()
    self.model_dropdown.addItems(self.get_available_models())
    
    # Add "Add Model" button
    self.add_model_btn = QPushButton("Add Model")
    self.add_model_btn.clicked.connect(self.add_model_dialog)

# In gui_processing.py - Pass selected model
def setup_processor(self):
    selected_model = self.model_dropdown.currentText()
    self.processor = InterruptibleInferenceCore(
        model_type=selected_model  # This is the only new parameter
    )

# New method for adding models
def add_model_dialog(self):
    url, ok = QInputDialog.getText(
        self, "Add Model", 
        "Enter GitHub URL:",
        QLineEdit.Normal,
        "https://github.com/..."
    )
    if ok and url:
        # Launch installer in background
        self.install_model(url)
```

## Example: How a New Model Gets Integrated

Let's say someone wants to use "SuperMatter" model:

1. **User Action**: Pastes `https://github.com/awesome/SuperMatter` and clicks Install

2. **Automatic Analysis**: The installer reads the README and finds:
   ```python
   # Example from README
   model = SuperMatter.load_pretrained()
   result = model.process(video_frames, mask, config={'quality': 'high'})
   ```

3. **Generated Adapter**:
   ```python
   class SuperMatterAdapter(BaseModelAdapter):
       def __init__(self, model_path="", **kwargs):
           import SuperMatter
           self.model = SuperMatter.load_pretrained()
       
       def process_video(self, input_path, mask_path, output_path, **kwargs):
           # Load video
           frames = self.load_video_frames(input_path)
           mask = Image.open(mask_path)
           
           # Process with SuperMatter
           config = {'quality': 'high'}
           result = self.model.process(frames, mask, config)
           
           # Convert to expected output format
           fg_path = self.save_video(result.foreground, output_path, "_fg")
           alpha_path = self.save_video(result.alpha, output_path, "_alpha")
           
           return (fg_path, alpha_path)
   ```

4. **Result**: SuperMatter appears in the model dropdown, works exactly like MatAnyone

## Benefits

1. **No Frontend Changes**: GUI code stays pristine
2. **Future-Proof**: Any new model can be added
3. **Privacy-First**: Local LLM, no data leaves machine
4. **User-Friendly**: Copy/paste GitHub URL, done
5. **Maintainable**: Clear separation of concerns

## Technical Requirements

- **Local LLM**: CodeLlama 7B (6-8GB VRAM)
- **Python Libraries**: llama-cpp-python for LLM inference
- **Storage**: ~5GB for LLM model weights
- **No API Keys**: Completely offline after LLM download

## Implementation Priority

1. Create base adapter interface
2. Implement model registry
3. Create MatAnyone adapter (wrap existing code)
4. Add model selection to GUI (minimal change)
5. Implement model installer with LLM
6. Test with various model types

## Summary

This architecture allows MatAnyone to evolve from a single-model tool to a universal video matting framework. Users can integrate any model by simply providing a GitHub URL, and the system automatically handles the rest. The existing GUI remains unchanged, preserving all current functionality while gaining unlimited extensibility.