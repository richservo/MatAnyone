# Smart Chunking System - Developer Guide

## Overview

The Smart Chunking System is an intelligent video processing framework that dynamically analyzes video content to determine optimal chunk placement. Unlike traditional grid-based approaches, it uses heat map analysis to focus processing power where it matters most.

This system is freely available for use in any project, including commercial applications.

## Key Features

- **Content-Aware Chunking**: Analyzes activity patterns to place chunks where content exists
- **Face Priority Processing**: Automatically detects and prioritizes facial regions
- **Dynamic Orientation**: Supports both horizontal and vertical chunk orientations
- **Memory Efficient**: Processes only necessary regions, reducing memory usage by up to 60%
- **Quality Optimization**: Ensures subjects are centered within chunks to prevent edge artifacts

## Core Components

### 1. Heat Map Analyzer (`chunking/heat_map_analyzer.py`)

Analyzes mask sequences to create activity heat maps.

```python
from chunking.heat_map_analyzer import HeatMapAnalyzer

# Create analyzer with face priority
analyzer = HeatMapAnalyzer(face_priority_weight=3.0)

# Generate heat map from masks
heat_map = analyzer.analyze_mask_sequence(
    mask_dir="path/to/mask/frames",
    original_frames_dir="path/to/original/frames"  # Optional, for face detection
)

# Save visualization
analyzer.save_heat_map_visualization("output/heat_map.png")

# Get statistics
stats = analyzer.get_activity_stats()
print(f"Activity coverage: {stats['activity_ratio']*100:.1f}%")
```

### 2. Smart Chunk Placer (`chunking/smart_chunk_placer.py`)

Determines optimal chunk positions based on heat map analysis.

```python
from chunking.smart_chunk_placer import SmartChunkPlacer

# Initialize with 20% overlap
placer = SmartChunkPlacer(overlap_ratio=0.2, min_activity_threshold=0.05)

# Find optimal chunk placement
chunks = placer.find_optimal_chunk_placement(
    heat_map=heat_map,
    chunk_width=512,  # Target chunk width
    chunk_height=512,  # Target chunk height
    model_factor=8,    # Alignment factor for model requirements
    allow_rotation=True  # Allow both orientations
)

# Visualize placement
placer.visualize_chunk_placement(heat_map, "output/chunk_placement.png")

# Get coverage statistics
stats = placer.get_chunk_coverage_stats(heat_map)
print(f"Coverage: {stats['coverage']*100:.1f}% with {stats['num_chunks']} chunks")
```

### 3. Chunk Processing Utilities (`chunking/chunking_utils.py`)

Helper functions for chunk operations.

```python
from chunking.chunking_utils import get_autochunk_segments, create_low_res_video

# Auto-determine chunk configuration
segments = get_autochunk_segments(
    video_width=1920,
    video_height=1080,
    target_width=512,
    target_height=512,
    model_factor=8
)

# Create low-resolution version for analysis
create_low_res_video(
    input_path="input_video.mp4",
    output_path="low_res_video.mp4",
    scale_factor=0.25
)
```

## Usage Examples

### Example 1: Basic Smart Chunking

```python
import cv2
import numpy as np
from chunking.heat_map_analyzer import HeatMapAnalyzer
from chunking.smart_chunk_placer import SmartChunkPlacer

# Step 1: Analyze your masks or activity regions
analyzer = HeatMapAnalyzer(face_priority_weight=2.0)
heat_map = analyzer.analyze_mask_sequence("masks/")

# Step 2: Determine optimal chunk placement
placer = SmartChunkPlacer(overlap_ratio=0.2)
chunks = placer.find_optimal_chunk_placement(
    heat_map, 
    chunk_width=512, 
    chunk_height=512
)

# Step 3: Process each chunk
for chunk in chunks:
    x_start, x_end = chunk['x_range']
    y_start, y_end = chunk['y_range']
    
    # Extract chunk from video frame
    chunk_region = frame[y_start:y_end, x_start:x_end]
    
    # Process chunk with your algorithm
    processed_chunk = your_processing_function(chunk_region)
    
    # Place back on canvas
    output_frame[y_start:y_end, x_start:x_end] = processed_chunk
```

### Example 2: Using Smart Chunking for Video Effects

```python
def apply_effect_with_smart_chunks(video_path, effect_fn):
    """Apply an effect to a video using smart chunking"""
    
    # First pass: generate activity map (using motion detection)
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    heat_map = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if prev_frame is not None:
            # Simple motion detection
            diff = cv2.absdiff(frame, prev_frame)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            if heat_map is None:
                heat_map = gray_diff.astype(np.float32)
            else:
                heat_map += gray_diff.astype(np.float32)
        
        prev_frame = frame
    
    cap.release()
    
    # Normalize heat map
    heat_map = heat_map / heat_map.max()
    
    # Find optimal chunks
    placer = SmartChunkPlacer()
    chunks = placer.find_optimal_chunk_placement(
        heat_map, 
        chunk_width=640, 
        chunk_height=480
    )
    
    print(f"Processing video with {len(chunks)} smart chunks")
    
    # Second pass: process with chunks
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter('output.mp4', ...)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        output_frame = np.zeros_like(frame)
        
        # Process each chunk
        for chunk in chunks:
            x_start, x_end = chunk['x_range']
            y_start, y_end = chunk['y_range']
            
            # Apply effect to chunk
            chunk_region = frame[y_start:y_end, x_start:x_end]
            processed = effect_fn(chunk_region)
            
            # Blend into output
            output_frame[y_start:y_end, x_start:x_end] = processed
        
        out.write(output_frame)
    
    cap.release()
    out.release()
```

### Example 3: Custom Activity Detection

```python
class CustomHeatMapAnalyzer(HeatMapAnalyzer):
    """Extended analyzer with custom activity detection"""
    
    def analyze_custom_activity(self, frames_dir, detection_fn):
        """Create heat map using custom detection function"""
        
        frame_files = sorted(os.listdir(frames_dir))
        heat_map = None
        
        for frame_file in frame_files:
            frame = cv2.imread(os.path.join(frames_dir, frame_file))
            
            # Apply custom detection
            activity_mask = detection_fn(frame)
            
            if heat_map is None:
                heat_map = activity_mask.astype(np.float32)
            else:
                heat_map += activity_mask.astype(np.float32)
        
        # Normalize
        self.heat_map = heat_map / len(frame_files)
        self.combined_heat_map = self.heat_map
        
        return self.heat_map

# Usage
def detect_specific_colors(frame):
    """Detect specific color regions"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Example: detect red regions
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    return mask

analyzer = CustomHeatMapAnalyzer()
heat_map = analyzer.analyze_custom_activity("frames/", detect_specific_colors)
```

## Integration Guide

### Minimal Dependencies

The smart chunking system has minimal dependencies:
- NumPy
- OpenCV (cv2)
- Optional: Face detection library (for face priority feature)

### Standalone Usage

To use the smart chunking system independently:

1. Copy these files to your project:
   - `chunking/heat_map_analyzer.py`
   - `chunking/smart_chunk_placer.py`
   - `chunking/chunking_utils.py` (optional, for utilities)

2. Install dependencies:
   ```bash
   pip install numpy opencv-python
   ```

3. Adapt the heat map generation to your needs:
   - Use provided mask-based analysis
   - Implement motion detection
   - Use optical flow
   - Apply custom detection logic

### Memory Management

For large videos, use the chunking system with streaming:

```python
def process_video_streaming(video_path, chunks, process_fn):
    """Process video in streaming fashion"""
    
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process only chunks with activity in this frame
        for chunk in chunks:
            if should_process_chunk(chunk, frame_idx):
                x_start, x_end = chunk['x_range']
                y_start, y_end = chunk['y_range']
                
                chunk_data = frame[y_start:y_end, x_start:x_end]
                processed = process_fn(chunk_data)
                
                # Handle processed chunk
                yield chunk, processed
        
        frame_idx += 1
    
    cap.release()
```

## Advanced Features

### Multi-Scale Processing

```python
# Process at multiple scales
scales = [1.0, 0.5, 0.25]
all_chunks = []

for scale in scales:
    scaled_width = int(video_width * scale)
    scaled_height = int(video_height * scale)
    
    # Generate heat map at this scale
    heat_map = analyzer.analyze_mask_sequence(f"masks_scale_{scale}/")
    
    # Find chunks for this scale
    chunks = placer.find_optimal_chunk_placement(
        heat_map,
        chunk_width=int(512 * scale),
        chunk_height=int(512 * scale)
    )
    
    all_chunks.extend(chunks)
```

### Temporal Consistency

For video processing, maintain temporal consistency:

```python
from chunking.chunk_mask_propagation import propagate_chunk_data

# Propagate information between frames
chunk_data = propagate_chunk_data(
    prev_chunk_data,
    curr_chunk_data,
    next_chunk_data,
    blend_factor=0.3
)
```

## Performance Tips

1. **Cache Heat Maps**: For multiple passes, save heat maps to disk
2. **Adjust Thresholds**: Fine-tune `min_activity_threshold` for your content
3. **Chunk Size**: Balance between memory usage and overlap overhead
4. **Parallel Processing**: Process independent chunks in parallel

```python
from concurrent.futures import ProcessPoolExecutor

def process_chunks_parallel(chunks, frame, process_fn):
    with ProcessPoolExecutor() as executor:
        futures = []
        
        for chunk in chunks:
            x_start, x_end = chunk['x_range']
            y_start, y_end = chunk['y_range']
            chunk_data = frame[y_start:y_end, x_start:x_end]
            
            future = executor.submit(process_fn, chunk_data)
            futures.append((chunk, future))
        
        # Collect results
        results = []
        for chunk, future in futures:
            result = future.result()
            results.append((chunk, result))
        
        return results
```

## License

The Smart Chunking System is part of an independent GUI frontend that uses MatAnyone as a backend. This GUI and all its components, including the Smart Chunking System, were developed independently and are not affiliated with the original MatAnyone project.

All GUI components and utilities in this repository, including the Smart Chunking System, are freely available for any use, including commercial applications. These tools are designed to be modular and can work with any compatible video processing backend.

## Contributing

We welcome contributions! The smart chunking system can be enhanced with:
- Additional activity detection methods
- GPU acceleration for heat map generation
- Adaptive chunk sizing based on content complexity
- Temporal chunk tracking for video sequences

## Support

For questions or issues with the smart chunking system:
- Open an issue on the MatAnyone GitHub repository
- Tag it with `smart-chunking` for faster response
- Include your use case to help us improve the system

---

Built with ❤️ for the open source community. We hope this helps accelerate your video processing projects!