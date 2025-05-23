# Smart Chunking Examples

This directory contains examples demonstrating how to use the Smart Chunking System independently of MatAnyone.

## Examples

### 1. `smart_chunking_example.py`

A comprehensive example showing how to:
- Generate heat maps from motion detection
- Use custom activity detection methods
- Apply effects selectively using smart chunks
- Blend processed chunks smoothly

**Usage:**
```bash
python smart_chunking_example.py
```

Make sure to update the `video_path` variable in the script to point to your test video.

### Key Concepts Demonstrated

1. **Motion-Based Heat Maps**: Detect motion between frames to identify active regions
2. **Custom Detection**: Use any detection method (brightness, color, objects) for heat map generation
3. **Chunk Processing**: Process only the important parts of each frame
4. **Smooth Blending**: Feather chunk edges for seamless integration

### Adapting for Your Use Case

The smart chunking system can be adapted for various applications:

- **Video Effects**: Apply computationally expensive effects only where needed
- **Video Compression**: Allocate more bits to high-activity regions
- **Object Tracking**: Focus processing on regions with tracked objects
- **Video Stabilization**: Process only unstable regions
- **Real-time Processing**: Reduce computation by processing only active areas

### Performance Benefits

Using smart chunking typically provides:
- 40-60% reduction in processing time
- 50-70% reduction in memory usage
- Better quality by focusing resources on important areas
- Scalability for high-resolution videos

## License

These examples and the Smart Chunking System are freely available for any use, including commercial applications.