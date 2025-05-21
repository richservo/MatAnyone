# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MatAnyone is a video matting framework that extracts foreground objects from videos using a segmentation mask. It supports:

- Processing videos to extract foreground objects with alpha transparency
- Applying segmentation masks to identify specific objects in videos
- Multiple target assignment for matting different objects
- Interactive GUI for easy video processing
- Enhanced chunking strategies for processing long or high-resolution videos

## Repository Structure

The repository is organized into a modular folder structure:

1. **Root Directory**:
   - `matanyone_gui.py`: Main entry point for the graphical interface
   - `inference_matanyone.py`: Command-line script for MatAnyone inference
   - `inference_hf.py`: Hugging Face integration for inference

2. **Core Processing (`/core`)**:
   - `inference_core.py`: Core inference functionality
   - `chunk_processor.py`: Base chunk processing
   - `enhanced_chunk_processor.py`: Advanced chunk processing
   - `checkpoint_processor.py`: Checkpoint processing logic

3. **Mask Operations (`/mask`)**:
   - `mask_generator.py`: Tools for creating masks
   - `sam_generator.py`: Segment Anything Model integration
   - `mask_utils.py`: Mask manipulation utilities
   - `mask_analysis.py`: Mask analysis tools
   - `mask_operations.py`: Mask transformation operations
   - `mask_enhancement.py`: Mask quality enhancements

4. **UI Components (`/ui`)**:
   - `gui_config.py`: Configuration management for GUI
   - `gui_events.py`: Event handling logic for GUI
   - `gui_processing.py`: Processing logic for GUI
   - `gui_widgets.py`: UI widget creation and management
   - `ui_components.py`: Reusable UI components
   - `mask_generator_ui_main.py`: Mask generator interface
   - `mask_ui_base.py`, `mask_ui_editor.py`, etc.: Mask editing UI components

5. **Chunking Operations (`/chunking`)**:
   - `chunking_utils.py`: Utilities for chunking operations
   - `chunk_mask_analysis.py`: Analysis of masks within chunks
   - `chunk_mask_propagation.py`: Propagation of mask data between chunks
   - `chunk_optimizer.py`: Optimization of chunk operations
   - `chunk_visualization.py`: Visualization of chunk operations
   - `chunk_weight_optimization.py`: Optimization of reassembly weights

6. **Utilities (`/utils`)**:
   - `video_utils.py`: Video manipulation utilities
   - `image_processing.py`: Image processing utilities
   - `extraction_utils.py`: Data extraction utilities
   - `reassembly_utils.py`: Functions for reassembling processed chunks
   - `memory_utils.py`: Memory management utilities

7. **Tests (`/tests`)**:
   - `test_enhanced_chunk_processor.py`: Tests for enhanced chunk processor
   - `test_autochunk_script.py`: Tests for automatic chunking

## Git Operations

### Git Commits and Pushes

When making changes to the repository, follow these steps for git commits:

1. Check status of changes:
   ```bash
   git status
   ```

2. Add changes to staging area:
   ```bash
   git add <filename>  # For specific files
   git add .           # For all changes
   ```

3. Commit changes with a descriptive message:
   ```bash
   git commit -m "Brief description of changes"
   ```

4. Push changes to the remote repository:
   ```bash
   git push origin main
   ```

If you encounter authentication issues:
- Make sure you have the right credentials configured
- For HTTPS, you may need to use a Personal Access Token if password authentication is disabled
- For SSH, ensure your SSH keys are properly set up

### Managing Branches

1. View all branches:
   ```bash
   git branch
   ```

2. Switch to a different branch:
   ```bash
   git checkout <branch-name>
   ```

3. Create and switch to a new branch:
   ```bash
   git checkout -b <new-branch-name>
   ```

4. Merge changes from another branch:
   ```bash
   git checkout main
   git merge <branch-name>
   ```

## Backup and Versioning Process

### File Backups

When starting a new Claude Code session, create a backup of the current working scripts:

1. Determine the next version number by checking the highest version in `/Volumes/Storage/Richard/MatAnyone/BKUP/`
   ```bash
   find /Volumes/Storage/Richard/MatAnyone/BKUP -maxdepth 1 -type d -name "v*" | sort -V | tail -1
   ```
   - Use the next sequential number (e.g., v054 if v053 is the highest)

2. Create a new directory for the backup with proper subdirectories:
   ```bash
   mkdir -p /Volumes/Storage/Richard/MatAnyone/BKUP/v054/{core,mask,ui,chunking,utils}
   ```

3. Copy the current working scripts to this backup directory, maintaining folder structure:
   ```bash
   # Copy root files
   cp /Volumes/Storage/Richard/MatAnyone/*.py /Volumes/Storage/Richard/MatAnyone/BKUP/v054/
   
   # Copy module files to their respective directories
   cp /Volumes/Storage/Richard/MatAnyone/core/*.py /Volumes/Storage/Richard/MatAnyone/BKUP/v054/core/
   cp /Volumes/Storage/Richard/MatAnyone/mask/*.py /Volumes/Storage/Richard/MatAnyone/BKUP/v054/mask/
   cp /Volumes/Storage/Richard/MatAnyone/ui/*.py /Volumes/Storage/Richard/MatAnyone/BKUP/v054/ui/
   cp /Volumes/Storage/Richard/MatAnyone/chunking/*.py /Volumes/Storage/Richard/MatAnyone/BKUP/v054/chunking/
   cp /Volumes/Storage/Richard/MatAnyone/utils/*.py /Volumes/Storage/Richard/MatAnyone/BKUP/v054/utils/
   
   # Also copy any non-Python files that are needed (like the TCL theme)
   cp /Volumes/Storage/Richard/MatAnyone/ui/azure_dark_tcl.tcl /Volumes/Storage/Richard/MatAnyone/BKUP/v054/ui/
   ```

This backup approach ensures:
- A stable reference point exists at the start of each development session
- If changes introduce issues, there's a known working version to revert to
- Only create a new backup folder when starting a session with working code, not for individual changes
- The backup maintains the same folder structure as the original project for easier reference

4. After making any significant changes to the codebase, ensure a backup is created before committing changes.

5. When multiple files are being modified together for a feature or bug fix, they should all be backed up to the same version directory.

### File Version Headers

When updating a file, update the header comment at the top of the file:

1. Update the version number in the format `vX.YYYYYYYYY` where:
   - X is the major version (currently 1)
   - YYYYYYYYY is a 9-digit number derived from the epoch timestamp

2. Update the timestamp with the current date and time.

3. Add detailed comments about the changes made in this version.

Example header format:
```python
# filename.py - v1.123456789
# Updated: Day, Month DD, YYYY at HH:MM:SS TZ
# Changes in this version:
# - Added feature X
# - Improved performance of Y
# - Fixed bug in Z
```

To get the current epoch timestamp in Python (can be used to generate the version number):
```python
import time
int(time.time())
```

## Running the Application

### Running the GUI

Launch the MatAnyone GUI with:

```bash
python matanyone_gui.py
```

You can also provide initial paths as command-line arguments:

```bash
python matanyone_gui.py --input INPUT_VIDEO --mask MASK_PATH --output OUTPUT_DIRECTORY
```

### Command-line Inference

Process videos directly from the command line:

```bash
python inference_matanyone.py -i INPUT_VIDEO -m MASK_PATH -o OUTPUT_DIRECTORY
```

Additional options:
- `--warmup N`: Number of warmup iterations (default: 10)
- `--erode_kernel SIZE`: Erosion kernel size (default: 10)
- `--dilate_kernel SIZE`: Dilation kernel size (default: 10)
- `--suffix STRING`: Output suffix for different targets
- `--save_image`: Save individual frames (default: False)
- `--max_size N`: Limit maximum size (-1 means no limit)

### Launching the Hugging Face Demo

Run the interactive demo locally:

```bash
cd hugging_face
python app.py
```

## Development Guide

### Environment Setup

1. Create and activate conda environment:
```bash
conda create -n matanyone python=3.8 -y
conda activate matanyone
```

2. Install dependencies:
```bash
pip install -e .
```

3. For Hugging Face demo:
```bash
pip install -r hugging_face/requirements.txt
```

### Testing

When implementing changes:

1. Always test changes in the GUI after implementation to verify functionality and user experience:
```bash
python matanyone_gui.py
```

2. Test core functionality with sample videos through command-line for faster validation:
```bash
python inference_matanyone.py -i inputs/video/test-sample1.mp4 -m inputs/mask/test-sample1.png
```

3. Test with different resolution videos:
```bash
python inference_matanyone.py -i inputs/video/test-sample2.mp4 -m inputs/mask/test-sample2.png
```

4. Test with longer videos:
```bash
python inference_matanyone.py -i inputs/video/test-sample3.mp4 -m inputs/mask/test-sample3.png
```

5. For GUI-related changes, verify functionality across all affected components of the interface.

### Important Implementation Details

1. **Enhanced Chunking System**:
   - Used for processing long or high-resolution videos
   - Videos are split into chunks, processed independently, then reassembled
   - `enhanced_chunk_processor.py` handles chunk processing with mask propagation
   - Ensures continuity across chunk boundaries with special blending strategies

2. **Mask Generation and Processing**:
   - First-frame masks can be loaded from file or generated with SAM/SAM2
   - Masks can be eroded/dilated to optimize performance
   - `mask_enhancement.py` contains utilities for optimizing masks

3. **Memory Management**:
   - For high-resolution videos, use chunk processing to avoid OOM errors
   - `memory_utils.py` contains functions to clean up GPU memory
   - Each chunk is processed separately to reduce memory usage

4. **GUI Architecture**:
   - Main app class in `matanyone_gui.py` coordinates all components
   - Config management in `gui_config.py` handles persistent settings
   - Event handling in `gui_events.py` processes user interactions
   - Widget creation in `gui_widgets.py` builds UI components
   - Processing logic in `gui_processing.py` handles video processing tasks

### Common Issues and Solutions

1. **Out of Memory Errors**:
   - Use enhanced chunk processing with `--max_size` to limit resolution
   - Increase number of chunks for high-resolution videos
   - Enable auto-chunking to automatically determine optimal chunk count

2. **Mask Quality Issues**:
   - Adjust erosion/dilation parameters to improve mask quality
   - Use SAM2 for better initial mask generation
   - Apply mask enhancement algorithms for challenging videos

3. **Processing Speed**:
   - Lower resolution with `--max_size` for faster processing
   - Use multiple smaller chunks instead of fewer large chunks
   - Disable bidirectional processing for faster (but lower quality) results

4. **UI Coordinate Issues**:
   - If cursor positioning in mask generator UI is offset after panning or zooming, use backup version v051* which contains the fix for this issue
   - The fixed version properly handles coordinate transformations between screen, display, and original image spaces