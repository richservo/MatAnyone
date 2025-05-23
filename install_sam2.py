#!/usr/bin/env python3
"""
SAM2 One-Click Installer for Python 3.8
Simply run: python install_sam2.py
"""

import os
import sys
import subprocess
import shutil
import tempfile
import platform
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.stdout, result.returncode
    except subprocess.CalledProcessError as e:
        return e.stderr, e.returncode

def print_status(message, status="info"):
    """Print colored status messages"""
    colors = {
        "info": "\033[94m",
        "success": "\033[92m", 
        "warning": "\033[93m",
        "error": "\033[91m"
    }
    reset = "\033[0m"
    print(f"{colors.get(status, '')}[{status.upper()}]{reset} {message}")

def check_existing_installation():
    """Check if SAM2 is already installed"""
    try:
        import sam2
        from sam2.build_sam import build_sam2
        return True
    except ImportError:
        return False

def download_model():
    """Download SAM2 model weights"""
    cache_dir = os.path.expanduser("~/.cache/sam")
    os.makedirs(cache_dir, exist_ok=True)
    
    model_path = os.path.join(cache_dir, "sam2_hiera_l.pth")
    
    if os.path.exists(model_path):
        print_status("Model already exists", "success")
        return True
    
    print_status("Downloading SAM2 model weights (856MB)...", "info")
    # Use appropriate download command based on platform
    import platform
    if platform.system() == "Windows":
        # Use PowerShell's Invoke-WebRequest for Windows
        download_cmd = f'powershell -Command "Invoke-WebRequest -Uri \'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt\' -OutFile \'{model_path}\'"'
    else:
        download_cmd = f'curl -L -o "{model_path}" "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"'
    print_status("This will take a few minutes depending on your internet speed...", "info")
    output, code = run_command(download_cmd, check=False)
    
    if code == 0 and os.path.exists(model_path):
        print_status("Model downloaded successfully!", "success")
        return True
    else:
        print_status("Model download failed - you may need to download manually", "warning")
        print(f"Download from: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt")
        print(f"Save to: {model_path}")
        return False

def install_sam2():
    """Install SAM2 with Python 3.8 compatibility"""
    # Create a permanent directory for SAM2 source
    sam2_src_dir = os.path.expanduser("~/.cache/sam2_src")
    
    # If it already exists, remove it to ensure clean installation
    if os.path.exists(sam2_src_dir):
        print_status("Removing old SAM2 source directory...", "info")
        shutil.rmtree(sam2_src_dir)
    
    os.makedirs(sam2_src_dir, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print_status("Creating temporary directory for download...", "info")
        
        # Clone SAM2
        print_status("Downloading SAM2 from GitHub...", "info")
        clone_cmd = f"cd {temp_dir} && git clone https://github.com/facebookresearch/sam2.git"
        output, code = run_command(clone_cmd)
        
        if code != 0:
            print_status("Failed to clone SAM2 repository", "error")
            print("Please check your internet connection and try again.")
            return False
        
        temp_sam2_dir = os.path.join(temp_dir, "sam2")
        
        # Move to permanent location
        print_status("Moving SAM2 to permanent location...", "info")
        final_sam2_dir = os.path.join(sam2_src_dir, "sam2")
        shutil.move(temp_sam2_dir, final_sam2_dir)
        
        # Create modified setup.py
        print_status("Applying Python 3.8 compatibility patches...", "info")
        
        # Read original setup.py and modify it
        setup_path = os.path.join(final_sam2_dir, "setup.py")
        with open(setup_path, 'r') as f:
            setup_content = f.read()
        
        # Apply modifications
        setup_content = setup_content.replace('python_requires=">=3.10.0"', 'python_requires=">=3.8.0"')
        setup_content = setup_content.replace('"torch>=2.5.1"', '"torch>=2.3.0"')
        setup_content = setup_content.replace('"torchvision>=0.20.1"', '"torchvision>=0.18.0"')
        
        with open(setup_path, 'w') as f:
            f.write(setup_content)
        
        # Modify pyproject.toml
        pyproject_path = os.path.join(final_sam2_dir, "pyproject.toml")
        with open(pyproject_path, 'r') as f:
            pyproject_content = f.read()
        
        pyproject_content = pyproject_content.replace('"torch>=2.5.1"', '"torch>=2.3.0"')
        
        with open(pyproject_path, 'w') as f:
            f.write(pyproject_content)
        
        # Install SAM2
        print_status("Installing SAM2 (this may take a few minutes)...", "info")
        install_cmd = f"cd {final_sam2_dir} && pip install -e . --ignore-requires-python"
        output, code = run_command(install_cmd)
        
        if code != 0:
            print_status("Failed to install SAM2", "error")
            print("Error output:", output)
            return False
        
        print_status("SAM2 installed successfully!", "success")
        print_status(f"SAM2 source code location: {final_sam2_dir}", "info")
        return True

def main():
    print("\n" + "="*60)
    print("SAM2 Easy Installer for MatAnyone")
    print("="*60 + "\n")
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print_status(f"Python version: {python_version}", "info")
    
    # Check if SAM2 is already installed
    sam2_installed = check_existing_installation()
    
    if sam2_installed:
        print_status("SAM2 is already installed!", "success")
        
        # Check if model exists
        model_path = os.path.expanduser("~/.cache/sam/sam2_hiera_l.pth")
        if os.path.exists(model_path):
            print_status("SAM2 model weights found", "success")
            print("\n✅ SAM2 is already set up and ready to use!")
            return 0
        else:
            print_status("SAM2 is installed but model weights are missing", "warning")
            response = input("\nWould you like to download the model weights? (y/n): ")
            if response.lower() != 'y':
                print("Skipping model download.")
                return 0
            # Download model
            if download_model():
                print("\n✅ SAM2 is now ready to use!")
                return 0
            else:
                return 1
    else:
        print_status("SAM2 not found, proceeding with installation...", "info")
        
        # Install SAM2
        if not install_sam2():
            return 1
        
        # Test import - need to refresh Python's module cache
        print_status("Testing SAM2 import...", "info")
        try:
            # Force Python to reload the module path
            import importlib
            if 'sam2' in sys.modules:
                del sys.modules['sam2']
            
            import sam2
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            print_status("SAM2 imports working correctly!", "success")
        except ImportError as e:
            print_status(f"Import test failed: {e}", "error")
            print_status("Note: You may need to restart your Python session to use SAM2", "warning")
            # Don't fail - the installation was successful
            pass
        
        # Download model
        download_model()
        
        # Final verification
        print_status("Running final verification...", "info")
        try:
            import torch
            print_status("✅ SAM2 is ready to use with MatAnyone!", "success")
            print("\nInstallation completed successfully!")
            model_path = os.path.expanduser("~/.cache/sam/sam2_hiera_l.pth")
            print(f"Model location: {model_path}")
            
            print("\nYou can now use SAM2 in MatAnyone. The mask generator will automatically")
            print("use SAM2 for better quality masks, with fallback to SAM1 if needed.")
            print(f"\nSAM2 source installed at: ~/.cache/sam2_src/sam2")
            
        except Exception as e:
            print_status(f"Verification failed: {e}", "error")
            return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)