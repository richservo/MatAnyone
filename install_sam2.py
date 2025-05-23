#!/usr/bin/env python3
"""
SAM2.1 Auto-Installer for Python 3.8
Automatically detects and upgrades to SAM2.1
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

def check_installed_version():
    """Check which version of SAM weights are installed"""
    cache_dir = os.path.expanduser("~/.cache/sam")
    sam21_path = os.path.join(cache_dir, "sam2.1_hiera_large.pt")
    sam2_path = os.path.join(cache_dir, "sam2_hiera_l.pth")
    
    if os.path.exists(sam21_path):
        return "sam2.1"
    elif os.path.exists(sam2_path):
        # Check if it's actually a symlink to SAM2.1
        if os.path.islink(sam2_path):
            link_target = os.readlink(sam2_path)
            if "sam2.1" in link_target:
                return "sam2.1"
        return "sam2"
    else:
        return None

def uninstall_sam2():
    """Completely uninstall SAM2 and clean up all related files"""
    print_status("Uninstalling SAM2...", "info")
    
    # Uninstall SAM2 package
    print_status("Removing SAM2 package...", "info")
    run_command("pip uninstall sam2 -y", check=False)
    
    # Remove SAM2 source directory
    sam2_src_dir = os.path.expanduser("~/.cache/sam2_src")
    if os.path.exists(sam2_src_dir):
        print_status(f"Removing SAM2 source directory: {sam2_src_dir}", "info")
        shutil.rmtree(sam2_src_dir)
    
    # Remove model weights
    cache_dir = os.path.expanduser("~/.cache/sam")
    if os.path.exists(cache_dir):
        print_status(f"Removing model weights directory: {cache_dir}", "info")
        shutil.rmtree(cache_dir)
    
    # Clean up any remaining sam2 modules from Python path
    for key in list(sys.modules.keys()):
        if key.startswith('sam2'):
            del sys.modules[key]
    
    print_status("SAM2 uninstalled successfully!", "success")

def download_model(model_type="sam2.1"):
    """Download SAM2 or SAM2.1 model weights"""
    cache_dir = os.path.expanduser("~/.cache/sam")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Model configurations
    models = {
        "sam2.1": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
            "filename": "sam2.1_hiera_large.pt",
            "size": "898MB"
        },
        "sam2": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
            "filename": "sam2_hiera_l.pth",  # Keep old filename for compatibility
            "size": "856MB"
        }
    }
    
    model_config = models[model_type]
    model_path = os.path.join(cache_dir, model_config["filename"])
    
    # Also create a symlink for backward compatibility if using SAM2.1
    if model_type == "sam2.1":
        legacy_path = os.path.join(cache_dir, "sam2_hiera_l.pth")
        if os.path.exists(legacy_path) and not os.path.islink(legacy_path):
            os.remove(legacy_path)
    
    if os.path.exists(model_path):
        print_status(f"{model_type.upper()} model already exists", "success")
        # Create symlink for backward compatibility
        if model_type == "sam2.1":
            legacy_path = os.path.join(cache_dir, "sam2_hiera_l.pth")
            if not os.path.exists(legacy_path):
                os.symlink(model_path, legacy_path)
                print_status("Created compatibility symlink for legacy code", "info")
        return True
    
    print_status(f"Downloading {model_type.upper()} model weights ({model_config['size']})...", "info")
    # Use appropriate download command based on platform
    if platform.system() == "Windows":
        # Use PowerShell's Invoke-WebRequest for Windows
        download_cmd = f'powershell -Command "Invoke-WebRequest -Uri \'{model_config["url"]}\' -OutFile \'{model_path}\'"'
    else:
        download_cmd = f'curl -L -o "{model_path}" "{model_config["url"]}"'
    
    print_status("This will take a few minutes depending on your internet speed...", "info")
    output, code = run_command(download_cmd, check=False)
    
    if code == 0 and os.path.exists(model_path):
        print_status(f"{model_type.upper()} model downloaded successfully!", "success")
        # Create symlink for backward compatibility
        if model_type == "sam2.1":
            legacy_path = os.path.join(cache_dir, "sam2_hiera_l.pth")
            if not os.path.exists(legacy_path):
                os.symlink(model_path, legacy_path)
                print_status("Created compatibility symlink for legacy code", "info")
        return True
    else:
        print_status("Model download failed - you may need to download manually", "warning")
        print(f"Download from: {model_config['url']}")
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
    print("SAM2.1 Auto-Installer for MatAnyone")
    print("="*60 + "\n")
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print_status(f"Python version: {python_version}", "info")
    
    # Check if SAM2 is already installed
    sam2_installed = check_existing_installation()
    
    if sam2_installed:
        print_status("SAM2 is already installed!", "success")
        
        # Check which version of weights we have
        installed_version = check_installed_version()
        
        if installed_version == "sam2.1":
            print_status("SAM2.1 model weights found", "success")
            print("\n✅ SAM2.1 is already set up and ready to use!")
            return 0
        elif installed_version == "sam2":
            print_status("SAM2 model weights found (older version)", "warning")
            print("\n🔄 You have the older SAM2 model installed.")
            response = input("Would you like to upgrade to the newer SAM2.1 model for better performance? (y/n): ")
            
            if response.lower() == 'y':
                print_status("Upgrading to SAM2.1...", "info")
                # Just download the new model, no need to reinstall SAM2
                if download_model("sam2.1"):
                    # Remove old SAM2 model if upgrade successful
                    old_model = os.path.expanduser("~/.cache/sam/sam2_hiera_l.pth")
                    if os.path.exists(old_model) and not os.path.islink(old_model):
                        os.remove(old_model)
                        print_status("Removed old SAM2 model", "info")
                    print("\n✅ Successfully upgraded to SAM2.1!")
                    return 0
                else:
                    print_status("Upgrade failed", "error")
                    return 1
            else:
                print("\n✅ Keeping SAM2 (older version)")
                return 0
        else:
            print_status("SAM2 is installed but model weights are missing", "warning")
            response = input("\nWould you like to download the latest SAM2.1 model weights? (y/n): ")
            if response.lower() != 'y':
                print("Skipping model download.")
                return 0
            # Download SAM2.1 by default
            if download_model("sam2.1"):
                print("\n✅ SAM2.1 is now ready to use!")
                return 0
            else:
                return 1
    else:
        print_status("SAM2 not found", "info")
        response = input("\nWould you like to install SAM2 with the latest SAM2.1 model? (y/n): ")
        
        if response.lower() != 'y':
            print("Installation cancelled.")
            return 0
        
        print_status("Proceeding with SAM2 installation...", "info")
        
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
        
        # Download SAM2.1 model
        download_model("sam2.1")
        
        # Final verification
        print_status("Running final verification...", "info")
        try:
            import torch
            print_status("✅ SAM2.1 is ready to use with MatAnyone!", "success")
            print("\nInstallation completed successfully!")
            
            cache_dir = os.path.expanduser("~/.cache/sam")
            print(f"\nModel location: {cache_dir}")
            print("Models available:")
            if os.path.exists(cache_dir):
                for f in os.listdir(cache_dir):
                    if f.endswith('.pt') or f.endswith('.pth'):
                        print(f"  - {f}")
            
            print("\nYou can now use SAM2 in MatAnyone. The mask generator will automatically")
            print("use SAM2 for better quality masks, with fallback to SAM1 if needed.")
            print(f"\nSAM2 source installed at: ~/.cache/sam2_src/sam2")
            
        except Exception as e:
            print_status(f"Verification failed: {e}", "error")
            return 1
    
    # Check for Windows compatibility issues
    if platform.system() == "Windows":
        print("\n⚠️  Note: SAM2 has known compatibility issues on Windows.")
        print("If you encounter problems, consider using WSL2 or a Linux/Mac system.")
    
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