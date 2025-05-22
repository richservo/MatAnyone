#!/usr/bin/env python3
"""
GUI test script to check mask metadata with file selection dialog
"""
import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox

sys.path.append('/Volumes/Storage/Richard/MatAnyone')

from mask.mask_utils import get_keyframe_metadata_from_mask

def check_mask_metadata():
    """Open file dialog and check selected mask for metadata"""
    # Create root window (hidden)
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Open file dialog
    mask_path = filedialog.askopenfilename(
        title="Select Mask File to Check",
        filetypes=[
            ("PNG files", "*.png"),
            ("All image files", "*.png *.jpg *.jpeg"),
            ("All files", "*.*")
        ],
        initialdir=os.getcwd()
    )
    
    if not mask_path:
        print("No file selected.")
        root.destroy()
        return
    
    print(f"Checking mask metadata for: {mask_path}")
    print("-" * 50)
    
    try:
        # Check if file exists
        if not os.path.exists(mask_path):
            print("❌ Error: File does not exist")
            root.destroy()
            return
            
        # Check metadata
        metadata = get_keyframe_metadata_from_mask(mask_path)
        
        if metadata is not None:
            print(f"✅ FOUND keyframe metadata: frame {metadata}")
            result_msg = f"✅ Keyframe metadata found!\n\nFrame number: {metadata}"
        else:
            print("❌ NO keyframe metadata found")
            result_msg = "❌ No keyframe metadata found in this mask."
            
        # Show result in popup
        messagebox.showinfo("Metadata Check Result", result_msg)
        
    except Exception as e:
        error_msg = f"❌ Error reading metadata: {e}"
        print(error_msg)
        messagebox.showerror("Error", error_msg)
    
    root.destroy()

if __name__ == "__main__":
    print("Opening file dialog to select mask file...")
    check_mask_metadata()