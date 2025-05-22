# matanyone_gui.py - v1.1737774300
# Updated: Friday, January 24, 2025 at 17:45:00 PST
# Changes in this version:
# - Refactored large monolithic GUI into multiple modules for better maintainability
# - Split functionality into separate modules: gui_config, gui_processing, gui_events, gui_widgets
# - Maintained all existing functionality while improving code organization
# - Simplified main application class to focus on initialization and coordination
# - No changes needed to other modules - all imports and dependencies handled internally

"""
MatAnyone Video Processor GUI - Main entry point.
Provides user interface for video matting with MatAnyone.
"""

import os
import sys
import tkinter as tk
from tkinter import StringVar, BooleanVar, IntVar, DoubleVar
import platform

# Import our modular components
from ui.gui_config import ConfigManager
from ui.gui_processing import ProcessingManager
from ui.gui_events import EventHandler
from ui.gui_widgets import WidgetManager

# Import our UI components
from ui.ui_components import TextRedirector


class MatAnyoneApp:
    """Main application class for MatAnyone GUI"""
    
    def __init__(self, root):
        """Initialize the MatAnyone application"""
        self.root = root
        self.root.title("MatAnyone Video Processor")
        self.root.geometry("950x770")  # Slightly taller to accommodate new controls
        
        # Initialize managers
        self.config_manager = ConfigManager()
        self.widget_manager = WidgetManager(self)
        self.event_handler = EventHandler(self)
        self.processing_manager = ProcessingManager(self)
        
        # Apply platform-specific styling
        self.widget_manager.apply_theme()
        
        # Create variables for file paths
        self.video_path = StringVar()
        self.mask_path = StringVar()
        self.output_path = StringVar()
        
        # Default output path is current directory
        self.output_path.set(os.path.join(os.getcwd(), "outputs"))
        
        # Variable for input type (video or image sequence)
        self.input_type = StringVar(value="video")
        
        # Mask generator settings
        self.mask_brush_size = 5  # Default brush size
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Setup the original stdout before creating the console
        self.original_stdout = sys.stdout
        
        # Create UI elements
        self.widget_manager.create_all_widgets()
        
        # Redirect stdout to console after initialization
        sys.stdout = TextRedirector(self.console)
        
        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.event_handler.on_closing)
        
        # Load saved settings
        self.config_manager.load_settings(self)
        
        # Check existing mask for keyframe metadata
        self.event_handler.check_existing_mask_keyframe()
        
        # Print welcome message
        print("Welcome to MatAnyone Video Processor GUI")
        print(f"Current directory: {os.getcwd()}")
        print("Ready to process videos")
    
    def clear_console(self):
        """Clear the console output"""
        try:
            self.console.configure(state=tk.NORMAL)
            self.console.delete(1.0, tk.END)
            self.console.insert(tk.END, "Console cleared. Ready for new output.\n")
            self.console.configure(state=tk.DISABLED)
            self.console.update()  # Force update
        except Exception as e:
            # If there's an error clearing the console, print to the original stdout
            if hasattr(self, 'original_stdout'):
                print(f"Error clearing console: {str(e)}", file=self.original_stdout)
    
    # Delegate methods to event handler for backward compatibility
    def toggle_enhanced_options(self):
        """Delegate to event handler"""
        self.event_handler.toggle_enhanced_options()
    
    def toggle_autochunk(self):
        """Delegate to event handler"""
        self.event_handler.toggle_autochunk()
    
    def update_input_label(self):
        """Delegate to event handler"""
        self.event_handler.update_input_label()
    
    def browse_input(self):
        """Delegate to event handler"""
        self.event_handler.browse_input()
    
    def browse_video(self):
        """Delegate to event handler"""
        self.event_handler.browse_video()
    
    def browse_sequence(self):
        """Delegate to event handler"""
        self.event_handler.browse_sequence()
    
    def browse_mask(self):
        """Delegate to event handler"""
        self.event_handler.browse_mask()
    
    def generate_mask(self):
        """Delegate to event handler"""
        self.event_handler.generate_mask()
    
    def browse_output(self):
        """Delegate to event handler"""
        self.event_handler.browse_output()


# Add ttk import to maintain compatibility with widget references
from tkinter import ttk


if __name__ == "__main__":
    # Handle command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="MatAnyone Video Processor GUI")
    parser.add_argument('--input', type=str, help='Input video path')
    parser.add_argument('--mask', type=str, help='Input mask path')
    parser.add_argument('--output', type=str, help='Output directory')
    args = parser.parse_args()
    
    # Create root window
    root = tk.Tk()
    app = MatAnyoneApp(root)
    
    # Set initial values from command line arguments
    if args.input:
        app.video_path.set(args.input)
    if args.mask:
        app.mask_path.set(args.mask)
    if args.output:
        app.output_path.set(args.output)
    
    # Start GUI main loop
    root.mainloop()
