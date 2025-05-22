# gui_config.py - v1.1737774900
# Updated: Friday, January 24, 2025 at 17:45:00 PST
# Changes in this version:
# - Added video quality/codec/bitrate settings to configuration management
# - Save and load new video encoding parameters (codec, quality, custom bitrate)
# - Extended settings persistence to include new video quality options

"""
Configuration management for MatAnyone GUI.
Handles saving and loading user settings.
"""

import os
import json
import platform

# Try to import appdirs, use fallback if not available
try:
    import appdirs
except ImportError:
    print("Required package 'appdirs' not found. Using built-in fallback...")
    # Create a simple replacement for appdirs functionality
    class AppDirsReplacement:
        def user_config_dir(self, appname):
            """Simple replacement for appdirs.user_config_dir"""
            if platform.system() == "Windows":
                path = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), appname)
            elif platform.system() == "Darwin":  # macOS
                path = os.path.join(os.path.expanduser("~"), "Library", "Application Support", appname)
            else:  # Linux and other Unix-like
                path = os.path.join(os.path.expanduser("~"), ".config", appname)
            return path
    
    # Use our replacement
    appdirs = AppDirsReplacement()


class ConfigManager:
    """Manages configuration saving and loading for the MatAnyone GUI"""
    
    def __init__(self):
        """Initialize the configuration manager"""
        self.config_dir = appdirs.user_config_dir("MatAnyoneGUI")
        self.config_file = os.path.join(self.config_dir, "settings.json")
    
    def save_settings(self, app):
        """
        Save current settings to config file
        
        Args:
            app: MatAnyoneApp instance to save settings from
        """
        try:
            # Create config directory if it doesn't exist
            os.makedirs(self.config_dir, exist_ok=True)
            
            # Prepare settings dictionary
            settings = {
                'input_type': app.input_type.get(),
                'video_path': app.video_path.get(),
                'mask_path': app.mask_path.get(),
                'output_path': app.output_path.get(),
                'n_warmup': app.n_warmup.get(),
                'r_erode': app.r_erode.get(),
                'r_dilate': app.r_dilate.get(),
                'max_size': app.max_size.get(),
                'num_chunks': app.num_chunks.get(),
                'chunk_type': app.chunk_type.get(),
                'save_image': app.save_image.get(),
                'bidirectional': app.bidirectional.get(),
                'reverse_dilate': app.reverse_dilate.get(),
                'blend_method': app.blend_method.get(),
                'lowres_blend_method': app.lowres_blend_method.get(),
                'cleanup_temp': app.cleanup_temp.get(),
                'use_enhanced_chunks': app.use_enhanced_chunks.get(),
                'use_autochunk': app.use_autochunk.get(),
                'use_heat_map_chunking': app.use_heat_map_chunking.get(),
                'face_priority_weight': app.face_priority_weight.get(),
                'low_res_scale': app.low_res_scale.get(),
                'mask_threshold': app.mask_threshold.get(),
                'prioritize_faces': app.prioritize_faces.get(),
                'use_parallel_processing': app.use_parallel_processing.get(),
                'window_geometry': app.root.geometry(),
                # New video quality settings
                'video_codec': app.video_codec.get(),
                'video_quality': app.video_quality.get(),
                'custom_bitrate_enabled': app.custom_bitrate_enabled.get(),
                'custom_bitrate': app.custom_bitrate.get(),
                # Mask generator settings
                'mask_brush_size': getattr(app, 'mask_brush_size', 5)
            }
            
            # Save to file
            with open(self.config_file, 'w') as f:
                json.dump(settings, f, indent=2)
                
            print(f"Settings saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving settings: {str(e)}")
    
    def load_settings(self, app):
        """
        Load settings from config file
        
        Args:
            app: MatAnyoneApp instance to load settings into
        """
        try:
            # Check if config file exists
            if not os.path.exists(self.config_file):
                print("No saved settings found")
                return
            
            # Load settings
            with open(self.config_file, 'r') as f:
                settings = json.load(f)
            
            # Apply settings to variables
            if 'input_type' in settings:
                app.input_type.set(settings['input_type'])
                app.update_input_label()
                
            if 'video_path' in settings and os.path.exists(settings['video_path']):
                app.video_path.set(settings['video_path'])
                
            if 'mask_path' in settings and os.path.exists(settings['mask_path']):
                app.mask_path.set(settings['mask_path'])
                
            if 'output_path' in settings:
                app.output_path.set(settings['output_path'])
                
            if 'n_warmup' in settings:
                app.n_warmup.set(settings['n_warmup'])
                
            if 'r_erode' in settings:
                app.r_erode.set(settings['r_erode'])
                
            if 'r_dilate' in settings:
                app.r_dilate.set(settings['r_dilate'])
                
            if 'max_size' in settings:
                app.max_size.set(settings['max_size'])
                
            if 'num_chunks' in settings:
                app.num_chunks.set(settings['num_chunks'])
                
            if 'chunk_type' in settings:
                app.chunk_type.set(settings['chunk_type'])
                
            if 'save_image' in settings:
                app.save_image.set(settings['save_image'])
                
            if 'bidirectional' in settings:
                app.bidirectional.set(settings['bidirectional'])
                
            if 'reverse_dilate' in settings:
                app.reverse_dilate.set(settings['reverse_dilate'])
                
            if 'blend_method' in settings:
                app.blend_method.set(settings['blend_method'])
                
            if 'lowres_blend_method' in settings:
                app.lowres_blend_method.set(settings['lowres_blend_method'])
                
            if 'cleanup_temp' in settings:
                app.cleanup_temp.set(settings['cleanup_temp'])
                
            if 'use_enhanced_chunks' in settings:
                app.use_enhanced_chunks.set(settings['use_enhanced_chunks'])
                
            if 'use_autochunk' in settings:
                app.use_autochunk.set(settings['use_autochunk'])
                
            if 'use_heat_map_chunking' in settings:
                app.use_heat_map_chunking.set(settings['use_heat_map_chunking'])
                
            if 'face_priority_weight' in settings:
                app.face_priority_weight.set(settings['face_priority_weight'])
                
            if 'low_res_scale' in settings:
                app.low_res_scale.set(settings['low_res_scale'])
                
            if 'mask_threshold' in settings:
                app.mask_threshold.set(settings['mask_threshold'])
                
            if 'prioritize_faces' in settings:
                app.prioritize_faces.set(settings['prioritize_faces'])
                
            if 'use_parallel_processing' in settings:
                app.use_parallel_processing.set(settings['use_parallel_processing'])
                
            if 'window_geometry' in settings:
                app.root.geometry(settings['window_geometry'])
            
            # Load new video quality settings
            if 'video_codec' in settings:
                app.video_codec.set(settings['video_codec'])
                
            if 'video_quality' in settings:
                app.video_quality.set(settings['video_quality'])
                
            if 'custom_bitrate_enabled' in settings:
                app.custom_bitrate_enabled.set(settings['custom_bitrate_enabled'])
                
            if 'custom_bitrate' in settings:
                app.custom_bitrate.set(settings['custom_bitrate'])
            
            # Load mask generator settings
            if 'mask_brush_size' in settings:
                app.mask_brush_size = settings['mask_brush_size']
                
            # Update UI based on loaded settings
            app.toggle_enhanced_options()
            app.toggle_autochunk()
            
            # Also update custom bitrate UI state
            if hasattr(app.widget_manager, 'toggle_custom_bitrate'):
                app.widget_manager.toggle_custom_bitrate()
            
            print("Settings loaded successfully")
        except Exception as e:
            print(f"Error loading settings: {str(e)}")
            # If there's an error, we'll just use the default settings
