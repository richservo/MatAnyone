# gui_processing.py - v1.1737774900
# Updated: Friday, January 24, 2025 at 17:55:00 PST
# Changes in this version:
# - Added video quality settings (codec, quality, custom bitrate) to processing parameters
# - Pass video encoding settings to all processing functions
# - Enhanced parameter collection to include new video quality options

"""
Processing management for MatAnyone GUI.
Handles video processing in separate threads with progress monitoring.
"""

import os
import sys
import threading
import time
import re
import traceback
import platform
from tkinter import messagebox

# Import our image processing module
try:
    from utils.image_processing import InterruptibleInferenceCore
except ImportError as e:
    print(f"Error importing image_processing module: {str(e)}")
    print("Please make sure the image_processing.py file is in the same directory.")
    sys.exit(1)


class ProcessingManager:
    """Manages video processing operations and thread management"""
    
    def __init__(self, app):
        """
        Initialize the processing manager
        
        Args:
            app: Reference to the main MatAnyoneApp instance
        """
        self.app = app
        self.processor = None
        self.processing_thread = None
        self.processing = False
        self.cancel_processing = False
        self.progress_timer_id = None
    
    def process_video(self):
        """Start or cancel video processing"""
        # If already processing, this becomes a cancel button
        if self.processing:
            # Set cancel flag to interrupt processing
            self.cancel_processing = True
            self.app.process_button.configure(text="Cancelling...", state="disabled")
            self.app.status_var.set("Cancelling...")
            print("Cancelling processing...")
            return

        # Validate inputs
        if not self.validate_inputs():
            return
        
        # Reset flags
        self.cancel_processing = False
        self.processing = True
        
        # Reset and prepare progress bar
        self.app.progress['value'] = 0
        self.app.progress['mode'] = 'determinate'
        self.app.progress_text.set("0%")
        self.app.progress_stage.set("")
        
        # Disable UI during processing
        self.app.process_button.configure(text="Cancel", state="normal")
        self.app.status_var.set("Initializing...")
                
        # Start processing in a separate thread
        self.processing_thread = threading.Thread(target=self.run_processing)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def validate_inputs(self):
        """
        Validate all inputs before starting processing
        
        Returns:
            bool: True if all inputs are valid, False otherwise
        """
        input_path = self.app.video_path.get()
        if not input_path:
            if self.app.input_type.get() == "video":
                messagebox.showerror("Error", "Please select an input video")
            else:
                messagebox.showerror("Error", "Please select an image sequence folder")
            return False
        
        # Check if input exists
        if not os.path.exists(input_path):
            messagebox.showerror("Error", "Input path does not exist")
            return False
        
        # For image sequence, validate that folder contains images
        if self.app.input_type.get() == "sequence":
            if not os.path.isdir(input_path):
                messagebox.showerror("Error", "Selected path is not a directory")
                return False
            
            # Check for image files
            image_extensions = ['.jpg', '.jpeg', '.png']
            image_files = [f for f in os.listdir(input_path) 
                          if os.path.isfile(os.path.join(input_path, f)) and 
                          os.path.splitext(f.lower())[1] in image_extensions]
            
            if not image_files:
                messagebox.showerror("Error", "No image files found in the selected directory")
                return False
            
            # Sort image files (important for sequences)
            try:
                image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
            except:
                # Fallback to regular sort if the numeric sort fails
                image_files.sort()
        
        if not self.app.mask_path.get():
            # Offer to generate a mask if none is selected
            if self.app.input_type.get() == "video":
                # Import here to avoid circular imports
                try:
                    from mask.mask_generator import MaskGeneratorUI
                    HAS_MASK_GENERATOR = True
                except ImportError:
                    HAS_MASK_GENERATOR = False
                
                if HAS_MASK_GENERATOR:
                    if messagebox.askyesno("No Mask Selected", 
                                          "No mask is selected. Would you like to generate one?"):
                        self.app.generate_mask()
                        return False
                else:
                    messagebox.showerror("Error", "Please select a mask image")
                    return False
            else:
                messagebox.showerror("Error", "Please select a mask image")
                return False
        
        if not os.path.exists(self.app.mask_path.get()):
            messagebox.showerror("Error", "Mask file does not exist")
            return False
        
        if not os.path.exists(self.app.output_path.get()):
            try:
                os.makedirs(self.app.output_path.get(), exist_ok=True)
                print(f"Created output directory: {self.app.output_path.get()}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not create output directory: {str(e)}")
                return False
        
        # Validate num_chunks
        num_chunks = self.app.num_chunks.get()
        if num_chunks > 1 and self.app.input_type.get() == "sequence":
            messagebox.showerror("Error", "Chunking is currently only supported for video files, not image sequences")
            return False
        
        # Enhanced chunking is also only for videos
        if self.app.use_enhanced_chunks.get() and self.app.input_type.get() == "sequence":
            messagebox.showerror("Error", "Enhanced chunk processing is only supported for video files")
            return False
        
        # Autochunk mode requires enhanced chunking
        if self.app.use_autochunk.get() and not self.app.use_enhanced_chunks.get():
            messagebox.showerror("Error", "Auto-chunk mode requires Enhanced Chunk Processing to be enabled")
            return False
        
        # Enhanced chunking requires num_chunks > 1 (unless using autochunk)
        if self.app.use_enhanced_chunks.get() and num_chunks <= 1 and not self.app.use_autochunk.get():
            messagebox.showerror("Error", "Enhanced chunk processing requires Number of Chunks to be 2 or more")
            return False
        
        return True
    
    def run_processing(self):
        """Main processing function that runs in a separate thread"""
        try:
            # Clear console
            self.app.console.after(0, self.app.clear_console)
            
            # Update status
            self.app.status_var.set("Loading model...")
            self.update_progress(0, "Loading model...")
            
            # Initialize the model if not already initialized or if it's not our interruptible version
            if self.processor is None or not isinstance(self.processor, InterruptibleInferenceCore):
                print("Initializing MatAnyone model...")
                try:
                    # Show status during initialization which can take time
                    self.app.status_var.set("Loading model (may take a minute)...")
                    self.app.root.update_idletasks()
                    
                    # Initialize the model
                    self.processor = InterruptibleInferenceCore("PeiqingYang/MatAnyone")
                    print("Model loaded successfully")
                    self.update_progress(5, "Model loaded successfully")
                except Exception as e:
                    self.app.root.after(0, self.processing_error, 
                                       f"Failed to initialize model: {str(e)}\n\n"
                                       f"Please make sure you have properly installed MatAnyone "
                                       f"and have an internet connection for model download.")
                    return
            
            # Update status
            input_type_str = "image sequence" if self.app.input_type.get() == "sequence" else "video"
            
            # If using chunks, indicate that in the status
            num_chunks = self.app.num_chunks.get()
            chunk_type = self.app.chunk_type.get()
            
            if self.app.use_autochunk.get():
                self.update_progress(10, f"Processing {input_type_str} with automatic chunk sizing...")
            elif num_chunks > 1 and self.app.use_enhanced_chunks.get():
                if chunk_type == "grid":
                    self.update_progress(10, f"Processing {input_type_str} with enhanced grid processing...")
                else:
                    self.update_progress(10, f"Processing {input_type_str} with enhanced strip processing...")
            elif num_chunks > 1:
                self.update_progress(10, f"Processing {input_type_str} in {num_chunks} chunks...")
            else:
                self.update_progress(10, f"Processing {input_type_str}...")
            
            # Set up output paths
            output_dir = self.app.output_path.get()
            video_name = os.path.basename(self.app.video_path.get())
            if video_name.endswith(('.mp4', '.mov', '.avi')):
                video_name = os.path.splitext(video_name)[0]
                
            # Ensure the output directories exist if saving individual frames
            if bool(self.app.save_image.get()):
                fgr_dir = os.path.join(output_dir, f"{video_name}/fgr")
                pha_dir = os.path.join(output_dir, f"{video_name}/pha")
                os.makedirs(fgr_dir, exist_ok=True)
                os.makedirs(pha_dir, exist_ok=True)
            
            # Process parameters
            process_params = {
                'input_path': self.app.video_path.get(),
                'mask_path': self.app.mask_path.get(),
                'output_path': self.app.output_path.get(),
                'n_warmup': self.app.n_warmup.get(),
                'r_erode': self.app.r_erode.get(),
                'r_dilate': self.app.r_dilate.get(),
                'save_image': bool(self.app.save_image.get()),
                'max_size': self.app.max_size.get()
            }
            
            # Add video quality parameters
            video_quality_params = {
                'video_codec': self.app.video_codec.get(),
                'video_quality': self.app.video_quality.get(),
                'custom_bitrate': self.app.custom_bitrate.get() if self.app.custom_bitrate_enabled.get() else None
            }
            process_params.update(video_quality_params)
            
            # Setup progress monitoring
            self.setup_progress_monitoring()
            
            # Define a checker for cancellation
            def check_cancellation():
                if hasattr(self, 'cancel_processing') and self.cancel_processing:
                    print("Requesting interrupt for processing...")
                    self.processor.request_interrupt()
                    return
                
                # Schedule the next check if still processing
                if hasattr(self, 'processing') and self.processing:
                    self.app.root.after(100, check_cancellation)
            
            # Start checking for cancellation
            self.app.root.after(100, check_cancellation)
            
            # Start the processing
            try:
                print("Starting video processing with parameters:")
                for key, value in process_params.items():
                    print(f"  {key}: {value}")
                
                self.update_progress(15, "Starting video processing...")
                
                # Check if we're processing in chunks
                num_chunks = self.app.num_chunks.get()
                
                # Check if we're doing bidirectional processing
                bidirectional = bool(self.app.bidirectional.get())
                
                if self.app.use_enhanced_chunks.get() and self.app.use_autochunk.get():
                    print("Using auto-chunk mode based on low-res resolution")
                elif num_chunks > 1:
                    if chunk_type == "grid":
                        print(f"Processing in a grid of chunks")
                    else:
                        print(f"Processing in {num_chunks} horizontal strips")
                        
                    if bidirectional:
                        print("Bidirectional processing with chunks is enabled.")
                        print("This will process each chunk in both directions for better results.")
                
                if bidirectional:
                    print("Bidirectional processing enabled")
                    
                # Add bidirectional parameters
                process_params['bidirectional'] = bidirectional
                process_params['blend_method'] = self.app.blend_method.get()
                process_params['reverse_dilate'] = self.app.reverse_dilate.get()
                process_params['cleanup_temp'] = bool(self.app.cleanup_temp.get())
                
                # Determine the processing method
                if num_chunks > 1 and self.app.use_enhanced_chunks.get():
                    if self.app.use_autochunk.get():
                        print("Using automatic chunk sizing based on low-res dimensions")
                    elif chunk_type == "grid":
                        print("Using improved enhanced grid chunk processing with advanced mask analysis")
                    else:
                        print("Using improved enhanced strip chunk processing with advanced mask analysis")
                    
                    # Add enhanced chunk parameters
                    process_params['low_res_scale'] = self.app.low_res_scale.get()
                    process_params['mask_skip_threshold'] = self.app.mask_threshold.get()
                    process_params['chunk_type'] = self.app.chunk_type.get()
                    process_params['prioritize_faces'] = bool(self.app.prioritize_faces.get())
                    process_params['use_autochunk'] = bool(self.app.use_autochunk.get())
                    process_params['parallel_processing'] = bool(self.app.use_parallel_processing.get())
                    
                    # Print mask threshold
                    print(f"Using mask coverage threshold of {self.app.mask_threshold.get()}%")
                    if self.app.prioritize_faces.get():
                        print("Face detection is enabled for optimal keyframe selection")
                    if self.app.use_parallel_processing.get():
                        print("Parallel processing is enabled for faster performance")
                    
                    # Add low-res blend method if specifically selected
                    lowres_blend = self.app.lowres_blend_method.get()
                    if lowres_blend and lowres_blend != "(Same as final)":
                        process_params['lowres_blend_method'] = lowres_blend
                        print(f"Using '{lowres_blend}' for low-resolution mask blending")
                    
                    # Run enhanced chunk processing with new implementation
                    result = self.processor.process_video_with_enhanced_chunking(
                        num_chunks=num_chunks,
                        progress_callback=self.progress_callback,
                        **process_params
                    )
                elif num_chunks > 1:
                    # Use standard chunk processing
                    print("Using standard chunk processing")
                    result = self.processor.process_video_in_chunks(
                        num_chunks=num_chunks,
                        progress_callback=self.progress_callback,
                        **process_params
                    )
                else:
                    # Process normally (no chunks)
                    print("Processing video without chunking")
                    result = self.processor.process_video(
                        progress_callback=self.progress_callback,
                        **process_params
                    )
                
                # Process completed successfully
                if not (hasattr(self, 'cancel_processing') and self.cancel_processing):
                    self.update_progress(100, "Processing complete!")
                    self.app.root.after(0, self.processing_complete, *result)
            except KeyboardInterrupt:
                # Handle cancellation requested through our interrupt mechanism
                self.app.root.after(0, self.processing_cancelled)
            except Exception as e:
                # Handle other errors
                self.app.root.after(0, self.processing_error, str(e))
            finally:
                # Reset processing flag
                self.processing = False
                
                # Clear any existing progress monitor timer
                if self.progress_timer_id:
                    self.app.root.after_cancel(self.progress_timer_id)
                    self.progress_timer_id = None
            
        except Exception as e:
            # Update UI in the main thread
            self.app.root.after(0, self.processing_error, str(e))
            self.processing = False
    
    def update_progress(self, percentage, status_text=None, stage=None):
        """
        Update progress bar with percentage and optional status text
        
        Args:
            percentage: Progress percentage (0-100)
            status_text: Optional status text to display
            stage: Optional processing stage description
        """
        if not self.processing:
            return
        
        # Update progress bar
        self.app.progress['value'] = percentage
        
        # Update progress percentage text
        self.app.progress_text.set(f"{int(percentage)}%")
        
        # Update stage indicator if provided
        if stage:
            # Format stage text with a better visual indicator
            self.app.progress_stage.set(f"• {stage}")
        else:
            self.app.progress_stage.set("")
        
        # Update status text if provided
        if status_text:
            self.app.status_var.set(status_text)
        
        # Force UI update
        self.app.root.update_idletasks()
        
    def progress_callback(self, percentage, stage, status):
        """
        Callback function for inference_core progress updates
        
        Args:
            percentage: Progress percentage (0-100)
            stage: Processing stage description
            status: Detailed status message
        """
        # Use update_progress to update the UI with both progress and stage info
        self.update_progress(percentage, status, stage)
    
    def setup_progress_monitoring(self):
        """Setup an improved system to monitor progress by checking console output"""
        # Cancel any existing timer
        if self.progress_timer_id:
            self.app.root.after_cancel(self.progress_timer_id)
        
        # Create a function to check for progress updates in console
        def check_console_for_progress():
            if not self.processing:
                return
            
            # Get the console text
            self.app.console.configure(state="normal")
            console_text = self.app.console.get("1.0", "end")
            self.app.console.configure(state="disabled")
            
            # Pattern to match percentages
            percentage_found = False
            
            # Look for progress indicators in the text
            progress_matches = re.findall(r"(\d+\.\d+|\d+)%", console_text)
            
            if progress_matches:
                try:
                    # Use the most recent percentage found
                    latest_percentage = float(progress_matches[-1])
                    percentage_found = True
                    
                    # Update progress if it's greater than current value
                    current_progress = self.app.progress['value']
                    if latest_percentage > current_progress:
                        self.update_progress(latest_percentage)
                except:
                    pass
            
            # Check for frame processing progress if no direct percentages
            if not percentage_found:
                # Match patterns like "Processed 100/500 frames"
                frame_matches = re.findall(r"Processed (\d+)/(\d+) frames", console_text)
                blending_matches = re.findall(r"Blending progress: (\d+)/(\d+) frames", console_text)
                reassembly_matches = re.findall(r"Reassembled (\d+)/(\d+) frames", console_text)
                
                progress_updates = []
                
                # Calculate progress percentages from different patterns
                if frame_matches:
                    try:
                        current, total = frame_matches[-1]
                        percentage = (float(current) / float(total)) * 100
                        # Scale to 0-85% range for frame processing
                        scaled_percentage = min(85, (percentage * 0.85))
                        progress_updates.append(scaled_percentage)
                    except:
                        pass
                
                if blending_matches:
                    try:
                        current, total = blending_matches[-1]
                        percentage = (float(current) / float(total)) * 100
                        # Scale to 85-95% range for blending
                        scaled_percentage = 85 + ((percentage * 10) / 100)
                        progress_updates.append(scaled_percentage)
                    except:
                        pass
                
                if reassembly_matches:
                    try:
                        current, total = reassembly_matches[-1]
                        percentage = (float(current) / float(total)) * 100
                        # Scale to 85-95% range for reassembly
                        scaled_percentage = 85 + ((percentage * 10) / 100)
                        progress_updates.append(scaled_percentage)
                    except:
                        pass
                
                # Use the highest progress value
                if progress_updates:
                    max_progress = max(progress_updates)
                    current_progress = self.app.progress['value']
                    if max_progress > current_progress:
                        self.update_progress(max_progress)
            
            # Schedule another check after a short delay
            if self.processing:
                # Use a shorter interval for more responsive updates
                self.progress_timer_id = self.app.root.after(100, check_console_for_progress)
        
        # Start the progress monitoring
        self.progress_timer_id = self.app.root.after(200, check_console_for_progress)
    
    def processing_complete(self, foreground_path, alpha_path):
        """
        Called when processing is complete
        
        Args:
            foreground_path: Path to the output foreground video
            alpha_path: Path to the output alpha video
        """
        # Stop progress bar and update UI
        self.app.progress['value'] = 100
        self.app.progress_text.set("100%")
        self.app.progress_stage.set("• Complete")
        self.app.process_button.configure(text="Process Video", state="normal")
        self.app.status_var.set("Processing complete!")
        self.processing = False
        
        # Clear any existing progress monitor timer
        if self.progress_timer_id:
            self.app.root.after_cancel(self.progress_timer_id)
            self.progress_timer_id = None
        
        # Get video quality info for the message
        codec = self.app.video_codec.get()
        quality = self.app.video_quality.get()
        quality_info = f" ({codec} / {quality})" if codec != "Auto" else f" ({quality})"
        
        # Show success message
        messagebox.showinfo(
            "Processing Complete",
            f"Video processing complete{quality_info}!\n\nForeground: {os.path.basename(foreground_path)}\nAlpha Matte: {os.path.basename(alpha_path)}\n\nSaved to: {self.app.output_path.get()}"
        )
        
        # Ask if user wants to open the output folder
        if messagebox.askyesno("Open Folder", "Do you want to open the output folder?"):
            self.open_folder(self.app.output_path.get())
    
    def processing_cancelled(self):
        """Called when processing is cancelled by the user"""
        self.app.progress['value'] = 0
        self.app.progress_text.set("")
        self.app.progress_stage.set("")
        self.app.process_button.configure(text="Process Video", state="normal")
        self.app.status_var.set("Processing cancelled")
        
        # Cleanup temp files if the option is enabled
        if hasattr(self, 'processor') and self.processor and bool(self.app.cleanup_temp.get()):
            try:
                print("Performing cleanup of temporary files after cancellation...")
                # Use the video utils cleanup function directly
                from utils.video_utils import cleanup_temporary_files
                # Search for temp files in the output directory
                output_dir = self.app.output_path.get()
                
                # Look for common temp directories
                import os
                possible_temp_dirs = [
                    os.path.join(output_dir, "enhanced_chunks_temp"),
                    os.path.join(output_dir, "chunks_temp"),
                    os.path.join(output_dir, "video_processor_temp"),
                    os.path.join(output_dir, "temp")
                ]
                
                # Clean up any temp directories that exist
                temp_files = [d for d in possible_temp_dirs if os.path.exists(d)]
                if temp_files:
                    cleanup_temporary_files(temp_files, True)
                    print("Temporary files cleanup completed after cancellation")
                else:
                    print("No temporary directories found to clean up")
            except Exception as e:
                print(f"Error during cleanup after cancellation: {str(e)}")
                
        messagebox.showinfo("Cancelled", "Processing has been cancelled by the user")
        self.processing = False
        
        # Clear any existing progress monitor timer
        if self.progress_timer_id:
            self.app.root.after_cancel(self.progress_timer_id)
            self.progress_timer_id = None
    
    def processing_error(self, error_message):
        """
        Called when processing encounters an error
        
        Args:
            error_message: The error message to display
        """
        self.app.progress['value'] = 0
        self.app.progress_text.set("")
        self.app.progress_stage.set("• Error")
        self.app.process_button.configure(text="Process Video", state="normal")
        self.app.status_var.set("Error occurred")
        
        # Clear any existing progress monitor timer
        if self.progress_timer_id:
            self.app.root.after_cancel(self.progress_timer_id)
            self.progress_timer_id = None
        
        # Create a more user-friendly error message
        friendly_message = error_message
        
        # Common error patterns and more helpful messages
        if "CUDA out of memory" in error_message:
            friendly_message = (
                "GPU memory error: Out of CUDA memory.\n\n"
                "Try one of the following options:\n"
                "1. Reduce the Max Size parameter\n"
                "2. Increase the number of Chunks\n"
                "3. Enable Enhanced Chunk Processing\n"
                "4. Use a smaller input video\n"
                "5. Try a different codec/quality setting"
            )
        elif "MPS out of memory" in error_message:
            friendly_message = (
                "Metal Performance Shaders (MPS) memory error.\n\n"
                "Try one of the following options:\n"
                "1. Reduce the Max Size parameter\n"
                "2. Increase the number of Chunks\n"
                "3. Enable Enhanced Chunk Processing\n"
                "4. Use a smaller input video\n"
                "5. Try a different codec/quality setting"
            )
        elif "ffmpeg" in error_message.lower():
            friendly_message = (
                "Error with video encoding (ffmpeg problem).\n\n"
                "Please ensure ffmpeg is properly installed on your system.\n"
                "For Mac users: install with 'brew install ffmpeg'\n"
                "For Windows users: download from https://ffmpeg.org/download.html\n\n"
                "You can also try different codec/quality settings."
            )
        elif "Sizes of tensors must match" in error_message:
            friendly_message = (
                "Tensor dimension mismatch error.\n\n"
                "This can happen when processing video chunks with certain dimensions.\n\n"
                "Try these options:\n"
                "1. Use a different number of chunks\n" 
                "2. Try Grid chunks instead of Strips or vice versa\n"
                "3. Set Max Size to 512 instead of -1\n"
                "4. Enable Enhanced Chunk Processing\n"
                "5. Process without chunks (set Number of Chunks to 1)"
            )
        
        messagebox.showerror("Processing Error", f"An error occurred:\n\n{friendly_message}")
        self.processing = False
    
    def open_folder(self, path):
        """
        Open a folder in the system file explorer
        
        Args:
            path: Path to the folder to open
        """
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":  # macOS
            import subprocess
            subprocess.call(["open", path])
        else:  # Linux
            import subprocess
            subprocess.call(["xdg-open", path])
