"""
# Updated: Saturday, May 10, 2025
UI helper components for MatAnyone GUI.
"""

import tkinter as tk
from tkinter import ttk


class TextRedirector:
    """
    Class for redirecting stdout to a text widget
    """
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = ""
        self.max_lines = 1000  # Maximum number of lines to keep in the console
        self.update_scheduled = False
        self.original_stdout = None
        if hasattr(tk, 'sys') and hasattr(tk.sys, '__stdout__'):
            self.original_stdout = tk.sys.__stdout__

    def write(self, string):
        # Don't process empty strings
        if not string:
            return
            
        self.buffer += string
        # Only schedule an update if one isn't already pending
        if not self.update_scheduled:
            self.update_scheduled = True
            self.text_widget.after(10, self.update_text_widget)
    
    def update_text_widget(self):
        try:
            if self.buffer:
                self.text_widget.configure(state=tk.NORMAL)
                self.text_widget.insert(tk.END, self.buffer)
                
                # Limit the number of lines in the console
                line_count = int(self.text_widget.index('end-1c').split('.')[0])
                if line_count > self.max_lines:
                    self.text_widget.delete('1.0', f'{line_count-self.max_lines}.0')
                    
                self.text_widget.see(tk.END)
                self.text_widget.configure(state=tk.DISABLED)
                self.buffer = ""
        except Exception as e:
            if self.original_stdout:
                print(f"Error updating text widget: {str(e)}", file=self.original_stdout)
        finally:
            self.update_scheduled = False
            # If there's more in the buffer, schedule another update
            if self.buffer:
                self.text_widget.after(10, self.update_text_widget)
                self.update_scheduled = True
    
    def flush(self):
        # Make sure to flush any remaining content
        if self.buffer and not self.update_scheduled:
            self.text_widget.after(0, self.update_text_widget)
            self.update_scheduled = True


class Tooltip:
    """
    Simple tooltip implementation for tkinter widgets
    """
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        widget.bind("<Enter>", self.show_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)
    
    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        # Create a toplevel window
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(self.tooltip_window, text=self.text, 
                      background="#ffffe0", relief="solid", borderwidth=1,
                      font=("TkDefaultFont", "8", "normal"))
        label.pack(ipadx=2, ipady=2)
    
    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class ProgressDialog:
    """
    Progress dialog with cancel capability
    """
    def __init__(self, parent, title="Processing", message="Processing...", cancellable=True):
        self.parent = parent
        self.cancellable = cancellable
        self.cancelled = False
        self.on_cancel_callback = None
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("400x150")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Center the dialog on parent
        self.center_on_parent()
        
        # Create widgets
        self.message_label = tk.Label(self.dialog, text=message, pady=10)
        self.message_label.pack(fill=tk.X)
        
        self.progress_bar = ttk.Progressbar(self.dialog, mode='indeterminate', length=300)
        self.progress_bar.pack(pady=10, padx=20)
        
        # Create cancel button if cancellable
        if cancellable:
            self.cancel_button = ttk.Button(self.dialog, text="Cancel", command=self.cancel)
            self.cancel_button.pack(pady=10)
    
    def center_on_parent(self):
        """Center the dialog on the parent window"""
        self.dialog.update_idletasks()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.dialog.geometry(f"+{x}+{y}")
    
    def start(self):
        """Start the progress animation"""
        self.progress_bar.start(15)
    
    def stop(self):
        """Stop the progress animation"""
        self.progress_bar.stop()
    
    def update_message(self, message):
        """Update the progress message"""
        self.message_label.config(text=message)
        self.dialog.update_idletasks()
    
    def update_progress(self, value, maximum=100):
        """Update the progress bar value"""
        if self.progress_bar['mode'] == 'indeterminate':
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate', maximum=maximum)
        
        self.progress_bar['value'] = value
        self.dialog.update_idletasks()
    
    def close(self):
        """Close the dialog"""
        self.dialog.destroy()
    
    def set_on_cancel(self, callback):
        """Set a callback for when the user cancels"""
        self.on_cancel_callback = callback
    
    def cancel(self):
        """Handle cancel button click"""
        self.cancelled = True
        if self.on_cancel_callback:
            self.on_cancel_callback()
        
        # Disable the cancel button after cancellation
        if self.cancellable:
            self.cancel_button.config(state=tk.DISABLED)
            self.message_label.config(text="Cancelling... Please wait")
    
    def on_close(self):
        """Handle window close attempt"""
        # If cancellable, treat closing as cancel
        if self.cancellable and not self.cancelled:
            self.cancel()
        # Otherwise, do nothing (prevent closing)


class CustomSlider(ttk.Frame):
    """
    Custom slider widget with optional spinbox (value label removed)
    """
    def __init__(self, parent, variable, from_=0, to=100, length=200, show_value=False, show_spinbox=False):
        super().__init__(parent)
        
        self.variable = variable
        self.from_ = from_
        self.to = to
        
        # Create slider
        self.slider = ttk.Scale(
            self, 
            from_=from_, 
            to=to, 
            orient=tk.HORIZONTAL, 
            length=length,
            variable=variable,
            command=self._on_slider_change
        )
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Create value label if needed (now optional and disabled by default)
        if show_value:
            self.value_label = ttk.Label(self, width=3, anchor=tk.CENTER)
            self.value_label.pack(side=tk.LEFT, padx=(5, 0))
            self._update_value_label()
        
        # Create spinbox if needed
        if show_spinbox:
            self.spinbox = ttk.Spinbox(
                self,
                from_=from_,
                to=to,
                textvariable=variable,
                width=5,
                command=self._on_spinbox_change
            )
            self.spinbox.pack(side=tk.LEFT, padx=(5, 0))
            
            # Bind spinbox to update slider
            self.spinbox.bind('<Return>', self._on_spinbox_change)
            self.spinbox.bind('<FocusOut>', self._on_spinbox_change)
    
    def _on_slider_change(self, value):
        """Handle slider changes"""
        # Update the variable
        try:
            value = int(float(value))
            self.variable.set(value)
        except:
            pass
        
        # Update value label if it exists
        if hasattr(self, 'value_label'):
            self._update_value_label()
    
    def _on_spinbox_change(self, event=None):
        """Handle spinbox changes"""
        try:
            value = int(self.variable.get())
            # Constrain to valid range
            value = max(self.from_, min(self.to, value))
            self.variable.set(value)
            
            # Update value label if it exists
            if hasattr(self, 'value_label'):
                self._update_value_label()
                
            # Update slider position
            self.slider.set(value)
        except:
            pass
    
    def _update_value_label(self):
        """Update the value label with the current value"""
        self.value_label.configure(text=str(self.variable.get()))


def create_message_dialog(parent, title, message, button_text="OK", button_command=None):
    """Create a simple message dialog"""
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    dialog.grab_set()
    
    # Configure dialog size
    dialog.geometry("500x350")
    
    # Center the dialog on parent
    dialog.update_idletasks()
    parent_width = parent.winfo_width()
    parent_height = parent.winfo_height()
    parent_x = parent.winfo_rootx()
    parent_y = parent.winfo_rooty()
    
    dialog_width = dialog.winfo_width()
    dialog_height = dialog.winfo_height()
    
    x = parent_x + (parent_width - dialog_width) // 2
    y = parent_y + (parent_height - dialog_height) // 2
    
    dialog.geometry(f"+{x}+{y}")
    
    # Make dialog resizable
    dialog.resizable(True, True)
    
    # Configure column and row weights
    dialog.columnconfigure(0, weight=1)
    dialog.rowconfigure(0, weight=1)
    
    # Create a frame for the message with scrollbar support
    message_frame = ttk.Frame(dialog)
    message_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
    message_frame.columnconfigure(0, weight=1)
    message_frame.rowconfigure(0, weight=1)
    
    # Create scrolled text widget for the message
    text_widget = tk.Text(message_frame, wrap=tk.WORD, width=60, height=15)
    text_widget.grid(row=0, column=0, sticky="nsew")
    
    # Add scrollbar
    scrollbar = ttk.Scrollbar(message_frame, orient=tk.VERTICAL, command=text_widget.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    text_widget['yscrollcommand'] = scrollbar.set
    
    # Insert message text
    text_widget.insert(tk.END, message)
    text_widget.configure(state=tk.DISABLED)
    
    # Create button frame
    button_frame = ttk.Frame(dialog)
    button_frame.grid(row=1, column=0, pady=10)
    
    # Create button
    def on_close():
        if button_command:
            button_command()
        dialog.destroy()
    
    button = ttk.Button(button_frame, text=button_text, command=on_close)
    button.pack(padx=10)
    
    # Handle window close
    dialog.protocol("WM_DELETE_WINDOW", on_close)
    
    return dialog
