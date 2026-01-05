"""
File Browser Component for selecting images and videos.
"""

from pathlib import Path
from typing import Optional, Callable
from tkinter import filedialog
import customtkinter as ctk


class FileBrowser(ctk.CTkFrame):
    """
    File browser component for selecting image or video files.
    """
    
    def __init__(
        self,
        master,
        on_file_selected: Optional[Callable[[Path], None]] = None,
        **kwargs
    ):
        super().__init__(master, **kwargs)
        
        self.on_file_selected = on_file_selected
        self.selected_path: Optional[Path] = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the file browser UI."""
        # Configure grid
        self.grid_columnconfigure(1, weight=1)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self,
            text="📁 File",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w"
        )
        self.title_label.grid(row=0, column=0, padx=10, pady=(5, 0), sticky="w", columnspan=3)
        
        # Browse button
        self.browse_btn = ctk.CTkButton(
            self,
            text="Select Image/Video",
            command=self._browse_file,
            width=150
        )
        self.browse_btn.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        
        # Path display
        self.path_label = ctk.CTkLabel(
            self,
            text="No file selected",
            anchor="w",
            text_color="gray"
        )
        self.path_label.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        
        # File type indicator
        self.type_label = ctk.CTkLabel(
            self,
            text="",
            width=60
        )
        self.type_label.grid(row=1, column=2, padx=10, pady=10, sticky="e")
    
    def _browse_file(self):
        """Open file dialog and handle selection."""
        filetypes = [
            ("All supported", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv"),
            ("Images", "*.jpg *.jpeg *.png *.bmp"),
            ("Videos", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*"),
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select Image or Video",
            filetypes=filetypes
        )
        
        if filepath:
            self.set_path(Path(filepath))
    
    def set_path(self, path: Path):
        """Set the selected file path."""
        self.selected_path = path
        
        # Update display
        display_path = str(path)
        if len(display_path) > 50:
            display_path = "..." + display_path[-47:]
        self.path_label.configure(text=display_path, text_color=("gray10", "gray90"))
        
        # Update type indicator
        suffix = path.suffix.lower()
        if suffix in ['.jpg', '.jpeg', '.png', '.bmp']:
            self.type_label.configure(text="🖼️ Image")
        elif suffix in ['.mp4', '.avi', '.mov', '.mkv']:
            self.type_label.configure(text="🎬 Video")
        else:
            self.type_label.configure(text="📄 File")
        
        # Callback
        if self.on_file_selected:
            self.on_file_selected(path)
    
    def get_path(self) -> Optional[Path]:
        """Get the currently selected path."""
        return self.selected_path
    
    def is_video(self) -> bool:
        """Check if selected file is a video."""
        if self.selected_path is None:
            return False
        return self.selected_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']
    
    def is_image(self) -> bool:
        """Check if selected file is an image."""
        if self.selected_path is None:
            return False
        return self.selected_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
