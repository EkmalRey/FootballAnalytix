"""
Configuration Panel Component for adjusting processing parameters.
"""

from typing import Callable, Optional
import customtkinter as ctk


class ConfigPanel(ctk.CTkFrame):
    """
    Configuration panel for adjusting detection and visualization parameters.
    """
    
    def __init__(
        self,
        master,
        on_config_changed: Optional[Callable[[dict], None]] = None,
        **kwargs
    ):
        super().__init__(master, **kwargs)
        
        self.on_config_changed = on_config_changed
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the configuration UI."""
        # Configure grid
        self.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self,
            text="⚙️ Configuration",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w"
        )
        self.title_label.grid(row=0, column=0, padx=10, pady=(5, 0), sticky="w", columnspan=4)
        
        # Confidence slider
        self.conf_label = ctk.CTkLabel(self, text="Confidence:")
        self.conf_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        
        self.conf_slider = ctk.CTkSlider(
            self,
            from_=0.1,
            to=0.95,
            number_of_steps=85,
            command=self._on_conf_changed
        )
        self.conf_slider.set(0.65)
        self.conf_slider.grid(row=1, column=1, padx=5, pady=10, sticky="ew")
        
        self.conf_value_label = ctk.CTkLabel(self, text="0.65", width=40)
        self.conf_value_label.grid(row=1, column=2, padx=5, pady=10, sticky="w")
        
        # Spacer
        spacer = ctk.CTkLabel(self, text="")
        spacer.grid(row=1, column=3, padx=10, sticky="ew")
        
        # Checkbox frame
        self.checkbox_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.checkbox_frame.grid(row=2, column=0, columnspan=4, padx=10, pady=(0, 10), sticky="ew")
        self.checkbox_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        # Toggle checkboxes
        self.detection_var = ctk.BooleanVar(value=True)
        self.detection_cb = ctk.CTkCheckBox(
            self.checkbox_frame,
            text="Detection",
            variable=self.detection_var,
            command=self._notify_change
        )
        self.detection_cb.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.teams_var = ctk.BooleanVar(value=True)
        self.teams_cb = ctk.CTkCheckBox(
            self.checkbox_frame,
            text="Teams",
            variable=self.teams_var,
            command=self._notify_change
        )
        self.teams_cb.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        self.ids_var = ctk.BooleanVar(value=True)
        self.ids_cb = ctk.CTkCheckBox(
            self.checkbox_frame,
            text="Player IDs",
            variable=self.ids_var,
            command=self._notify_change
        )
        self.ids_cb.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        
        self.keypoints_var = ctk.BooleanVar(value=True)
        self.keypoints_cb = ctk.CTkCheckBox(
            self.checkbox_frame,
            text="Keypoints",
            variable=self.keypoints_var,
            command=self._notify_change
        )
        self.keypoints_cb.grid(row=0, column=3, padx=5, pady=5, sticky="w")
    
    def _on_conf_changed(self, value):
        """Handle confidence slider change."""
        self.conf_value_label.configure(text=f"{value:.2f}")
        self._notify_change()
    
    def _notify_change(self):
        """Notify callback of configuration change."""
        if self.on_config_changed:
            self.on_config_changed(self.get_config())
    
    def get_config(self) -> dict:
        """Get current configuration as dictionary."""
        return {
            "confidence": self.conf_slider.get(),
            "show_object_detection": self.detection_var.get(),
            "show_clustering": self.teams_var.get(),
            "show_id": self.ids_var.get(),
            "show_detected_keypoints": self.keypoints_var.get(),
        }
    
    def set_config(self, config: dict):
        """Set configuration from dictionary."""
        if "confidence" in config:
            self.conf_slider.set(config["confidence"])
            self.conf_value_label.configure(text=f"{config['confidence']:.2f}")
        if "show_object_detection" in config:
            self.detection_var.set(config["show_object_detection"])
        if "show_clustering" in config:
            self.teams_var.set(config["show_clustering"])
        if "show_id" in config:
            self.ids_var.set(config["show_id"])
        if "show_detected_keypoints" in config:
            self.keypoints_var.set(config["show_detected_keypoints"])
