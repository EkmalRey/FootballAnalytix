"""
Main Application Window using PyQt6.

Features:
- Start/Stop/Pause inference controls with immediate response
- Live preview with minimap and stats
- Frame navigation (prev/next) using cache
- Live toggle for all visualization options
- Proper debouncing to avoid CPU overload
"""

from pathlib import Path
import logging
import cv2
import numpy as np
import psutil
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QSlider, 
    QCheckBox, QFrame, QSplitter, QStatusBar, QProgressBar,
    QSizePolicy, QSpacerItem
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSlot, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QAction, QFont, QPixmap

from .styles import STYLESHEET
from .utils.image_utils import cv2_to_pixmap
from .controllers.inference_controller import InferenceController
from .workers import ModelLoaderWorker, InferenceWorker, ContinuousInferenceWorker
from .components.video_widget import VideoDisplayWidget


class MainWindow(QMainWindow):
    """
    Main application window for FootballAnalytix Pro.
    
    Key design decisions:
    - Single-frame inference uses request IDs to cancel stale requests
    - Scrubbing/seeking does NOT auto-trigger inference (only on release)
    - Cache is used for instant playback of processed frames
    - Continuous inference has immediate pause/stop response
    """
    
    # Signal to request inference in the worker thread
    # Args: frame, frame_idx, file_path, request_id
    request_inference = pyqtSignal(np.ndarray, int, object, int)

    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("FootballAnalytix Pro")
        self.resize(1600, 900)
        
        self.setWindowIcon(QIcon("gui/resources/logo.png"))
        
        # Apply professional theme
        self.setStyleSheet(STYLESHEET)
        
        # Core Components
        self.controller = InferenceController()
        self.video_capture = None
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame_index = 0
        self.is_playing = False
        self.current_file_path = None
        self.was_playing = False
        
        # Frame Display State
        self.current_frame_display = None
        self.current_raw_frame = None
        
        # Inference State
        self.is_inference_running = False
        self.inference_paused = False
        self.processed_frame_count = 0
        
        # Request ID for cancelling stale inference requests
        self._inference_request_id = 0
        
        # Slider state - track if user is actively scrubbing
        self._slider_is_being_dragged = False
        
        # Frame Cache for Replay (stores processed results)
        self.frame_cache = {}  # frame_idx -> {'frame': ..., 'minimap': ..., 'stats': ...}
        self.max_cache_size = 500
        
        # Debounce timer for config changes
        self._config_debounce_timer = QTimer()
        self._config_debounce_timer.setSingleShot(True)
        self._config_debounce_timer.timeout.connect(self._apply_config_changes_debounced)
        
        # UI Setup
        self._setup_ui()
        
        # Threading for Single-Frame Inference
        self.inference_thread = QThread()
        self.inference_worker = InferenceWorker(self.controller)
        self.inference_worker.moveToThread(self.inference_thread)
        
        # Connect signals for single-frame processing
        self.request_inference.connect(self._handle_inference_request)
        self.inference_worker.finished.connect(self._on_inference_complete)
        self.inference_worker.error.connect(self._on_inference_error)
        self.inference_thread.start()
        
        # Continuous Inference Worker (created on demand)
        self.continuous_worker = None
        
        # Timer for playback
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._play_next_frame)
        
        # Timer for system metrics (update every 2 seconds)
        self._system_metrics_timer = QTimer()
        self._system_metrics_timer.timeout.connect(self._update_system_metrics)
        self._system_metrics_timer.start(2000)
        
        # Load models
        self.status_bar.showMessage("Initializing models...")
        self.loading_progress.setVisible(True)
        
        self.loader_worker = ModelLoaderWorker(self.controller)
        self.loader_worker.finished.connect(self._on_models_loaded)
        self.loader_worker.progress.connect(lambda msg: self.status_bar.showMessage(msg))
        self.loader_worker.start()

    @pyqtSlot(np.ndarray, int, object, int)
    def _handle_inference_request(self, frame: np.ndarray, frame_idx: int, file_path, request_id: int):
        """Handle inference request - forwards to worker with request ID."""
        self.inference_worker.set_request_id(request_id)
        self.inference_worker.process_frame(frame, frame_idx, file_path, request_id)

    def _setup_ui(self):
        """Setup the main UI layout."""
        # Central Widget
        central_widget = QWidget()
        central_widget.setObjectName("CentralWidget")
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # --- Sidebar (Controls) ---
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(320)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setSpacing(20)
        sidebar_layout.setContentsMargins(25, 30, 25, 30)
        
        # Title / Brand (Updated with Logo)
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)
        title_layout.setSpacing(10)
        title_layout.setContentsMargins(0, 0, 0, 0)
        
        # Logo
        logo_label = QLabel()
        logo_pixmap = QPixmap("gui/resources/logo.png")
        if not logo_pixmap.isNull():
            # Scale logo to fit nicely
            scaled_logo = logo_pixmap.scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(scaled_logo)
            logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(logo_label)
        
        brand_lbl = QLabel("FOOTBALL")
        brand_lbl.setObjectName("BrandLabel")
        brand_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        sub_brand_lbl = QLabel("ANALYTIX PRO")
        sub_brand_lbl.setObjectName("SubBrandLabel")
        sub_brand_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        title_layout.addWidget(brand_lbl)
        title_layout.addWidget(sub_brand_lbl)
        sidebar_layout.addWidget(title_container)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setObjectName("Separator")
        sidebar_layout.addWidget(line)

        # File Controls
        file_group = QFrame()
        file_group.setObjectName("ControlPanel")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(10)
        
        lbl_media = QLabel("MEDIA SOURCE")
        lbl_media.setObjectName("SectionHeader")
        file_layout.addWidget(lbl_media)
        
        self.load_btn = QPushButton("IMPORT MEDIA")
        self.load_btn.setObjectName("SecondaryButton")
        self.load_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.load_btn.clicked.connect(self._load_file)
        self.load_btn.setEnabled(False)  # Wait for models
        
        self.file_lbl = QLabel("No media selected")
        self.file_lbl.setObjectName("StatusLabel")
        self.file_lbl.setWordWrap(True)
        
        file_layout.addWidget(self.load_btn)
        file_layout.addWidget(self.file_lbl)
        sidebar_layout.addWidget(file_group)
        
        # Configuration
        config_group = QFrame()
        config_group.setObjectName("ControlPanel")
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(12)
        
        config_layout.addWidget(QLabel("CONFIDENCE THRESHOLD", objectName="SectionHeader"))
        
        slider_container = QHBoxLayout()
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(10, 95)  # 0.10 to 0.95
        self.conf_slider.setValue(65)
        self.conf_slider.setObjectName("ModernSlider")
        self.conf_slider.valueChanged.connect(self._on_confidence_changed)
        self.conf_slider.sliderReleased.connect(self._schedule_config_update)
        
        self.conf_val_lbl = QLabel("0.65")
        self.conf_val_lbl.setObjectName("ValueLabel")
        self.conf_val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.conf_val_lbl.setFixedWidth(50)
        
        slider_container.addWidget(self.conf_slider)
        slider_container.addWidget(self.conf_val_lbl)
        config_layout.addLayout(slider_container)
        
        config_layout.addWidget(QLabel("VISUALIZATION LAYERS", objectName="SectionHeader"))
        
        self.check_detect = QCheckBox("Object Detection")
        self.check_teams = QCheckBox("Team Clustering")
        self.check_ids = QCheckBox("Player Identifiers")
        self.check_keypoints = QCheckBox("Keypoint Detection")
        
        for chk in [self.check_detect, self.check_teams, self.check_ids, self.check_keypoints]:
            chk.setChecked(True)
            chk.setCursor(Qt.CursorShape.PointingHandCursor)
            
        # Connect signals
        self.check_detect.toggled.connect(self._on_detection_toggled)
        self.check_teams.toggled.connect(self._schedule_config_update)
        self.check_ids.toggled.connect(self._schedule_config_update)
        self.check_keypoints.toggled.connect(self._schedule_config_update)
        
        config_layout.addWidget(self.check_detect)
        config_layout.addWidget(self.check_teams)
        config_layout.addWidget(self.check_ids)
        config_layout.addWidget(self.check_keypoints)
        
        sidebar_layout.addWidget(config_group)
        
        sidebar_layout.addStretch()
        
        # Action Buttons
        action_group = QFrame()
        action_layout = QVBoxLayout(action_group)
        action_layout.setSpacing(10)
        action_layout.setContentsMargins(0, 0, 0, 0)
        
        self.auto_process_chk = QCheckBox("Real-time Inference")
        self.auto_process_chk.setObjectName("ToggleSwitch")
        self.auto_process_chk.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_process_chk.setToolTip("Automatically run inference during playback (uses more CPU)")
        
        # Primary action button (Start/Stop)
        self.process_btn = QPushButton("START ANALYSIS")
        self.process_btn.setObjectName("PrimaryButton")
        self.process_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.process_btn.setFixedHeight(50)
        self.process_btn.clicked.connect(self._toggle_inference)
        self.process_btn.setEnabled(False)
        
        # Stop button 
        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setObjectName("SecondaryButton")
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.clicked.connect(self._stop_inference)
        self.stop_btn.setVisible(False)
        
        action_layout.addWidget(self.auto_process_chk)
        action_layout.addWidget(self.process_btn)
        action_layout.addWidget(self.stop_btn)
        
        sidebar_layout.addWidget(action_group)
        
        # --- Main Content Area ---
        content_area = QWidget()
        content_area.setObjectName("ContentArea")
        content_layout = QVBoxLayout(content_area)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(20)
        
        # Video/Minimap Splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(2)
        
        # Video Player
        video_container = QFrame()
        video_container.setObjectName("VideoPanel")
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(0)
        
        # Video Display Area
        self.display_frame = QFrame()
        self.display_frame.setObjectName("DisplayFrame")
        display_layout = QVBoxLayout(self.display_frame)
        display_layout.setContentsMargins(2, 2, 2, 2)
        
        self.video_display = VideoDisplayWidget()
        display_layout.addWidget(self.video_display)
        video_layout.addWidget(self.display_frame)
        
        # Controls Bar
        controls_container = QFrame()
        controls_container.setObjectName("ControlsBar")
        controls_container.setFixedHeight(70)
        controls_layout = QHBoxLayout(controls_container)
        controls_layout.setContentsMargins(15, 5, 15, 5)
        controls_layout.setSpacing(10)
        
        # Navigation buttons
        # Navigation buttons
        self.prev_btn = QPushButton()
        self.prev_btn.setIcon(QIcon("gui/resources/prev.svg"))
        self.prev_btn.setObjectName("NavButton")
        self.prev_btn.setFixedSize(36, 36)
        self.prev_btn.setIconSize(QSize(20, 20))
        self.prev_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.prev_btn.clicked.connect(self._prev_frame)
        self.prev_btn.setToolTip("Previous Frame")
        
        self.play_btn = QPushButton()
        self.play_btn.setIcon(QIcon("gui/resources/play.svg"))
        self.play_btn.setObjectName("PlayButton")
        self.play_btn.setFixedSize(48, 48) # Slightly larger
        self.play_btn.setIconSize(QSize(24, 24))
        self.play_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.play_btn.clicked.connect(self._toggle_playback)
        
        self.next_btn = QPushButton()
        self.next_btn.setIcon(QIcon("gui/resources/next.svg"))
        self.next_btn.setObjectName("NavButton")
        self.next_btn.setFixedSize(36, 36)
        self.next_btn.setIconSize(QSize(20, 20))
        self.next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_btn.clicked.connect(self._next_frame)
        self.next_btn.setToolTip("Next Frame")
        
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setObjectName("TimeSlider")
        self.time_slider.setRange(0, 0)
        self.time_slider.setCursor(Qt.CursorShape.PointingHandCursor)
        self.time_slider.sliderPressed.connect(self._on_slider_pressed)
        self.time_slider.sliderReleased.connect(self._on_slider_released)
        self.time_slider.sliderMoved.connect(self._on_slider_moved)
        
        self.frame_lbl = QLabel("00:00 / 00:00")
        self.frame_lbl.setObjectName("TimeLabel")
        self.frame_lbl.setFixedWidth(120)
        self.frame_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # FPS indicator
        self.fps_lbl = QLabel("-- FPS")
        self.fps_lbl.setObjectName("TimeLabel")
        self.fps_lbl.setFixedWidth(80)
        self.fps_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        controls_layout.addWidget(self.prev_btn)
        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.next_btn)
        controls_layout.addWidget(self.time_slider)
        controls_layout.addWidget(self.frame_lbl)
        controls_layout.addWidget(self.fps_lbl)
        
        video_layout.addWidget(controls_container)
        
        self.splitter.addWidget(video_container)
        
        # Minimap & Stats (Right Side)
        right_panel = QFrame()
        right_panel.setObjectName("StatsPanel")
        right_panel.setFixedWidth(380)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(20)
        right_layout.setContentsMargins(20, 20, 20, 20)
        
        # Minimap Section
        minimap_group = QFrame()
        minimap_group.setObjectName("SubPanel")
        minimap_layout = QVBoxLayout(minimap_group)
        minimap_layout.setContentsMargins(15, 15, 15, 15)
        
        minimap_header = QLabel("TACTICAL MAP")
        minimap_header.setObjectName("PanelHeader")
        minimap_layout.addWidget(minimap_header)
        
        self.minimap_display = VideoDisplayWidget()
        self.minimap_display.setMinimumSize(300, 220)
        self.minimap_display.setStyleSheet("background-color: #121212; border-radius: 4px;")
        minimap_layout.addWidget(self.minimap_display)
        
        right_layout.addWidget(minimap_group)
        
        # Stats Section
        stats_group = QFrame()
        stats_group.setObjectName("SubPanel")
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setContentsMargins(15, 15, 15, 15)
        stats_layout.setSpacing(15)
        
        stats_layout.addWidget(QLabel("LIVE METRICS", objectName="PanelHeader"))
        
        # Stat Rows
        self.stat_team0 = QLabel("0")
        self.stat_team1 = QLabel("0")
        self.stat_tracked = QLabel("0")
        self.stat_progress = QLabel("0%")
        
        stats_layout.addLayout(self._create_stat_row("Team A", self.stat_team0, "#4cc9f0"))
        stats_layout.addLayout(self._create_stat_row("Team B", self.stat_team1, "#f72585"))
        # stats_layout.addLayout(self._create_stat_row("Tracked Entities", self.stat_tracked, "#dec0f0")) # Removed per user request
        stats_layout.addLayout(self._create_stat_row("Progress", self.stat_progress, "#00ff9d"))
        
        right_layout.addWidget(stats_group)
        
        # Processing Progress
        progress_group = QFrame()
        progress_group.setObjectName("SubPanel")
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setContentsMargins(15, 15, 15, 15)
        progress_layout.setSpacing(10)
        
        progress_layout.addWidget(QLabel("PROCESSING", objectName="PanelHeader"))
        
        self.process_progress = QProgressBar()
        self.process_progress.setRange(0, 100)
        self.process_progress.setValue(0)
        self.process_progress.setTextVisible(True)
        self.process_progress.setFormat("%p% - %v/%m frames")
        progress_layout.addWidget(self.process_progress)
        
        self.process_status_lbl = QLabel("Ready")
        self.process_status_lbl.setStyleSheet("color: #888;")
        progress_layout.addWidget(self.process_status_lbl)
        
        right_layout.addWidget(progress_group)
        right_layout.addStretch()
        
        self.splitter.addWidget(right_panel)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)
        
        content_layout.addWidget(self.splitter)
        
        # Add to main layout
        main_layout.addWidget(sidebar)
        main_layout.addWidget(content_area)
        
        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # System Metrics Labels (Left side)
        self.cpu_label = QLabel("CPU: --%")
        self.cpu_label.setObjectName("SystemMetricLabel")
        
        self.ram_label = QLabel("RAM: --%")
        self.ram_label.setObjectName("SystemMetricLabel")
        
        self.disk_label = QLabel("DISK: --%")
        self.disk_label.setObjectName("SystemMetricLabel")
        
        # Add to status bar (left side)
        self.status_bar.addWidget(self.cpu_label)
        self.status_bar.addWidget(self.ram_label)
        self.status_bar.addWidget(self.disk_label)
        
        # Loading progress (right side)
        self.loading_progress = QProgressBar()
        self.loading_progress.setFixedWidth(200)
        self.loading_progress.setRange(0, 0)
        self.status_bar.addPermanentWidget(self.loading_progress)
        
    def _create_stat_row(self, label_text, value_label, color):
        """Create a styled stat row."""
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setStyleSheet("color: #888888; font-weight: 500;")
        
        value_label.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: bold;")
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        row.addWidget(lbl)
        row.addWidget(value_label)
        return row

    # =========================================================================
    # Model Loading
    # =========================================================================

    def _on_models_loaded(self, success, message):
        """Handle model loading completion."""
        self.loading_progress.setVisible(False)
        self.status_bar.showMessage(message)
        if success:
            self.load_btn.setEnabled(True)
            # self.process_btn.setEnabled(True) # ONLY enable if file is loaded too.
            if self.current_file_path is not None:
                self.process_btn.setEnabled(True)

    def _update_system_metrics(self):
        """Update system resource usage labels."""
        try:
            # CPU usage (averaged over interval)
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_label.setText(f"CPU: {cpu_percent:5.1f}%")
            
            # Memory usage
            memory = psutil.virtual_memory()
            ram_percent = memory.percent
            ram_used_gb = memory.used / (1024 ** 3)
            ram_total_gb = memory.total / (1024 ** 3)
            self.ram_label.setText(f"RAM: {ram_percent:5.1f}% ({ram_used_gb:.1f}/{ram_total_gb:.1f}GB)")
            
            # Disk usage (current working drive)
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            self.disk_label.setText(f"DISK: {disk_percent:5.1f}%")
            
            # Color coding based on usage (only affects color, font comes from CSS)
            if cpu_percent > 80:
                self.cpu_label.setStyleSheet("color: #ff4444;")
            elif cpu_percent > 50:
                self.cpu_label.setStyleSheet("color: #ffaa00;")
            else:
                self.cpu_label.setStyleSheet("color: #4cc9f0;")
            
            if ram_percent > 80:
                self.ram_label.setStyleSheet("color: #ff4444;")
            elif ram_percent > 60:
                self.ram_label.setStyleSheet("color: #ffaa00;")
            else:
                self.ram_label.setStyleSheet("color: #f72585;")
                
        except Exception as e:
            # Silently ignore errors (disk might not be accessible, etc.)
            pass

    # =========================================================================
    # File Loading
    # =========================================================================

    def _load_file(self):
        """Open file dialog and load video."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv);;Images (*.jpg *.png)"
        )
        
        if file_path:
            self.current_file_path = Path(file_path)
            self.file_lbl.setText(self.current_file_path.name)
            
            # Stop any running inference FIRST
            self._stop_inference()
            
            # Reset Video
            if self.video_capture:
                self.video_capture.release()
            
            # Clear frame cache
            self.frame_cache.clear()
            
            # Reset request ID
            self._inference_request_id = 0
                
            self.video_capture = cv2.VideoCapture(str(self.current_file_path))
            self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_capture.get(cv2.CAP_PROP_FPS) or 30.0
            
            self.time_slider.setRange(0, self.total_frames - 1)
            self.current_frame_index = 0
            self.processed_frame_count = 0
            
            # Update progress bar range
            self.process_progress.setMaximum(self.total_frames)
            self.process_progress.setValue(0)
            
            # Show first frame (raw, no inference)
            self._display_raw_frame_at(0)
            
            # Enable controls
            self.play_btn.setEnabled(True)
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
            if self.controller.models_loaded:
                self.process_btn.setEnabled(True)
            
            self.status_bar.showMessage(f"Loaded: {self.current_file_path.name} ({self.total_frames} frames @ {self.fps:.1f} FPS)")

    def _on_detection_toggled(self, checked):
        """Handle object detection toggle - disables dependent features."""
        self.check_teams.setEnabled(checked)
        if not checked:
            self.check_teams.setChecked(False)
        self._schedule_config_update()

    # =========================================================================
    # Frame Display - CORE LOGIC
    # =========================================================================

    def _display_raw_frame_at(self, frame_idx: int):
        """
        Display a RAW frame (no inference) at given index.
        Used during scrubbing to avoid CPU overload.
        """
        if not self.video_capture:
            return
        
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        
        # If we have a cached processed frame, use that
        if frame_idx in self.frame_cache:
            cached = self.frame_cache[frame_idx]
            if cached.get('frame') is not None:
                self.video_display.set_frame(cv2_to_pixmap(cached['frame']))
                if cached.get('minimap') is not None:
                    self.minimap_display.set_frame(cv2_to_pixmap(cached['minimap']))
                if cached.get('stats'):
                    self._update_stats_display(cached['stats'])
                self.current_frame_index = frame_idx
                self._update_time_label()
                return
        
        # Read from video (no inference)
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.video_capture.read()
        
        if ret:
            self.current_raw_frame = frame.copy()
            self.video_display.set_frame(cv2_to_pixmap(frame))
            self.current_frame_index = frame_idx
            self._update_time_label()

    def _display_frame_with_inference(self, frame_idx: int):
        """
        Display frame and run inference on it.
        Used for single-frame processing (not during scrubbing).
        """
        if not self.video_capture:
            return
        
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        
        # Check cache first
        if frame_idx in self.frame_cache:
            cached = self.frame_cache[frame_idx]
            if cached.get('frame') is not None:
                self.video_display.set_frame(cv2_to_pixmap(cached['frame']))
                if cached.get('minimap') is not None:
                    self.minimap_display.set_frame(cv2_to_pixmap(cached['minimap']))
                if cached.get('stats'):
                    self._update_stats_display(cached['stats'])
                self.current_frame_index = frame_idx
                self._update_time_label()
                return
        
        # Read from video
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.video_capture.read()
        
        if ret:
            self.current_raw_frame = frame.copy()
            self.video_display.set_frame(cv2_to_pixmap(frame))
            self.current_frame_index = frame_idx
            self._update_time_label()
            
            # Run inference
            self._run_single_inference()

    def _update_time_label(self):
        """Update the time display label."""
        current_sec = int(self.current_frame_index / self.fps)
        total_sec = int(self.total_frames / self.fps)
        self.frame_lbl.setText(
            f"{current_sec//60:02d}:{current_sec%60:02d} / {total_sec//60:02d}:{total_sec%60:02d}"
        )

    # =========================================================================
    # Playback Controls
    # =========================================================================

    def _toggle_playback(self):
        """Toggle video playback."""
        if self.is_playing:
            self._pause_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        """Start video playback."""
        if self.video_capture is None:
            return
        self.playback_timer.start(int(1000 / self.fps))
        self.play_btn.setIcon(QIcon("gui/resources/pause.svg"))
        self.is_playing = True

    def _pause_playback(self):
        """Pause video playback."""
        self.playback_timer.stop()
        self.play_btn.setIcon(QIcon("gui/resources/play.svg"))
        self.is_playing = False

    def _play_next_frame(self):
        """
        Advance to next frame during playback.
        
        IMPORTANT: Playback NEVER triggers inference automatically.
        It only shows:
        - Cached processed frames if available
        - Raw frames if not cached
        
        To run inference, user must explicitly click START ANALYSIS.
        """
        if self.current_frame_index < self.total_frames - 1:
            next_idx = self.current_frame_index + 1
            
            # ALWAYS use raw/cached frames during playback - NO automatic inference
            self._display_raw_frame_at(next_idx)
            
            # Update slider without triggering callbacks
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(next_idx)
            self.time_slider.blockSignals(False)
        else:
            self._pause_playback()

    def _prev_frame(self):
        """Go to previous frame."""
        if self.current_frame_index > 0:
            self._display_raw_frame_at(self.current_frame_index - 1)
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(self.current_frame_index)
            self.time_slider.blockSignals(False)

    def _next_frame(self):
        """Go to next frame."""
        if self.current_frame_index < self.total_frames - 1:
            self._display_raw_frame_at(self.current_frame_index + 1)
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(self.current_frame_index)
            self.time_slider.blockSignals(False)

    # =========================================================================
    # Slider Handling - Crucial for avoiding CPU overload
    # =========================================================================

    def _on_slider_pressed(self):
        """User started dragging slider - pause playback if playing."""
        self._slider_is_being_dragged = True
        if self.is_playing:
            self.was_playing = True
            self._pause_playback()
        else:
            self.was_playing = False

    def _on_slider_moved(self, value: int):
        """
        Slider is being moved - show raw frames only (no inference).
        This is critical for avoiding CPU overload during scrubbing.
        """
        if self._slider_is_being_dragged:
            self._display_raw_frame_at(value)

    def _on_slider_released(self):
        """
        User released slider - resume playback if needed.
        
        IMPORTANT: Does NOT trigger inference. User must explicitly
        click START ANALYSIS to run inference.
        """
        self._slider_is_being_dragged = False
        
        # Resume playback if it was playing before
        if self.was_playing:
            self._start_playback()

    # =========================================================================
    # Configuration
    # =========================================================================

    def _on_confidence_changed(self, value):
        """Update confidence threshold display."""
        self.conf_val_lbl.setText(f"{value/100:.2f}")
    
    def _schedule_config_update(self):
        """Schedule a config update with debouncing."""
        # Cancel any pending update
        self._config_debounce_timer.stop()
        # Schedule new update in 200ms
        self._config_debounce_timer.start(200)
    
    def _apply_config_changes_debounced(self):
        """
        Apply config changes after debounce.
        
        Updates the controller settings and clears the cache.
        Does NOT auto-trigger inference - user must explicitly run it.
        """
        # Update controller config
        cfg = {
            "confidence": self.conf_slider.value() / 100.0,
            "show_object_detection": self.check_detect.isChecked(),
            "show_clustering": self.check_teams.isChecked(),
            "show_id": self.check_ids.isChecked(),
            "show_detected_keypoints": self.check_keypoints.isChecked(),
        }
        self.controller.update_config(**cfg)
        
        # Invalidate cache (settings changed)
        self.frame_cache.clear()
        
        # Show status message
        self.status_bar.showMessage("Settings updated - cache cleared", 2000)

    # =========================================================================
    # Single Frame Inference
    # =========================================================================

    def _run_single_inference(self):
        """Run inference on the current frame with request cancellation."""
        if self.current_raw_frame is None:
            return
        
        # Increment request ID to cancel any pending older requests
        self._inference_request_id += 1
        
        # Trigger worker via signal (thread-safe)
        self.request_inference.emit(
            self.current_raw_frame.copy(),
            self.current_frame_index,
            self.current_file_path,
            self._inference_request_id
        )

    # =========================================================================
    # Continuous Inference
    # =========================================================================

    def _toggle_inference(self):
        """Toggle continuous inference processing."""
        if self.is_inference_running:
            if self.inference_paused:
                self._resume_inference()
            else:
                self._pause_inference()
        else:
            self._start_inference()

    def _start_inference(self):
        """Start continuous inference processing."""
        if self.current_file_path is None:
            self.status_bar.showMessage("No video loaded", 3000)
            return
        
        if not self.controller.models_loaded:
            self.status_bar.showMessage("Models not loaded", 3000)
            return
        
        # Stop playback
        if self.is_playing:
            self._pause_playback()
        
        # Enable Glow Effect
        self.display_frame.setProperty("active", True)
        self.display_frame.style().polish(self.display_frame)
        
        # Create new worker if needed
        if self.continuous_worker is None or not self.continuous_worker.isRunning():
            self.continuous_worker = ContinuousInferenceWorker(self.controller)
            self.continuous_worker.set_video(self.current_file_path)
            self.continuous_worker.set_start_frame(self.current_frame_index)
            
            # Connect signals
            self.continuous_worker.frame_processed.connect(self._on_continuous_frame)
            self.continuous_worker.progress.connect(self._on_continuous_progress)
            self.continuous_worker.status_changed.connect(self._on_continuous_status)
            self.continuous_worker.error.connect(self._on_inference_error)
            self.continuous_worker.finished_all.connect(self._on_inference_finished)
            self.continuous_worker.paused.connect(self._on_inference_paused)
            
            self.continuous_worker.start()
        
        self.is_inference_running = True
        self.inference_paused = False
        
        # Update UI
        self.process_btn.setText("PAUSE")
        self.stop_btn.setVisible(True)
        self.process_status_lbl.setText("Processing...")
        self.status_bar.showMessage("Inference running...")

    def _pause_inference(self):
        """Pause continuous inference - immediate response."""
        if self.continuous_worker is not None:
            self.continuous_worker.request_pause()
            # Wait for actual pause (up to 500ms)
            self.continuous_worker.wait_for_pause(500)
        
        self.inference_paused = True
        self.process_btn.setText("RESUME")
        self.process_status_lbl.setText("Paused")
        self.status_bar.showMessage("Inference paused")

    @pyqtSlot()
    def _on_inference_paused(self):
        """Called when worker actually pauses."""
        self.inference_paused = True
        self.process_btn.setText("RESUME")

    def _resume_inference(self):
        """Resume paused inference."""
        if self.continuous_worker is not None:
            self.continuous_worker.request_resume()
            self.inference_paused = False
            self.process_btn.setText("PAUSE")
            self.process_status_lbl.setText("Processing...")
            self.status_bar.showMessage("Inference resumed")

    def _stop_inference(self):
        """Stop continuous inference completely - immediate response."""
        if self.continuous_worker is not None:
            self.continuous_worker.request_stop()
            # Wait for thread to finish
            if not self.continuous_worker.wait(1000):
                # Force terminate if not responding
                self.continuous_worker.terminate()
                self.continuous_worker.wait(500)
            self.continuous_worker = None
        
        self.is_inference_running = False
        self.inference_paused = False
        
        # Disable Glow Effect
        self.display_frame.setProperty("active", False)
        self.display_frame.style().polish(self.display_frame)
        
        # Update UI
        self.process_btn.setText("START ANALYSIS")
        self.stop_btn.setVisible(False)
        self.process_status_lbl.setText("Stopped")
        self.status_bar.showMessage("Inference stopped")

    @pyqtSlot(int, dict)
    def _on_continuous_frame(self, frame_idx: int, result: dict):
        """Handle processed frame from continuous worker."""
        # Cache the result
        self._cache_result(frame_idx, result)
        
        # Update display
        if result.get('frame') is not None:
            self.video_display.set_frame(cv2_to_pixmap(result['frame']))
        
        if result.get('minimap') is not None:
            self.minimap_display.set_frame(cv2_to_pixmap(result['minimap']))
        
        if result.get('stats'):
            self._update_stats_display(result['stats'])
        
        # Update slider position
        self.current_frame_index = frame_idx
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(frame_idx)
        self.time_slider.blockSignals(False)
        self._update_time_label()

    @pyqtSlot(int, int, float)
    def _on_continuous_progress(self, current: int, total: int, fps: float):
        """Handle progress update from continuous worker."""
        self.processed_frame_count = current
        self.process_progress.setValue(current)
        
        progress_pct = (current / total * 100) if total > 0 else 0
        self.stat_progress.setText(f"{progress_pct:.1f}%")
        self.fps_lbl.setText(f"{fps:.1f} FPS")

    @pyqtSlot(str)
    def _on_continuous_status(self, status: str):
        """Handle status change from continuous worker."""
        self.process_status_lbl.setText(status)

    @pyqtSlot()
    def _on_inference_finished(self):
        """Handle inference completion."""
        self.is_inference_running = False
        self.inference_paused = False
        
        # Disable Glow Effect
        self.display_frame.setProperty("active", False)
        self.display_frame.style().polish(self.display_frame)
        
        self.process_btn.setText("REPLAY")
        self.stop_btn.setVisible(False)
        self.process_status_lbl.setText("Completed")
        self.status_bar.showMessage("Inference completed!")

    # =========================================================================
    # Result Handling
    # =========================================================================

    def _cache_result(self, frame_idx: int, result: dict):
        """Cache processed frame result with LRU eviction."""
        # Enforce cache size limit
        while len(self.frame_cache) >= self.max_cache_size:
            # Remove oldest entry (first key)
            oldest_key = next(iter(self.frame_cache))
            del self.frame_cache[oldest_key]
        
        self.frame_cache[frame_idx] = {
            'frame': result.get('frame'),
            'minimap': result.get('minimap'),
            'stats': result.get('stats'),
        }

    @pyqtSlot(dict)
    def _on_inference_complete(self, result):
        """Handle single-frame inference completion."""
        # Cache the result
        self._cache_result(self.current_frame_index, result)
        
        # Update Main View
        if result.get('frame') is not None:
            self.video_display.set_frame(cv2_to_pixmap(result['frame']))
             
        # Update Minimap
        if result.get('minimap') is not None:
            self.minimap_display.set_frame(cv2_to_pixmap(result['minimap']))
            
        # Update Stats
        if result.get('stats'):
            self._update_stats_display(result['stats'])

    def _update_stats_display(self, stats: dict):
        """Update the stats panel with new data."""
        self.stat_team0.setText(str(stats.get('team_0', 0)))
        self.stat_team1.setText(str(stats.get('team_1', 0)))
        self.stat_tracked.setText(str(stats.get('tracked', 0)))

    @pyqtSlot(str)
    def _on_inference_error(self, err_msg):
        """Handle inference error."""
        self.status_bar.showMessage(f"Error: {err_msg}", 5000)

    # =========================================================================
    # Cleanup
    # =========================================================================

    def closeEvent(self, event):
        """Handle window close - cleanup threads properly."""
        # Stop continuous worker first
        if self.continuous_worker is not None:
            self.continuous_worker.request_stop()
            if not self.continuous_worker.wait(1000):
                self.continuous_worker.terminate()
            self.continuous_worker = None
        
        # Stop inference thread
        self.inference_thread.quit()
        self.inference_thread.wait(1000)
        
        # Release video
        if self.video_capture:
            self.video_capture.release()
        
        event.accept()
