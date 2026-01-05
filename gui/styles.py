"""
Professional High-End Dark Theme for FootballAnalytix
"""

STYLESHEET = """
/* --- Global Reset & Fonts --- */
QWidget {
    font-family: 'Segoe UI', 'Roboto', sans-serif;
    font-size: 14px;
    color: #e4e4e7; /* Zinc-200 */
    background-color: transparent;
}

/* --- Main Window --- */
QMainWindow, QWidget#CentralWidget {
    background-color: #09090b; /* Zinc-950 - Very dark, almost black */
}

/* --- Sidebar --- */
QFrame#Sidebar {
    background-color: #121214; /* Slightly lighter than main bg */
    border-right: 1px solid #27272a; /* Zinc-800 */
}

/* --- Brand/Title --- */
QLabel#BrandLabel {
    color: #ffffff;
    font-size: 24px;
    font-weight: 800;
    letter-spacing: 1px;
}

QLabel#SubBrandLabel {
    color: #00ff9d; /* Neon Mint */
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 3px;
    margin-bottom: 10px;
}

/* --- Separators --- */
QFrame#Separator {
    color: #27272a;
    background-color: #27272a;
    border: none;
    height: 1px;
}

/* --- Headers --- */
QLabel#SectionHeader {
    color: #71717a; /* Zinc-500 */
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 10px;
}

/* --- Panels (Control Groups) --- */
QFrame#ControlPanel {
    background-color: transparent;
    border: none;
}

/* --- Buttons --- */
/* Primary Action Button (RUN INFERENCE) */
QPushButton#PrimaryButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00ff9d, stop:1 #00cc7a);
    color: #000000;
    font-weight: 700;
    font-size: 14px;
    border-radius: 6px;
    border: none;
}

QPushButton#PrimaryButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #33ffb0, stop:1 #00e68a);
}

QPushButton#PrimaryButton:pressed {
    background-color: #00b36e;
    margin-top: 1px;
}

QPushButton#PrimaryButton:disabled {
    background-color: #27272a;
    color: #52525b;
}

/* Secondary Button (Load Media) */
QPushButton#SecondaryButton {
    background-color: #18181b; /* Zinc-900 */
    color: #e4e4e7;
    border: 1px solid #27272a;
    border-radius: 6px;
    height: 38px;
    font-weight: 600;
}

QPushButton#SecondaryButton:hover {
    background-color: #27272a; /* Zinc-800 */
    border-color: #3f3f46;
}

QPushButton#SecondaryButton:pressed {
    background-color: #09090b;
}

/* Media Controls Button (Play) */
QPushButton#PlayButton {
    background-color: #00ff9d;
    color: #000000;
    border-radius: 20px; /* Circle */
    font-size: 18px;
    padding-bottom: 2px; /* Visual center adjustment */
}

QPushButton#PlayButton:hover {
    background-color: #33ffb0;
}

/* --- Input Fields & Labels --- */
QLabel#StatusLabel {
    color: #52525b; /* Zinc-600 */
    font-size: 12px;
    font-style: italic;
    border: 1px dashed #27272a;
    border-radius: 4px;
    padding: 8px;
}

/* --- Sliders --- */
QSlider::groove:horizontal {
    border: none;
    height: 4px;
    background: #27272a;
    margin: 2px 0;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: #00ff9d;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #00ff9d;
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 9px;
}

QSlider::handle:horizontal:hover {
    background: #00ff9d;
    border-color: #ffffff;
}

QLabel#ValueLabel {
    color: #00ff9d;
    font-weight: 700;
    font-family: 'Consolas', 'Monaco', monospace;
}

/* --- Checkboxes --- */
QCheckBox {
    color: #d4d4d8;
    spacing: 12px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    background: #18181b;
    border: 1px solid #52525b;
    border-radius: 4px;
}

QCheckBox::indicator:hover {
    border-color: #00ff9d;
    background: #27272a;
}

QCheckBox::indicator:checked {
    background-color: #00ff9d;
    border-color: #00ff9d;
    /* We can use a pure CSS checkmark or rely on default symbol if color contrasts well.
       Since we can't easily embed images without files, we'll style the background. */
    image: url(none); /* Remove default */
}
/* Trick: Use a pseudo-element logic or background color to show checked state plainly if icon missing */

/* --- Right Panel (Stats) --- */
QFrame#StatsPanel {
    background-color: #09090b;
    border-left: 1px solid #27272a;
}

QFrame#SubPanel {
    background-color: #121214;
    border: 1px solid #27272a;
    border-radius: 8px;
}

QLabel#PanelHeader {
    color: #ffffff;
    font-weight: 700;
    font-size: 13px;
    border-bottom: 2px solid #27272a;
    padding-bottom: 10px;
    margin-bottom: 5px;
}

/* --- Video Area & Controls --- */
QWidget#ContentArea {
    background-color: #000000; /* Pure black backing for video */
}

QFrame#VideoPanel {
    background-color: #000000;
    border: 1px solid #27272a;
    border-radius: 8px;
}

QFrame#ControlsBar {
    background-color: #121214;
    border-top: 1px solid #27272a;
    border-bottom-left-radius: 8px;
    border-bottom-right-radius: 8px;
}

QLabel#TimeLabel {
    font-family: 'Consolas', 'Monaco', monospace;
    font-weight: 600;
    color: #a1a1aa;
}

/* --- Scrollbars (Webkit-style for Qt) --- */
QScrollBar:vertical {
    border: none;
    background: #09090b;
    width: 8px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #3f3f46;
    min-height: 20px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #52525b;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* --- Status Bar --- */
QStatusBar {
    background-color: #18181b;
    color: #71717a;
    border-top: 1px solid #27272a;
}

/* --- Progress Bar --- */
QProgressBar {
    background-color: #18181b;
    border: 1px solid #27272a;
    border-radius: 4px;
    height: 24px;
    text-align: center;
    color: #e4e4e7;
    font-weight: 600;
    font-size: 12px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00cc7a, stop:1 #00ff9d);
    border-radius: 3px;
}

/* --- Stop Button (Danger Style) --- */
QPushButton#StopButton, QPushButton#SecondaryButton[text="STOP"] {
    background-color: #dc2626;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    height: 38px;
    font-weight: 600;
}

QPushButton#StopButton:hover {
    background-color: #ef4444;
}

QPushButton#StopButton:pressed {
    background-color: #b91c1c;
}

/* --- Navigation Buttons (Prev/Next) --- */
QPushButton#NavButton {
    font-size: 14px;
    background-color: transparent;
    border: 1px solid #3f3f46;
    border-radius: 18px; /* Circle since 36x36 */
}

QPushButton#NavButton:hover {
    background-color: #27272a;
    border-color: #00ff9d;
}

QPushButton#NavButton:pressed {
    background-color: #3f3f46;
}

/* --- FPS Label --- */
QLabel#FpsLabel {
    font-family: 'Consolas', 'Monaco', monospace;
    font-weight: 700;
    color: #00ff9d;
    background-color: #18181b;
    border-radius: 4px;
    padding: 4px 8px;
}

/* --- System Metrics (Status Bar) --- */
QLabel#SystemMetricLabel {
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 11px;
    padding: 0 8px;
    font-weight: 600;
}

/* --- Processing Status Label --- */
QLabel#ProcessingStatus {
    color: #a1a1aa;
    font-style: italic;
    font-family: 'Segoe UI', sans-serif;
    font-size: 11px;
}

/* --- Toggle Switch Enhancement --- */
QCheckBox#ToggleSwitch::indicator {
    width: 36px;
    height: 20px;
    border-radius: 10px;
    background: #27272a;
    border: 1px solid #3f3f46;
}

QCheckBox#ToggleSwitch::indicator:checked {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00cc7a, stop:1 #00ff9d);
    border-color: #00ff9d;
}

QCheckBox#ToggleSwitch::indicator:hover {
    border-color: #00ff9d;
}

/* --- Splitter Handle --- */
QSplitter::handle {
    background-color: #27272a;
}

QSplitter::handle:hover {
    background-color: #00ff9d;
}

/* --- Tooltips --- */
QToolTip {
    background-color: #27272a;
    color: #e4e4e7;
    border: 1px solid #3f3f46;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
}

/* --- DisplayFrame Border Glow Effect --- */
QFrame#DisplayFrame {
    border: 2px solid #27272a;
    border-radius: 8px;
    background-color: #000000;
}

QFrame#DisplayFrame[active="true"] {
    border-color: #00ff9d;
}
"""
