#!/usr/bin/env python
"""
FootballAnalytix GUI Launcher

Professional Desktop Application for Football Analytics.
"""

import sys
import logging
import os
from pathlib import Path

# Suppress YOLO logs
os.environ['YOLO_VERBOSE'] = 'False'
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Ensure project root is in path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication
from gui.app import MainWindow


def main():
    """Launch the Professional FootballAnalytix GUI."""
    print("Starting FootballAnalytix Pro...")
    
    app = QApplication(sys.argv)
    
    # Set style properties
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
