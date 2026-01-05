# FootballAnalytix

**FootballAnalytix** is a comprehensive, modular computer vision platform designed for football (soccer) and futsal data analysis. It leverages state-of-the-art deep learning models to extract meaningful insights from match footage, including player tracking, team classification, and tactical visualization.

> **Note**: This repository serves as the clean, modularized version of the original codebase. Complete data and legacy code can be found in the previous repository.

## 🚀 Features

-   **Multi-Object Tracking**: Robustly detects and tracks players, referees, and the ball using YOLO-based models.
-   **Team Classification**: Automatically clusters players into teams based on jersey colors using unsupervised learning (KMeans).
-   **Pitch Calibration (Homography)**: Detects field keypoints to compute the homography matrix, mapping 2D video frames to a standard 2D pitch coordinate system.
-   **Minimap Visualization**: Generates a real-time top-down 2D minimap showing player positions and tactical movements.
-   **Modular Design**: Structured as a reusable Python package (`footballanalytix`) for easy integration and scalability.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/EkmalRey/FootballAnalytix.git
    cd FootballAnalytix
    ```

2.  **Install dependencies**:
    It is recommended to use a virtual environment (e.g., `venv` or `conda`).
    ```bash
    pip install -r requirements.txt
    ```
    *Core dependencies include: `ultralytics`, `torch`, `opencv-python`, `pandas`, `scikit-learn`, and `supervision`.*

## 📂 Project Structure

```
FootballAnalytix/
├── footballanalytix/       # Main Python package
│   ├── pipeline.py         # Core processing logic
│   ├── field.py            # Field detection & homography transformation
│   ├── players.py          # Player detection & jersey color clustering
│   ├── visualization.py    # Drawing utilities & minimap rendering
│   ├── config.py           # Configuration parameters and constants
│   └── ...
├── Pipelines/              # Jupyter Notebooks & demo scripts
│   ├── _1_Combined.ipynb   # Full end-to-end pipeline walk-through
│   └── ...
├── Models/                 # Directory for storing YOLO models (*.pt)
├── Data/                   # Directory for input videos and datasets
└── requirements.txt        # Python dependency list
```

## 💻 Usage

You can use the library directly in your Python scripts or run the provided notebooks.

### Quick Start (Python)

```python
from footballanalytix import (
    load_players_model,
    load_field_model,
    CombinedState,
    process_combined_frame,
    initialize_combined_state
)
import cv2
from pathlib import Path

# 1. Setup paths and load models
# Ensure your YOLO models are in the 'Models/' directory
video_path = Path("Data/your_video.mp4")
players_model = load_players_model() 
field_model = load_field_model()

# 2. Initialize State (collects initial team colors from video)
state = initialize_combined_state(video_path, players_model)

# 3. Process Video Frame-by-Frame
cap = cv2.VideoCapture(str(video_path))
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Process the frame
    result = process_combined_frame(
        frame=frame,
        frame_idx=frame_idx,
        state=state,
        players_model=players_model,
        field_model=field_model
    )
    
    # 'result' contains the annotated frame, minimap, and stats
    cv2.imshow("Analysis", result["frame"])
    cv2.imshow("Minimap", result["minimap"])
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
```

### Running Notebooks
Check the `Pipelines/` directory for interactive examples. 
- **`_1_Combined.ipynb`** is the recommended starting point to see the full pipeline in action.

## ⚙️ Configuration
You can adjust parameters such as confidence thresholds, team colors, and minimap settings in `footballanalytix/config.py`.

## 🤝 Contributing
Contributions are welcome! Please ensure you have the required models downloaded and placed in the `Models/` folder before running tests.