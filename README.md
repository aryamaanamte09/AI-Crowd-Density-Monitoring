# AI-Based Crowd Density Monitoring System

An AI-powered crowd density monitoring application developed as part of an Embedded Systems Architecture project.

The system uses computer vision and object tracking to detect and monitor people and other selected objects in video streams, estimate crowd density, and display the results through an interactive Streamlit interface.

## Project Overview

This project explores the integration of multiple technologies into a single working application, combining object detection, multi-object tracking, video processing, GPU acceleration, and an interactive user interface.

## Technology Stack

- **Python** – Core programming language
- **YOLO26n (Ultralytics)** – Object detection
- **ByteTrack** – Multi-object tracking
- **OpenCV** – Video processing and annotation
- **Streamlit** – Interactive application interface
- **PyTorch and CUDA** – GPU-accelerated inference

## Key Features

- Object detection in uploaded videos
- Multi-object tracking with ByteTrack
- Custom ByteTrack configuration for improved tracking continuity during temporary occlusions
- Crowd density classification
- Annotated video output with bounding boxes and tracking information
- Summary statistics for detected objects
- GPU acceleration using CUDA when available
- Live-camera detection, if supported by the configured environment

## Project Structure

```text
AI-Crowd-Density-Monitoring/
├── source_code/
├── dataset_samples/
├── input_output_samples/
├── execution_proof/
├── USER_GUIDE.md
├── README.md
└── .gitignore
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/aryamaanamte09/AI-Crowd-Density-Monitoring.git
```

2. Navigate to the project directory.

3. Create and activate a virtual environment.

4. Install the dependencies:

```bash
pip install -r source_code/requirements.txt
```

5. Install the appropriate CUDA-enabled PyTorch build separately if GPU acceleration is required.

6. Ensure the YOLO model weights are available at the path expected by the application.

## Running the Application

```bash
streamlit run source_code/app.py
```

Adjust the command if your final project structure or application entry point differs.

## Performance

The standalone CPU-versus-GPU benchmark processed 1,165 frames:

| Execution mode | Processing time |
|---|---:|
| CPU | 56.80 seconds |
| CUDA GPU | 29.64 seconds |

This represents approximately 1.92× speedup in the standalone benchmark. The complete Streamlit workflow took approximately 63 seconds in the observed test. These are different measurements and should not be directly compared.

## Documentation and Samples

Refer to `USER_GUIDE.md` for setup, execution, and testing instructions.

Representative inputs, outputs, and execution evidence are provided in their respective folders.

## Learning Outcomes

This project provided practical experience in integrating multiple software components, working with pretrained computer vision models, implementing object tracking, debugging application workflows, and optimising inference performance.

It was a valuable first major project in understanding how individual technologies can be combined into a functional end-to-end system.

## Author

Aryamaan Amte
