## ANRO-Video-to-CSV
This project extracts and interprets reactor telemetry data from control-room footage.
It uses GPU-accelerated OCR (EasyOCR) and adaptive preprocessing to extract values like temperature, pressure, and fuel percentage from video frames.

## Features
- Frame-by-frame OCR with confidence tracking
- GPU acceleration via PyTorch
- Automatic smoothing and correction for noise
- Exports clean CSV data for formula analysis

## Usage
1. Place your video file (`.mkv` or `.mp4`) in the project root.
2. Run `dataGrabber.py` inside your virtual environment:
   ```bash
   python Data\ Handler\dataGrabber.py
3. The processed data will be saved as reactor_readings_cleaned_new.csv.

## Requirements

Install dependencies:

pip install -r requirements.txt
