## ANRO-Video-to-CSV
This project extracts and interprets reactor telemetry data from control-room footage.

## Features
- Frame-by-frame OCR with confidence tracking
- Exports clean CSV data for formula analysis

## Usage
1. Place your video file (`.mkv` or `.mp4`) in the project root.
3. Run `boundaryFinder.py` inside your virtual environment, follow the instructions in the console.
4. Run `dataGrabber.py` inside your virtual environment.
5. The processed data will be saved as reactor_readings_cleaned.csv.

## Requirements
Python Version 3.12

Install dependencies:

pip install -r requirements.txt

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
