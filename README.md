## ANRO-Video-to-CSV
This project extracts and interprets reactor telemetry data from control-room footage.

## Features
- Frame-by-frame OCR with confidence tracking
- Exports clean CSV data for formula analysis

## Usage
1. Place your video file (`.mkv` or `.mp4`) in the project root.
2. Run `dataGrabber.py` inside your virtual environment.
3. The processed data will be saved as reactor_readings_cleaned.csv.

## Requirements

Install dependencies:

pip install -r requirements.txt
