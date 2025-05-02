# TrafficDT 

## Requirements

### Required Libraries
```
opencv-python
ultralytics
scipy
numpy
```

### Additional Software
- Python 3.x

## Step-by-Step Setup and Execution

### 1. Environment Setup

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install opencv-python ultralytics scipy numpy
```

### 2. YOLO Model Setup

Ensure you have the YOLO model file (`best_1.pt`) in your project root directory. This is used for object detection and tracking.

### 3. Prepare Video Data

**IMPORTANT:** All video files MUST be placed in the `./Data` folder. The system is configured to read from this location.

The default code is configured to use:
```
./Data/Bellevue_116th_NE12th__2017-09-11_09-08-31.mp4
```

If you want to use a different video file, you can modify the `video_path` variable in the main function in the script.

### 4. Create Output Directories

Make sure the following directories exist (the script will create them if they don't):
- `frames` - for storing processed video frames
- `sumo_files` - for storing generated SUMO configuration files

### 5. Run the Script

```bash
python script_with_export_interval.py
```

During execution, the script will:
- Display the processed video with tracking information
- Save frames to the `frames` folder
- Generate SUMO XML files at set intervals (default: every 2 minutes)

### 6. SUMO Simulation

After the script has finished generating the SUMO files:


## Project Structure

- `script_with_export_interval.py`: Main script for video processing and SUMO file generation
- `model_data.py`: Module for data preparation (not included in the snippet)
- `Data/`: Directory for input video files
- `frames/`: Output directory for processed video frames
- `sumo_files/`: Output directory for generated SUMO configuration files
