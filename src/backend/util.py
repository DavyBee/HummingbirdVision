from pathlib import Path

# defines the image types and video types allowed in the data folder
IMAGE_TYPES = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
VIDEO_TYPES = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.tlv']

# Default project folders to hold data, output results, and hold model weights #
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_DIR    = PROJECT_ROOT / "data"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "model_weights"
