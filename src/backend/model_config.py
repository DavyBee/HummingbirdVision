from .util import *

from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class ModelConfig:
    data_path:    Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    results_path: Path = field(default_factory=lambda: DEFAULT_RESULTS_DIR / "job1")
    weights_dir:  Path = field(default_factory=lambda: DEFAULT_WEIGHTS_DIR)
    
    conf:         float = 0.35
    iou:          float = 0.001
    batch_size:   int   = 32

    save_frames_csv: bool = True
    save_all_csv: bool = True

    def __post_init__(self):
        # ensure absolute Paths
        self.data_path    = Path(self.data_path)
        self.results_path = Path(self.results_path)
        self.weights_dir  = Path(self.weights_dir)

        # sanity-check that data and weights folders exist and contain files
        if self.data_path.exists():

            contains_data = False
            for file in self.data_path.rglob('*'):
                if file.suffix.lower() in IMAGE_TYPES or file.suffix.lower() in VIDEO_TYPES:
                    contains_data = True
                    break

            if contains_data is False:
                raise AssertionError(
                    f"No data found at {self.data_path}. "
                    f"Drop your videos/images into this folder."
                )
        else:
            raise AssertionError(
                f"No folder found at {self.data_path}. "
                f"Create this folder then drop your videos/images into it."
            )
        if not self.weights_dir.exists():
            raise AssertionError(
                f"No model weights found at {self.weights_dir}. "
                f"Place your .pt or files there."
            )