from .model_config import ModelConfig

from ultralytics import YOLO
from ultralytics import RTDETR
from typing import Any, Optional

# larger class model, specific models inherit from this structure
class UltralyticsModel:
    def __init__(self, config: Optional[ModelConfig]):
        self.config = config or ModelConfig()
        self.model: Any = None
        self.model_type: str = ''

    def __repr__(self):
        return f'Model Type: {self.model_type} --- {self.config}'
    
    def predict(self, batch_tup):
        results_list = []
        fnames, batch = zip(*batch_tup)
        fnames = list(fnames)
        batch = list(batch)
    
        results = self.model.predict(
                    source=batch, 
                    verbose=False,
                    conf=self.config.conf, 
                    iou=self.config.iou, 
                    batch=self.config.batch_size
                )
        
        #making dict to store file name, overhead low
        for name, result in zip(fnames, results):
            tup = (name, result)
            results_list.append(tup)

        return results_list


class YoloModel(UltralyticsModel):
    def __init__(self, config: Optional[ModelConfig]):
        self.config = config or ModelConfig()

        weights_file = self.config.weights_dir / "YOLO.pt"
        self.model = YOLO(str(weights_file))

        self.model_type = 'YOLO'

class DetrModel(UltralyticsModel):
    def __init__(self, config: Optional[ModelConfig]):
        self.config = config or ModelConfig()

        weights_file = self.config.weights_dir / "DETR.pt"
        self.model = RTDETR(str(weights_file))

        self.model_type = 'RTDETR'
