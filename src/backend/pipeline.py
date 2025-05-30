from .model import YoloModel, DetrModel
from .model_config import ModelConfig
from . import preprocessor
from . import processor

import pandas as pd

def pipeline(model):
    # preprocessor
    model.config.results_path = preprocessor.make_results_dir(model.config.results_path)
    images, videos = preprocessor.create_data_lists(model.config.data_path)

    #processor and postprocessor (postprocessor is called by processor)
    if images:
        df_img = processor.process_imgs(model, images)
    if videos:
        df_vid = processor.process_videos(model, videos)
    all_data = processor.get_all_data(images, videos)

    if images and videos and all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(model.config.results_path / 'all_saved_frames.csv', index = False)
        return df
    if images and not videos:
        return df_img
    if videos and not images:
        return df_vid
    
    info = {
        'file_name': str(name),
        'frame_num': index,
        'num_objs': len(conf_list),
        'confs': conf_list[0] if len(conf_list) == 1 else conf_list,
        'timestamp': 'ph'
        }

    df = pd.DataFrame(columns = ['file_name', 'frame_num', 'num_objs', 'confs', 'timestamp'])
    return df
        

def yolo_pipeline(config):
    model = YoloModel(config)
    pipeline(model)

def detr_pipeline(config):
    model = DetrModel(config) 
    pipeline(model)

if __name__ == '__main__':
    config = ModelConfig(data_path='/Dataset/valid/images')
    yolo_pipeline(config)
