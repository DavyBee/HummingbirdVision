from .util import IMAGE_TYPES, VIDEO_TYPES

from pathlib import Path
import cv2
import math

def make_results_dir(results_path):
    if results_path.exists():
            for i in range(1, 10000000):
                test_path = results_path.parent / f'job{i}'
                if not test_path.exists():
                    results_path = test_path
                    break

    results_path.mkdir(parents=True, exist_ok=True) #making results folder
    return results_path

def create_data_lists(data_path):
    images = []
    videos = []

    if data_path.is_dir():
        for file in data_path.rglob('*'):
            if file.is_file() and file.suffix.lower() in IMAGE_TYPES:
                images.append(file)
            if file.is_file() and file.suffix.lower() in VIDEO_TYPES:
                videos.append(file)
    elif data_path.is_file():
        file = data_path
        if file.suffix.lower() in IMAGE_TYPES:
            images.append(file)
        if file.suffix.lower() in VIDEO_TYPES:
            videos.append(file)

    return images, videos

# def get_total_batches(vid_path, batch_size):
#     cap = cv2.VideoCapture(str(vid_path), cv2.CAP_FFMPEG)

#     if not cap.isOpened():
#         raise IOError("Couldn't open video")
    
#     num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

#     num_batches = num_frames / batch_size
#     return math.ceil(num_batches)

def get_batch_vid(vid: Path, batch_size):
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        raise IOError (f'{vid} could not be opened')
    
    batch = []
    try: 
        while True:
            success, frame = cap.read()
            # don't break detection for entire process if one has corrupted frames
            if not success:
                break
            tup = (vid.name, frame)
            batch.append(tup)

            if len(batch) >= batch_size:
                yield batch
                batch = []

    finally:
        cap.release()

    if batch:
        yield batch

# each batch is [ (filename, frame) ]
def get_batch_imgs(imgs, batch_size):
    batch = []

    for file in imgs:
        img = cv2.imread(str(file))
        if img.size == 0:
            continue

        tup = (file.name, img)
        batch.append(tup)
    
        if len(batch) >= batch_size:
            yield batch
            batch = []
        
    if batch:
        yield batch