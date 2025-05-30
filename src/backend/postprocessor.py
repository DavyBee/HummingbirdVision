import torchvision.ops as ops

def is_frame_empty(frame):
    return frame.boxes.cls.numel() == 0

def apply_nms(frame, iou_threshold):
    # checking if no detections
    boxes_data = frame.boxes.data
    if boxes_data.size(0) == 0:
        return frame

    # get data
    boxes_xyxy = boxes_data[:, :4]  # [x1, y1, x2, y2]
    scores = boxes_data[:, 4]       # confidence scores

    # filtering data
    keep = ops.nms(boxes_xyxy, scores, iou_threshold)
    reduced_boxes = boxes_data[keep]
    frame.update(boxes=reduced_boxes)

    return frame

def add_row(name, frame, index):
    conf_list = [round(val, 4) for val in frame.boxes.conf.tolist()]
    info = {
        'file_name': str(name),
        'frame_num': index,
        'num_objs': len(conf_list),
        'confs': conf_list[0] if len(conf_list) == 1 else conf_list,
        'timestamp': 'ph'
        }
    return info

"""
Saves the frames with detections to the config.results_path directory,
returns the rows to be added to the video dataframe with detections from the batch
"""
def handle_results(batch_results, output_path, start_index, config, flag=False):
    frame_path = output_path / 'saved_frames'
    frame_path.mkdir(exist_ok = True)
    
    batch_rows_for_df = []
    
    for index, (name, frame) in enumerate(batch_results, start = start_index):
        if not is_frame_empty(frame):
            frame = apply_nms(frame, config.iou)
            frame.save(frame_path / f'{index:05d}.jpg')
            row_data = add_row(name, frame, index)
            batch_rows_for_df.append(row_data)
    
    return batch_rows_for_df