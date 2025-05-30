from . import postprocessor
from . import preprocessor

import time
import statistics
import pandas as pd

def process_batches(model, batch_gen, output_path, save_df):
    rows_for_df = []
    times = []

    for i, batch in enumerate(batch_gen):
        print(f'Processing batch {i}')
        start_time = time.perf_counter()

        results = model.predict(batch)
        ## saving individual frames in postprocessor, returning list of results for df
        batch_rows_for_df = postprocessor.handle_results(results, output_path, i * model.config.batch_size, model.config)
        rows_for_df.extend(batch_rows_for_df)

        end_time = time.perf_counter()
        times.append(end_time - start_time)

        # testing break
    df = pd.DataFrame(rows_for_df)
    if save_df:
        df.to_csv(output_path / 'frames.csv', index = False)

    return rows_for_df, times

def process_vid(model, vid):
    output_path = model.config.results_path / vid.name
    output_path.mkdir(exist_ok = True)
    batch_gen = preprocessor.get_batch_vid(vid, batch_size=model.config.batch_size)
    save_frames_csv = model.config.save_frames_csv

    vid_rows_for_df, times = process_batches(model, batch_gen, output_path, save_frames_csv)

    if times:
        avg_time_per_frame = statistics.fmean(times) * 1000 / model.config.batch_size
        print(f'Avg process time per frame: {avg_time_per_frame:.4f}ms\n-------------------------------')

    return vid_rows_for_df

def process_videos(model, videos):
    all_vid_rows = []

    for vid in videos:
        print(f'Processing Video: {vid.name}')
        vid_rows_for_df = process_vid(model, vid)
        all_vid_rows.extend(vid_rows_for_df)
        
    df = pd.DataFrame(all_vid_rows)

    if all_vid_rows and model.config.save_all_csv:
        df.to_csv(model.config.results_path / 'all_vid_data.csv', index=False)
    
    return df
    

def process_imgs(model, images):
    batch_gen = preprocessor.get_batch_imgs(images, model.config.batch_size)
    output_path = model.config.results_path

    print(f'Processing Images at {model.config.data_path}')
    rows_for_df, times = process_batches(model, batch_gen, output_path, model.config.save_all_csv)

    if times:
        avg_time_per_frame = statistics.fmean(times) * 1000 / model.config.batch_size
        print(f'Avg time per frame: {avg_time_per_frame:.4f}ms')

    df = pd.DataFrame(rows_for_df)
    return df

def get_all_data(images, videos):
    all_data = []
    if images:
        all_data.extend(images)
    if videos:
        all_data.extend(videos)
    
    return all_data