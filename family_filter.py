import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import subprocess
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
INPUT_VIDEO = "video.mp4"
OUTPUT_VIDEO = "family_friendly.mp4"
FRAME_INTERVAL = 1
BATCH_SIZE = 32
THRESHOLD = 0.85

def load_nsfw_model():
    model_path = os.path.join(os.path.dirname(__file__), "nsfw.299x299.h5")
    try:
        model = load_model(model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def extract_frames_with_timestamps(video_path, output_dir):
    """Extract frames while preserving exact timestamps"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_paths = []
    frame_timestamps = []
    
    print("Extracting frames with timestamps...")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % FRAME_INTERVAL == 0:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            frame_timestamps.append(timestamp)
            
        frame_count += 1
    
    cap.release()
    return frame_paths, fps, frame_size, frame_timestamps

def process_image_batch(model, batch_paths):
    batch_images = []
    for path in batch_paths:
        img = image.load_img(path, target_size=(299, 299))
        x = image.img_to_array(img)
        batch_images.append(x)
    
    batch_array = np.array(batch_images)
    batch_array = preprocess_input(batch_array)
    preds = model.predict(batch_array, verbose=0)
    
    safe_paths = []
    for i, pred in enumerate(preds):
        if (pred[2] + pred[0] + pred[4]) > THRESHOLD:
            safe_paths.append(batch_paths[i])
    return safe_paths

def classify_frames(model, frame_paths):
    print("Classifying frames...")
    safe_frames = []
    
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = []
        for i in range(0, len(frame_paths), BATCH_SIZE):
            batch_paths = frame_paths[i:i + BATCH_SIZE]
            futures.append(executor.submit(process_image_batch, model, batch_paths))
        
        for future in tqdm(futures, desc="Processing batches"):
            safe_frames.extend(future.result())
    
    return safe_frames

def create_video_with_accurate_audio(frames, timestamps, original_video, output_path, fps, frame_size):
    """Create video with perfect audio synchronization"""
    print("Creating video with perfect audio sync...")
    
    # 1. Create temporary video without audio
    temp_video = "temp_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, fps, frame_size)
    
    for frame_path in tqdm(frames, desc="Writing frames"):
        frame = cv2.imread(frame_path)
        if frame is not None:
            out.write(frame)
    out.release()
    
    # 2. Create audio filter file
    filter_file = "audio_filter.txt"
    with open(filter_file, 'w') as f:
        for i, ts in enumerate(timestamps):
            if i < len(timestamps)-1:
                duration = timestamps[i+1] - ts
            else:
                duration = 1.0/fps  # Last frame duration
            f.write(f"between(t,{ts},{ts+duration})*enable='eq(n,{i})';\n")
    
    # 3. Merge audio with perfect sync
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', original_video,
            '-i', temp_video,
            '-filter_complex_script', filter_file,
            '-map', '[0:a]',
            '-map', '[1:v]',
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Audio synced perfectly!")
    except Exception as e:
        print(f"Audio sync failed: {str(e)}. Using simple method.")
        # Fallback to simple audio copy
        cmd = [
            'ffmpeg', '-y',
            '-i', temp_video,
            '-i', original_video,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            output_path
        ]
        subprocess.run(cmd, check=True)
    
    # Cleanup
    if os.path.exists(temp_video):
        os.remove(temp_video)
    if os.path.exists(filter_file):
        os.remove(filter_file)

def main():
    temp_dir = "temp_frames"
    
    try:
        # 1. Load model
        model = load_nsfw_model()
        
        # 2. Extract frames with timestamps
        frame_paths, fps, frame_size, timestamps = extract_frames_with_timestamps(INPUT_VIDEO, temp_dir)
        
        # 3. Classify frames
        safe_frames = classify_frames(model, frame_paths)
        
        if not safe_frames:
            raise ValueError("No safe frames found!")
        
        # Get timestamps for safe frames
        safe_timestamps = [timestamps[frame_paths.index(fp)] for fp in safe_frames]
        
        # 4. Create final video with perfect audio sync
        create_video_with_accurate_audio(safe_frames, safe_timestamps, INPUT_VIDEO, OUTPUT_VIDEO, fps, frame_size)
        
        # 5. Print results
        print(f"\nProcessing complete!")
        print(f"Original frames: {len(frame_paths)} | Safe frames: {len(safe_frames)}")
        print(f"Reduction: {100*(1-len(safe_frames)/len(frame_paths)):.1f}% content removed")
    
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()