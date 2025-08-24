import torch
import clip
import os
import cv2
import glob
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import multiprocessing

# Ensure the 'spawn' start method is used for multiprocessing
multiprocessing.set_start_method('spawn', force=True)

def get_image_features(image, model, preprocess, device):
    image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
    return features

def compare_frames(frame1_features, frame2_features, threshold=0.9):
    similarity = torch.cosine_similarity(frame1_features, frame2_features)
    return similarity.item() < threshold

# Define the model initialization function
def init_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

# Define the function to process the video
def process_video_with_model(video_path, keyframes_root_folder, threshold=0.9):
    model, preprocess, device = init_model()

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    keyframe_folder = os.path.join(keyframes_root_folder, video_name)

    if not os.path.exists(keyframe_folder):
        os.makedirs(keyframe_folder)

    cap = cv2.VideoCapture(video_path)

    prev_features = None
    frame_number = 0
    keyframe_number = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Log progress every 100 frames
        if frame_number % 100 == 0:
            print(f"Processing: {video_name} -- Current frame: {frame_number}/{total_frames}")

        curr_features = get_image_features(frame, model, preprocess, device)

        if prev_features is not None:
            if compare_frames(prev_features, curr_features, threshold):
                key_frame_path = os.path.join(keyframe_folder, f"{video_name}_{frame_number:05d}.jpg")
                cv2.imwrite(key_frame_path, frame)
                keyframe_number += 1
        else:
            key_frame_path = os.path.join(keyframe_folder, f"{video_name}_{frame_number:05d}.jpg")
            cv2.imwrite(key_frame_path, frame)
            keyframe_number += 1

        prev_features = curr_features

        frame_number += 1

    cap.release()

# Function to process videos in parallel using ProcessPoolExecutor
def process_videos_in_parallel(video_paths, keyframes_root_folder, num_workers=4, threshold=0.9):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_video_with_model, video_path, keyframes_root_folder, threshold)
            for video_path in video_paths
        ]
        for future in futures:
            future.result()  # Ensure each future completes

# Script to find videos and process them
if __name__ == "__main__":
    # Define the pattern to match the desired file paths
    pattern = './Videos/Videos_L07/video/L*_V*.mp4'

    # Use glob to find all files matching the pattern
    matching_files = glob.glob(pattern)

    # Print the matching files
    for file_path in matching_files:
        print(f"Found video file: {file_path}")

    # Define the folder to save keyframes
    keyframes_root_folder = 'pank_keyframes'

    # Process videos in parallel
    process_videos_in_parallel(matching_files, keyframes_root_folder, num_workers=2, threshold=0.88)
