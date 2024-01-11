import cv2
import os
import re

def get_max_dimensions(directory):
    max_width, max_height = 0, 0
    for filename in os.listdir(directory):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(directory, filename)
            image = cv2.imread(path)
            if image is not None:
                h, w, _ = image.shape
                max_width, max_height = max(max_width, w), max(max_height, h)
    return max_width, max_height

def resize_frame(frame, target_size):
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

def numerical_sort_key(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0

def create_video_from_frames(input_folder, output_file, frame_rate=30):
    max_width, max_height = get_max_dimensions(input_folder)
    video_resolution = (max_width, max_height)
    video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, video_resolution)

    filenames = sorted(filter(lambda f: f.lower().endswith((".png", ".jpg", ".jpeg")), os.listdir(input_folder)), key=numerical_sort_key)

    for filename in filenames:
        path = os.path.join(input_folder, filename)
        frame = cv2.imread(path)
        if frame is not None:
            resized_frame = resize_frame(frame, video_resolution)
            video_writer.write(resized_frame)

    video_writer.release()
