import cv2
import os
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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
    
# Plot Bar Graph of image counts for each class
def plot_counts(class_counts):
    sortedClasses = sorted(class_counts.keys(), key=lambda x: class_counts[x], reverse=True)
    counts = [class_counts[class_name] for class_name in sortedClasses]

    plt.figure(figsize=(10, 6))
    plt.barh(sortedClasses, counts, color='blue')  
    plt.ylabel('Class')
    plt.xlabel('Number of Images')
    plt.title('Number of Images Per Class in Train Dataset')
    plt.tight_layout()  
    plt.show()

# Plot Pie Chart of image counts for each class
def plot_pie_chart(class_counts):
    sorted_classes = sorted(class_counts.keys(), key=lambda x: class_counts[x])
    sizes = [class_counts[class_name] for class_name in sorted_classes]

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=sorted_classes, autopct='%1.1f%%', startangle=140)
    plt.title('Percentage of Distribution of Image Classes in Train Dataset')
    plt.show()

# Averaging Image per Class    
def average_images_per_class(x_train, y_train, class_names):
    # Create a grid for subplots
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    # Iterate over the dataset and get the average
    for i, (ax, class_name) in enumerate(zip(axes.flatten(), class_names)):
        avg_image = np.mean(x_train[np.squeeze(y_train==i)], axis=0) / 255
        ax.imshow(avg_image)
        ax.set_title(f"Average image for {class_name}")
        ax.axis('off')  # Turn off axis

# Averaging Image for Dataset
def average_image(train_data):
    average_image = np.mean(train_data, axis=0) / 255

    plt.figure(figsize=(9, 9))
    plt.imshow(average_image)
    plt.axis('off')  # Turn off the axis
    plt.title("Average Image of the Dataset")
    plt.show()


