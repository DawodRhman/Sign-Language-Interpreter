import cv2
import os
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


# def image_shape(dataset_path,classes):
#     image_widths = []
#     image_heights = []
#     for class_name in classes:
#         class_path = os.path.join(dataset_path,class_name)
#         images = os.listdir(class_path)
#         for image in images:
#             image_path = os.path.join(class_path,image)
#             img = cv2.imread(image_path)
#             if img is not None:
#                 height, width, _ = img.shape  # Extract the height and width
#                 image_widths.append(width)
#                 image_heights.append(height)
#     width_counts = Counter(image_widths)
#     height_counts = Counter(image_heights)
    
#     #Plot the distributions of widths and heights
#     plt.figure(figsize=(12, 6))


#     #Plot width distribution
#     plt.subplot(1, 2, 1)
#     plt.bar(width_counts.keys(), width_counts.values(), color='blue', alpha=0.9)
#     plt.title('Image Width Distribution')
#     plt.xlabel('Width (pixels)')
#     plt.ylabel('Number of Images')

#     # Plot height distribution
#     plt.subplot(1, 2, 2)
#     plt.bar(height_counts.keys(), height_counts.values(),width=0.8, color='green', alpha=0.9)
#     plt.title('Image Height Distribution')
#     plt.xlabel('Height (pixels)')
#     plt.ylabel('Number of Images')

#     # Show the plots
#     plt.tight_layout()
#     plt.show()

# def image_shape_distribution(dataset_path, classes):
#     dimensions = []  # List to store (width, height) pairs

#     for class_name in classes:
#         class_path = os.path.join(dataset_path, class_name)
#         images = os.listdir(class_path)
        
#         for image in images:
#             image_path = os.path.join(class_path, image)
#             img = cv2.imread(image_path)
            
#             if img is not None:
#                 height, width, _ = img.shape
#                 dimensions.append((width, height))  # Append (width, height) tuple

#     # Count occurrences of each (width, height) pair
#     dimension_counts = Counter(dimensions)
    
#     # Extract data for plotting
#     labels = [f"{dim[0]}x{dim[1]}" for dim in dimension_counts.keys()]
#     counts = list(dimension_counts.values())
    
#     # Plot the distribution of unique image dimensions
#     plt.figure(figsize=(14, 8))
#     plt.bar(labels, counts, color='purple', alpha=0.7)
#     plt.title('Image Dimensions Distribution')
#     plt.xlabel('Image Dimensions (Width x Height)')
#     plt.ylabel('Number of Images')
#     plt.xticks(rotation=90)  # Rotate x-axis labels for readability
#     plt.tight_layout()
#     plt.show()
# def image_shape_distribution(dataset_path, classes):
#     dimensions = []  # List to store (width, height) pairs

#     for class_name in classes:
#         class_path = os.path.join(dataset_path, class_name)
#         images = os.listdir(class_path)
        
#         for image in images:
#             image_path = os.path.join(class_path, image)
#             img = cv2.imread(image_path)
            
#             if img is not None:
#                 height, width, _ = img.shape
#                 dimensions.append((width, height))  # Append (width, height) tuple

#     # Count occurrences of each (width, height) pair
#     dimension_counts = Counter(dimensions)
    
#     # Extract data for plotting
#     labels = [f"{dim[0]}x{dim[1]}" for dim in dimension_counts.keys()]
#     counts = list(dimension_counts.values())
    
#     # Calculate the y-axis limit as 1.5 times the maximum count
#     max_count = max(counts)
#     y_limit = max_count * 1.5
    
#     # Plot the distribution of unique image dimensions
#     plt.figure(figsize=(14, 8))
#     plt.bar(labels, counts, color='purple', alpha=0.7)
#     plt.title('Image Dimensions Distribution')
#     plt.xlabel('Image Dimensions (Width x Height)')
#     plt.ylabel('Number of Images')
#     plt.xticks(rotation=90)  # Rotate x-axis labels for readability
#     plt.ylim(0, y_limit)  # Set the y-axis range to 1.5 times the max count
#     plt.tight_layout()
#     plt.show()
def image_shape_distribution(dataset_path, classes):
    dimensions = []  # List to store (width, height) pairs

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        images = os.listdir(class_path)
        
        for image in images:
            image_path = os.path.join(class_path, image)
            img = cv2.imread(image_path)
            
            if img is not None:
                height, width, _ = img.shape
                dimensions.append((width, height))  # Append (width, height) tuple

    # Count occurrences of each (width, height) pair
    dimension_counts = Counter(dimensions)
    
    # Extract data for plotting
    labels = [f"{dim[0]}x{dim[1]}" for dim in dimension_counts.keys()]
    counts = list(dimension_counts.values())
    
    # Calculate the y-axis limit as 1.5 times the maximum count
    max_count = max(counts)
    y_limit = max_count * 1.5
    
    # Calculate y-axis interval as y_limit divided by 30
    y_interval = y_limit / 30

    # Plot the distribution of unique image dimensions
    plt.figure(figsize=(14, 8))
    plt.bar(labels, counts, color='purple', alpha=0.7)
    plt.title('Image Dimensions Distribution')
    plt.xlabel('Image Dimensions (Width x Height)')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=90)  # Rotate x-axis labels for readability
    
    # Set the y-axis limit and y-ticks interval
    plt.ylim(0, y_limit)
    plt.yticks(range(0, int(y_limit) + int(y_interval), int(y_interval)))  # Set y-ticks with calculated interval

    plt.tight_layout()
    plt.show()



dataset_path = "../mizo_dataset"
resized_dataset_path = "../resize_test"
classes = os.listdir(dataset_path)


image_shape_distribution(dataset_path,classes)
