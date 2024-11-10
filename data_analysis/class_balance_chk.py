import cv2
import os
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# def class_distribution(dataset_path,classes):
#     class_count = {}
#     for class_name in classes:
#         class_path = os.path.join(dataset_path,class_name) #Save each class directory
#         class_count[class_name] = len(os.listdir(class_path)) #Save count of images of each class
#     plt.bar(class_count.keys(),class_count.values())
#     plt.xlabel("Classes")
#     plt.ylabel("Number of Images")
#     plt.title("Class Distribution")
#     plt.show()
# def class_distribution(dataset_path, classes):
#     class_count = {}
#     for class_name in classes:
#         class_path = os.path.join(dataset_path, class_name)
#         class_count[class_name] = len(os.listdir(class_path))
#     plt.bar(class_count.keys(), class_count.values())
#     plt.xlabel("Classes")
#     plt.ylabel("Number of Images")
#     plt.ylim(0, 1000)  # Set y-axis range from 0 to 1000
#     plt.title("Class Distribution")
#     plt.show()

# /////////////////////////Final///////////////////////////////
def class_distribution(dataset_path, classes):
    class_count = {}
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        class_count[class_name] = len(os.listdir(class_path))
    
    plt.bar(class_count.keys(), class_count.values())
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")
    plt.ylim(0, 1000)  # Set y-axis range from 0 to 1000
    plt.yticks(range(0, 1001, 50))  # Set y-axis intervals to 50
    plt.title("Class Distribution")
    plt.show()

dataset_path = "../mizo_dataset"
classes = os.listdir(dataset_path)

dataset_path2 = "../split_data/train"
classes2 = os.listdir(dataset_path)

class_distribution(dataset_path,classes)
