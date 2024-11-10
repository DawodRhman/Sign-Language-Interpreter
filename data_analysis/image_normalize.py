import cv2
import os
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np




dataset_path = "./mizo_dataset"
resized_dataset_path = "./Weather_Data/resize_images"
classes = os.listdir(dataset_path)