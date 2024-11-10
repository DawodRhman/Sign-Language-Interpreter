import cv2
import os
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from scipy.stats import zscore



def dimension_average(dataset_path, classes):
    image_widths = []
    image_heights = []
    pixel_intensities = []
    class_z_scores = {}  # Store Z-scores for each class

    # Calculate means and standard deviations for each class's images
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        images = os.listdir(class_path)
        
        class_widths = []
        class_heights = []
        class_intensities = []
        
        for image in images:
            image_path = os.path.join(class_path, image)
            img = cv2.imread(image_path)
            
            if img is not None:
                height, width, channels = img.shape
                class_widths.append(width)
                class_heights.append(height)
                
                # Convert to grayscale and calculate intensity
                gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                class_intensities.append(np.mean(gray_img))
        
        # Calculate Z-scores for width, height, and intensity
        widths_z = zscore(class_widths)
        heights_z = zscore(class_heights)
        intensities_z = zscore(class_intensities)
        
        # Store Z-scores for later plotting
        class_z_scores[class_name] = {
            'width_z': widths_z,
            'height_z': heights_z,
            'intensity_z': intensities_z
        }
        
        # Append to the global lists for all classes (for final plotting)
        image_widths.extend(class_widths)
        image_heights.extend(class_heights)
        pixel_intensities.extend(class_intensities)

    # Plot Box and Whisker plots for each class based on Z-scores
    plt.figure(figsize=(16, 8))

    # Plot for Width Z-scores
    plt.subplot(1, 3, 1)
    plt.boxplot([class_z_scores[class_name]['width_z'] for class_name in classes], labels=classes)
    plt.title('Width Z-scores by Class')
    plt.ylabel('Z-score')

    # Plot for Height Z-scores
    plt.subplot(1, 3, 2)
    plt.boxplot([class_z_scores[class_name]['height_z'] for class_name in classes], labels=classes)
    plt.title('Height Z-scores by Class')
    plt.ylabel('Z-score')

    # Plot for Intensity Z-scores
    plt.subplot(1, 3, 3)
    plt.boxplot([class_z_scores[class_name]['intensity_z'] for class_name in classes], labels=classes)
    plt.title('Intensity Z-scores by Class')
    plt.ylabel('Z-score')

    plt.tight_layout()
    plt.show()

    # Calculate and display the mean for width, height, and intensity
    width_mean = np.mean(image_widths)
    height_mean = np.mean(image_heights)
    pixel_intensity_mean = np.mean(pixel_intensities)
    print(f"Width mean: {width_mean}\nHeight mean: {height_mean}\nIntensity mean: {pixel_intensity_mean}")

    # Plot the intensity distribution
    plt.hist(pixel_intensities, bins=20, color='gray')
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.title("Intensity Distribution")
    plt.show()




dataset_path = "../mizo_dataset"
resized_dataset_path = "./Weather_Data/resize_images"
classes = os.listdir(dataset_path)

dimension_average(dataset_path,classes)