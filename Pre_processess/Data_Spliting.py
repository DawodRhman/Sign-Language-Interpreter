# import splitfolders  # To install: pip install split-folders

# # Path to your dataset folder
# input_folder = 'path/to/your/dataset'  # e.g., "data/images"

# # Output folder where split datasets will be stored
# output_folder = 'path/to/output/folder'  # e.g., "data/output"

# # Split with ratios for training, validation, and test
# splitfolders.ratio(
#     input_folder, 
#     output=output_folder, 
#     seed=42, 
#     ratio=(0.7, 0.2, 0.1),  # 70% training, 20% validation, 10% testing
#     group_prefix=None,  # Keeps class structure for each split
#     move=False          # Set to True if you want to move files instead of copying
# )

# print("Dataset split completed!")

import os
import shutil
import random
import matplotlib.pyplot as plt

def split_dataset(dataset_path, output_path, classes, split_ratio=(0.7, 0.2, 0.1)):
    # Create directories for train, validate, test in the output path
    train_dir = os.path.join(output_path, 'train')
    validate_dir = os.path.join(output_path, 'validate')
    test_dir = os.path.join(output_path, 'test')

    # Create the directories if they do not exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validate_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        
        # Create class-specific directories in train, validate, and test
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(validate_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # List all images in the class directory
        images = os.listdir(class_path)
        random.shuffle(images)  # Shuffle images to randomize the split
        
        # Calculate the split sizes
        total_images = len(images)
        train_size = int(split_ratio[0] * total_images)
        validate_size = int(split_ratio[1] * total_images)
        
        # Split the images
        train_images = images[:train_size]
        validate_images = images[train_size:train_size + validate_size]
        test_images = images[train_size + validate_size:]
        
        # Copy the images into their respective directories
        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
        for img in validate_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(validate_dir, class_name, img))
        for img in test_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))
    
    print(f"Dataset split completed and saved to '{output_path}' folder.")

dataset_path = "../mizo_dataset"
output_path = "../split_data"  # Specify output path for split data
classes = os.listdir(dataset_path)

# Split the dataset into train, validate, and test sets in a separate folder
split_dataset(dataset_path, output_path, classes)

