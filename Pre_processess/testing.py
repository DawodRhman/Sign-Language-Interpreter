import os
import cv2
import shutil

def resize_and_save_images(dataset_path, classes, output_path, target_size=(500, 500)):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for class_name in classes:
        # Create the class directory in the output folder
        class_output_path = os.path.join(output_path, class_name)
        if not os.path.exists(class_output_path):
            os.makedirs(class_output_path)
        
        class_path = os.path.join(dataset_path, class_name)
        images = os.listdir(class_path)
        
        for image in images:
            image_path = os.path.join(class_path, image)
            img = cv2.imread(image_path)
            
            if img is not None:
                # Resize the image
                resized_img = cv2.resize(img, target_size)
                
                # Save the resized image in the corresponding class folder
                output_image_path = os.path.join(class_output_path, image)
                cv2.imwrite(output_image_path, resized_img)
                
                # Optionally, you can print the progress for large datasets
                print(f"Processed: {image_path} -> {output_image_path}")
            else:
                # Handle corrupted images
                print(f"Corrupted image: {image_path}")
                delete = input("Do you want to delete this image? (y/n): ").strip().lower()
                if delete == 'y':
                    os.remove(image_path)
                    print(f"Deleted: {image_path}")
    
    print("Image resizing and saving completed.")
    
dataset_path = "../Weather_Data"
resized_dataset_path = "../resize_test"
classes = os.listdir(dataset_path)

resize_and_save_images(dataset_path,classes,resized_dataset_path)

