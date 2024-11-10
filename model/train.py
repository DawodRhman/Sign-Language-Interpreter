import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Paths to the dataset
train_dir = "../split_data/train"
validate_dir = "../split_data/validate"
test_dir = "../split_data/test"

# Image dimensions and batch size
img_height, img_width = 500, 500
batch_size = 16

# Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

validate_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Loading the data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
)

validate_generator = validate_datagen.flow_from_directory(
    validate_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
)

# Load the ResNet50 model with pretrained weights
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))

# Unfreeze more layers of ResNet50 to allow the model to learn better
for layer in base_model.layers[:-10]:  # Unfreeze more layers
    layer.trainable = True

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)  # Add dropout to avoid overfitting
predictions = Dense(train_generator.num_classes, activation="softmax")(x)

# Define the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Print the model summary
print("Model Summary:")
model.summary()

# Compile the model with a slightly higher learning rate
model.compile(optimizer=Adam(learning_rate=0.0005), loss="categorical_crossentropy", metrics=["accuracy"])

# Early stopping callback with lower patience
early_stopping = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)

# Train the model for more epochs to give it time to learn
epochs = 50
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validate_generator,
    validation_steps=validate_generator.samples // batch_size,
    epochs=epochs,
    callbacks=[early_stopping],
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Get the true labels and predicted labels for precision, recall, and F1-score
test_generator.reset()
predictions = model.predict(test_generator, steps=test_generator.samples // batch_size + 1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Calculate Precision, Recall, and F1-Score
precision = precision_score(true_classes, predicted_classes, average='weighted', zero_division=1) * 100
recall = recall_score(true_classes, predicted_classes, average='weighted', zero_division=1) * 100
f1 = f1_score(true_classes, predicted_classes, average='weighted', zero_division=1) * 100

print(f"Test Precision: {precision:.2f}%")
print(f"Test Recall: {recall:.2f}%")
print(f"Test F1-Score: {f1:.2f}%")

# Plot training and validation accuracy and loss
plt.figure(figsize=(12, 6))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and Validation Accuracy")

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")

plt.tight_layout()
plt.show()

# Check for overfitting or underfitting based on accuracy and loss curves
train_accuracy = history.history["accuracy"][-1]
val_accuracy = history.history["val_accuracy"][-1]
train_loss = history.history["loss"][-1]
val_loss = history.history["val_loss"][-1]

print(f"Final Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Final Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Final Training Loss: {train_loss:.4f}")
print(f"Final Validation Loss: {val_loss:.4f}")

if train_accuracy > val_accuracy and train_loss < val_loss:
    print("Warning: Possible Overfitting!")
elif train_accuracy < 0.7 and val_accuracy < 0.7:
    print("Possible Underfitting: Consider increasing model complexity or training for more epochs.")
else:
    print("Model seems to be performing well.")
    
# Ask if the user wants to save the model
save_model = input("Do you want to save the trained model? (yes/no): ").strip().lower()

if save_model == "yes":
    # Save the model to a file
    model.save('sign_language_model.h5')
    print("Model saved as 'sign_language_model.h5'.")
else:
    print("Model not saved.")
