import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define paths to your dataset
train_data_dir = r'C:\Users\andre\OneDrive\Desktop\Uno_Cards_Dataset_v2\train'
augmented_data_dir = r'C:\Users\andre\OneDrive\Desktop\Uno_Cards_Dataset_v2\augmented_train'

# Check if the augmented directory exists; if not, create it
if not os.path.exists(augmented_data_dir):
    os.makedirs(augmented_data_dir)

# Image data generator with maximum augmentation
augmentation_generator = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=90,                # Rotate up to 90 degrees
    width_shift_range=0.4,            # Up to 40% horizontal shift
    height_shift_range=0.4,           # Up to 40% vertical shift
    shear_range=0.4,                  # Shear transformation
    zoom_range=0.5,                   # Up to 50% zoom in or out
    horizontal_flip=True,             # Flip horizontally
    vertical_flip=True,               # Flip vertically for more variety
    brightness_range=[0.5, 1.5],      # Adjust brightness
    channel_shift_range=150.0,        # Random channel shift for color variation
    fill_mode='nearest'
)

# Define the number of augmented images to generate per original image
num_augmented_images = 20

# Loop through each class directory in the train_data_dir
for class_name in os.listdir(train_data_dir):
    class_dir = os.path.join(train_data_dir, class_name)
    save_dir = os.path.join(augmented_data_dir, class_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load images for the current class
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        image = tf.keras.preprocessing.image.load_img(img_path)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.expand_dims(image, 0)

        # Generate augmented images
        aug_iter = augmentation_generator.flow(
            image, batch_size=1, save_to_dir=save_dir, save_prefix='aug', save_format='jpeg'
        )

        # Generate and save a fixed number of augmented images for each input image
        for i in range(num_augmented_images):
            next(aug_iter)

print("Maximum augmentation complete. Augmented images are saved to", augmented_data_dir)