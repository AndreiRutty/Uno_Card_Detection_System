import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

# Define paths to your augmented dataset and validation dataset
train_data_dir = r'C:\Users\andre\OneDrive\Desktop\Uno_Cards_Dataset_v2\augmented_train'  # Augmented images directory
validation_data_dir = r'C:\Users\andre\OneDrive\Desktop\Uno_Cards_Dataset_v2\samples'

# Image data generator with rescaling for training
train_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Image data generator for validation (only rescaling)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Create the training generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(512, 512),  # Resize all images to 150x150 pixels
    batch_size=128,  # Batch size of 32 images per iteration
    class_mode='categorical'  # Assuming multi-class classification
)

# Create the validation generator
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(512, 512),
    batch_size=128,
    class_mode='categorical'
)

model = Sequential()

# Block 1
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(512, 512, 3)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Block 2
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Block 3
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Block 4
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Block 5
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Fully connected layers
model.add(Flatten())  # Flatten the output
model.add(Dense(4096, activation='relu'))  # Input shape is inferred from Flatten layer
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(54, activation='softmax'))  # Assuming 54 classes

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary before training
model.summary()
# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=50,
    callbacks=[early_stopping]
)

# Save the model after training
model.save('uno_card_classifier.h5')


# Define early stopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
#
# # Train the model using augmented images
# model.fit(
#     train_generator,
#     validation_data=validation_generator,
#     epochs=50,  # Increased epochs for more training
#     callbacks=[early_stopping]
# )
#
# # Save the model
# model.save('uno_model.h5')
# model.save('uno_model.keras')  # Save model in Keras format
# model.summary()
