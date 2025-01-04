
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

def load_images_and_masks(img_path, mask_path, img_size=(64, 64)):
    images = []
    masks = []
    
    for img_file in sorted(os.listdir(img_path)):
        img = cv2.imread(os.path.join(img_path, img_file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size)
        images.append(img)
    
    for mask_file in sorted(os.listdir(mask_path)):
        mask = cv2.imread(os.path.join(mask_path, mask_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size)
        masks.append(mask)
    
    images = np.array(images).reshape(-1, img_size[0], img_size[1], 1) / 255.0
    masks = np.array(masks).reshape(-1, img_size[0], img_size[1], 1) / 255.0
    return images, masks

def unet_model(input_size=(64, 64, 1)):
    inputs = Input(input_size)

    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4)

    u5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Paths to image and mask directories
img_path = 'D:\\saltanalysis\\images'
mask_path = 'D:\\saltanalysis\\masks'

# Load and preprocess data
img_size = (64, 64)  # Reduced size for faster training
images, masks = load_images_and_masks(img_path, mask_path, img_size=img_size)

# Split into training and validation sets
train_images, val_images, train_masks, val_masks = train_test_split(images, masks, test_size=0.1, random_state=42)

# Define and compile the model
model = unet_model(input_size=(img_size[0], img_size[1], 1))
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
model_checkpoint = ModelCheckpoint('salt_segmentation_unet.keras', save_best_only=True, monitor='val_loss', mode='min')

# Train the model
history = model.fit(
    train_images, train_masks,
    validation_data=(val_images, val_masks),
    epochs=30,  # Reduced epochs for faster training
    batch_size=16,  # Adjust batch size as needed
    callbacks=[early_stopping, model_checkpoint]
)

# Predict on a few test images
predictions = model.predict(images[:5])

# Visualize predictions
for i in range(5):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(images[i].reshape(img_size[0], img_size[1]), cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('Ground Truth Mask')
    plt.imshow(masks[i].reshape(img_size[0], img_size[1]), cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Predicted Mask')
    plt.imshow(predictions[i].reshape(img_size[0], img_size[1]), cmap='gray')

    plt.show()