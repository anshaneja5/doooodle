# Import necessary libraries
from pathlib import Path
from quickdraw import QuickDrawData, QuickDrawDataGroup
from keras.utils import image_dataset_from_directory
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Rescaling, InputLayer
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
# Function to generate class images
image_size = (28, 28)
# def generate_class_images(name, max_drawings, recognized):
#     directory = Path("./content/dataset/" + name)
#
#     if not directory.exists():
#         directory.mkdir(parents=True)
#
#     images = QuickDrawDataGroup(name, max_drawings=max_drawings, recognized=recognized)
#     for img in images.drawings:
#         filename = directory.as_posix() + "/" + str(img.key_id) + ".png"
#         img.get_image(stroke_width=3).resize(image_size).save(filename)
#
# # Generate images for the first 9 classes in QuickDraw
# count =0
# for label in QuickDrawData().drawing_names:
#     if label == 'banana' or label == 'basketball' or label == 'ladder':
#         generate_class_images(label, max_drawings=20000, recognized=True)
#         count = count + 1
#     if count == 3:
#         break

# Define image size and initialize count
image_size = (28, 28)
# count = 0
# # #
dataset_dir = "./content/dataset"
batch_size = 32

train_ds = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=124,
    color_mode="grayscale",
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True
)

val_ds = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=124,
    color_mode="grayscale",
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True
)

# Preprocess the data
def preprocess(image, label):
    one_hot_label = tf.one_hot(label, depth=3)
    return image, one_hot_label
#
train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)

# Normalize
def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label

train_ds = train_ds.map(process)
val_ds = val_ds.map(process)
#
# import matplotlib.pyplot as plt
#
# Function to display a batch of images and their corresponding labels
# def display_batch(dataset):
#     # Get a batch of images and labels
#     for images, labels in dataset.take(1):
#         # Print the shape of images and labels
#         print(f'Images batch shape: {images.shape}')
#         print(f'Labels batch shape: {labels.shape}')
#         # print(images[0])
#         # Plot the first 5 images and their one-hot encoded labels
#         plt.figure(figsize=(10, 5))
#         for i in range(20):
#             plt.subplot(1, 5, i + 1)
#             plt.imshow(images[i].numpy().squeeze(), cmap='gray')
#             plt.title(f'Label: {tf.argmax(labels[i]).numpy()}')
#             plt.axis('off')
#         plt.show()
#
# # Display a batch from the training dataset
# display_batch(train_ds)


# Define the CNN model
input_shape = (28, 28, 1)
model = Sequential()
model.add(InputLayer(input_shape=input_shape))
model.add(Conv2D(6, kernel_size=(3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
#
model.add(Conv2D(8, kernel_size=(3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
#
model.add(Conv2D(10, kernel_size=(3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
#
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.1))
model.add(Dense(3, activation='softmax'))
#
# # Display the model summary
model.summary()
# #
# # Compile the model
optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_ds, epochs=10, validation_data=val_ds)
# #
# # # Test the model with an example image
import pickle
with open('model2.pkl', 'wb') as file:
    pickle.dump(model, file)
# from matplotlib import pyplot as plt
# model = tf.keras.models.load_model('doodles.keras')
# test_img = cv2.imread('debug_resized.png')
# test_image = cv2.resize(test_img, dsize=(28, 28))
# gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
# # gray_image = gray_image / 255.0
# # print(test_image)
# plt.imsave('gray_image.png', gray_image, cmap='gray')
# # plt.imshow(test_image, cmap='gray', vmin=0, vmax=1)
# gray_image = np.expand_dims(gray_image, axis=-1)
#
# # Add the batch dimension to make it (1, 28, 28, 1)
# final_image = np.expand_dims(gray_image, axis=0)
# print("final img is", final_image)
# predicted_class_indices = np.argmax(model.predict(final_image), axis=1)
# classes = [
#     'banana','basketball','birthday cake'
# ]
# #
# print(classes[predicted_class_indices[0]])