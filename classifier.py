import h5py
import numpy as np
import tensorflow as tf
import cv2
import itertools as it
import matplotlib.pyplot as plt
import random

def display_images(images, true_labels, predicted_labels=None, num_images=5, classes=None):
    """
    Display random images with their classifications.

    Parameters:
    - images: array of images.
    - true_labels: array of true labels.
    - predicted_labels: (optional) array of predicted labels.
    - num_images: number of images to display.
    - classes: (optional) list of class names corresponding to label indices.
    """

    plt.figure(figsize=(15, 6))  # Adjust size as needed

    # Randomly select images
    random_indices = random.sample(range(images.shape[0]), num_images)

    for i, idx in enumerate(random_indices):
        ax = plt.subplot(2,int(num_images/2), i + 1)
        plt.imshow(images[idx])
        plt.axis('off')

        true_label = true_labels[idx]
        if classes:
            true_label = classes[true_label]

        title = f'True: {true_label}'

        if predicted_labels is not None:
            predicted_label = predicted_labels[idx]
            if classes:
                predicted_label = classes[predicted_label]
            title += f'\nPred: {predicted_label}'
        ax.set_title(title, fontsize=5)

    plt.subplots_adjust(wspace=3, hspace=0)  # Adjust spacing
    plt.show()




train_sizes = [649,1112,1587,1216,200,1226,1097,1577,854,1124]
val_sizes = [216,371,529,405,67,409,366,526,285,375]
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

# To get the images and labels from file
with h5py.File('Galaxy10_DECals.h5', 'r') as F:
    images = np.array(F['images'])
    labels = np.array(F['ans'])

#print(labels[6500])
#cv2.imshow("Galaxy",images[6500])
#cv2.waitKey(0)
train_labels = []
validation_labels = []
test_labels = []
train = []
validation = []
test = []
for i in range(len(labels)):
    #grey_image = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY) 
    downscaled_image = cv2.resize(images[i], (images[i].shape[1] // 2, images[i].shape[0] // 2))
    if train_labels.count(labels[i]) < train_sizes[int(labels[i])]:
        train_labels.append(labels[i])
        train.append(downscaled_image)
    elif validation_labels.count(labels[i]) < val_sizes[int(labels[i])]:
        validation_labels.append(labels[i])
        validation.append(downscaled_image)
    else:
        grey_image = cv2.cvtColor(downscaled_image, cv2.COLOR_RGB2GRAY)
        expanded_image = np.stack((grey_image,)*3, axis=-1)
        test_labels.append(labels[i])
        test.append(expanded_image)

train = np.array(train)
train_labels = np.array(train_labels)
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
validation = np.array(validation)
validation_labels = np.array(validation_labels)
validation_labels = tf.keras.utils.to_categorical(validation_labels, 10)
test = np.array(test)
test_labels = np.array(test_labels)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)
# To convert to desirable type
train_labels = train_labels.astype(np.float32)
train = train.astype(np.float32)
validation_labels = validation_labels.astype(np.float32)
validation = validation.astype(np.float32)
test_labels = test_labels.astype(np.float32)
test = test.astype(np.float32)

input_shape = (128,128,3)
train /= 255
validation /= 255
test /= 255
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=8, kernel_size = (7,7), input_shape = input_shape))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Conv2D(filters=8, kernel_size = (5,5), input_shape = input_shape))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Conv2D(filters=8, kernel_size = (5,5), input_shape = input_shape))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Conv2D(filters=16, kernel_size = (5,5), input_shape = input_shape))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(filters=16, kernel_size = (3,3), input_shape = input_shape))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(filters=16, kernel_size = (3,3), input_shape = input_shape))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size = (3,3), input_shape = input_shape))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size = (3,3), input_shape = input_shape))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.35))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(0.45))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(units=10,activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

epochs = 300  # how many times the network will see the ENTIRE training set

print('-TRAINING----------------------------')
print('Input shape:', train.shape)
print('Number of training images: ', train.shape[0])
# Fit the model
model.fit(datagen.flow(train, train_labels, batch_size=32),
          epochs=epochs,
          validation_data=(validation, validation_labels))

test_loss, test_accuracy = model.evaluate(test, test_labels, batch_size=32)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Predict labels for test images
predicted_labels_test = np.argmax(model.predict(test), axis=1)
true_labels_test = np.argmax(test_labels, axis=1)

# Display test images
display_images(test, true_labels_test, predicted_labels_test, num_images=10, classes=['Disturbed Galaxies', 'Merging Galaxies', 'Round Smooth Galaxies', 'In-between Round Smooth Galaxies', 'Cigar Shaped Smooth Galaxies', 'Barred Spiral Galaxies', 'Unbarred Tight Spiral Galaxies','Unbarred Loose Spiral Galaxies','Edge-on Galaxies without Bulge','Edge-on Galaxies with Bulge'])

# Predict labels for test images
predicted_labels_train = np.argmax(model.predict(train), axis=1)
true_labels_train = np.argmax(train_labels, axis=1)

# Display test images
display_images(train, true_labels_train, predicted_labels_train, num_images=10, classes=['Disturbed Galaxies', 'Merging Galaxies', 'Round Smooth Galaxies', 'In-between Round Smooth Galaxies', 'Cigar Shaped Smooth Galaxies', 'Barred Spiral Galaxies', 'Unbarred Tight Spiral Galaxies','Unbarred Loose Spiral Galaxies','Edge-on Galaxies without Bulge','Edge-on Galaxies with Bulge'])






