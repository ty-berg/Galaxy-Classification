import h5py
import numpy as np
import tensorflow as tf
import cv2
import itertools as it
import matplotlib.pyplot as plt

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
        test_labels.append(labels[i])
        test.append(downscaled_image)

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
model.add(tf.keras.layers.Conv2D(filters=8, kernel_size = (5,5), input_shape = input_shape))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Conv2D(filters=8, kernel_size = (3,3), input_shape = input_shape))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))


model.add(tf.keras.layers.Conv2D(filters=16, kernel_size = (3,3), input_shape = input_shape))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size = (3,3), input_shape = input_shape))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(0.35))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(units=10,activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

epochs = 45  # how many times the network will see the ENTIRE training set

print('-TRAINING----------------------------')
print('Input shape:', train.shape)
print('Number of training images: ', train.shape[0])
# Fit the model
model.fit(datagen.flow(train, train_labels, batch_size=64),
          epochs=epochs,
          validation_data=(validation, validation_labels))

for layer in model.layers:
    if 'conv' in layer.name:
        filters, biases = layer.get_weights()

        # normalize filter values to 0-1 so we can visualize them
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)

fig, axes = plt.subplots(4, 4)
fig.tight_layout()

for i, j in it.product(range(4),range(4)):
    axes[i, j].imshow(filters[:,:,0,(i+1)*(j+1)-1],cmap='Greys')
    axes[i, j].set_aspect('equal', 'box')

plt.setp(axes, xticks = [], yticks = [])
plt.show()



