# Image Classification in Python
import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf
from keras import layers, models
import tensorflow_datasets as tfds

# Load the dataset
# Fetch the CIFAR-10 dataset
(training_data, testing_data) = tfds.load(
    'cifar10',
    split=['train', 'test'],
    as_supervised=True
)

# Normalize the data
def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

training_data = training_data.map(normalize).cache().shuffle(5000).batch(32).prefetch(tf.data.AUTOTUNE)
testing_data = testing_data.map(normalize).batch(32).prefetch(tf.data.AUTOTUNE)

# Define the class names
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Display 16 Images from the training dataset
# plt.figure(figsize=(10, 10))
# for i, (image, label) in enumerate(training_data.take(16)):
#     plt.subplot(4, 4, i+1)
#     plt.imshow(image.numpy())
#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel(class_names[label.numpy()])

# plt.show()

# Building and Training the Model
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(training_data, epochs=10, validation_data=testing_data)

# Evaluate the Model
loss, accuracy = model.evaluate(testing_data, batch_size=32)
print(f"loss: {loss}")
print(f"accuracy: {accuracy}")

# Save and Load the Model
model.save('image_classifier.model')
model = models.load_model('image_classifier.model')

# Load and Prepare the Image
img = cv.imread('car_image.avif')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (32, 32))
img = np.array([img]) / 255.0

# Display the Image
plt.imshow(img[0])
plt.show()

# Make the Prediction
prediction = model.predict(img)
index = np.argmax(prediction)
print(f'Prediction is {class_names[index]}')