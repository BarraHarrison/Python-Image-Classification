## Image Classification with Python

This project demonstrates a simple image classification program built using Python, TensorFlow, and Keras.
It trains a Convolutional Neural Network (CNN) on the CIFAR-10 dataset and evaluates its performance on unseen test data.
The program also includes functionality to classify a custom image.

## What does this program do?
- Trains a CNN model on the CIFAR-10 dataset (a collection of 60,000 32x32 color images in 10 classes such as planes, cars, birds, and more).
- Evaluates the trained model to determine its accuracy and loss on the test dataset.
- Allows users to classify a custom image (car_image.avif) using the trained model.

## Training Results
The CNN was trained for 10 epochs, and here are the results:
- Final Training Accuracy: ~80.54%
- Final Validation Accuracy: ~71.90%
- Final Validation Loss: ~0.91

## What does this even mean?
- The model correctly classifies approximately 72% of unseen images in the CIFAR-10 test set.
- The training accuracy (80.54%) is higher than the validation accuracy, which suggests slight overfitting, meaning the model performs better on the training data than on unseen data.

