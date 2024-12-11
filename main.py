# Image Classification in Python
import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 
from tensorflow import datasets, layers, models

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()