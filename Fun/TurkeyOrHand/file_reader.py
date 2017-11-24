import os
import cv2
import numpy as np

def read_image(filename, new_size=(28, 28)):
    image = cv2.imread(filename)
    final = cv2.resize(image, new_size, interpolation = cv2.INTER_AREA)
    return final.flatten()

class DataPoints:
    filenames = []
    images = []
    labels = []
    def __init__(self, categories, category_dir="training_data", split_training=True):
        for category in categories:
            for i in range(1, 40):
                filename = category_dir + "/" + category + "/" + "training_image(" + str(i) + ").jpg"
                self.filenames.append(filename)
                self.images.append(read_image(filename))
                category_val = [0, 0, 0, 0]
                category_val[categories.index(category)] = 1
                self.labels.append(category_val)
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
