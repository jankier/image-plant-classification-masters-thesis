import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, "./variable_models")
from variable_models.potato_gray import img_directory, classes, npy_directory, training_img_data_name, training_labels_data_name, test_img_data_name, test_labels_data_name, validation_img_data_name, validation_labels_data_name, title

def main():
    
    # Specification of dataset directory
    data_dir = img_directory
    
    # Specification of categories of images
    categories = classes
    
    # Preparation of acquired image data
    prepared_data = []
    idx = 0
    
    # Merging of the data
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        
        for img in os.listdir(path):
            # Displaying broken data
            # if idx == 1351:
            #     print(img)
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_UNCHANGED)
                img_resized = cv2.resize(img_array, (224, 224), interpolation = cv2.INTER_AREA)
                prepared_data.append([img_resized, class_num])
            except Exception as e:
                pass
            idx = idx + 1
    
    print(len(prepared_data))
    
    # Shuffling of the data to assure diversity of neighboring data elements
    random.seed(1)
    random.shuffle(prepared_data)
    
    # Conversion of data to fit CNN requirements
    X = []
    y = []
    i = 0
    
    # Searching for broken data
    for images, labels in prepared_data:
        if np.shape(images) == (224, 224, 3):
            X.append(images)
            y.append(labels)
        else:
            print(np.shape(images))
            print(i)
        i = i + 1
    print(len(X))
    X = np.array(X)
    y = np.array(y)
    
    # Splitting the data to train/test batches
    training_img, test_img, training_labels, test_labels = train_test_split(X, y, test_size=0.20, random_state=1)
    
    # Splitting the data to train/validation batches
    training_img, validation_img, training_labels, validation_labels = train_test_split(training_img, training_labels, test_size=0.25, random_state=1)
    
    # Creation of new directory for prepared data
    if not os.path.exists(npy_directory):
        os.mkdir(npy_directory)
    
    os.chdir(npy_directory)
    
    # Saving of the data
    np.save(training_img_data_name, training_img, allow_pickle=True)
    np.save(training_labels_data_name, training_labels, allow_pickle=True)
    np.save(test_img_data_name, test_img, allow_pickle=True)
    np.save(test_labels_data_name, test_labels, allow_pickle=True)
    np.save(validation_img_data_name, validation_img, allow_pickle=True)
    np.save(validation_labels_data_name, validation_labels, allow_pickle=True)
    
    print(title + " - conversion done")
    
main()