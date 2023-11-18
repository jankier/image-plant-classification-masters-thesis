import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
from sklearn.model_selection import train_test_split

def main():
    
    # Specification of dataset directory
    data_dir = "E:\Programowanie\magisterka\plantvillage_dataset\color"
    
    # Specification of categories of images
    categories = ["Tomato___Bacterial_spot", 
                  "Tomato___Early_blight",
                  "Tomato___healthy",
                  "Tomato___Late_blight",
                  "Tomato___Leaf_Mold",
                  "Tomato___Mosaic_virus",
                  "Tomato___Septoria_leaf_spot",
                  "Tomato___Spider_mites",
                  "Tomato___Target_Spot",
                  "Tomato___Yellow_Leaf_Curl_Virus"]
    
    #Preparation of acquired image data
    prepared_data = []
    
    #Merging of the data
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_UNCHANGED)
                img_resized = cv2.resize(img_array, (224, 224), interpolation = cv2.INTER_AREA)
                prepared_data.append([img_resized, class_num])
            except Exception as e:
                pass
            
    print(len(prepared_data))
    
    #Shuffling of the data to assure diversity of neighboring data elements
    random.shuffle(prepared_data)
    
    #Conversion of data to fit CNN requirements
    X = []
    y = []
    
    for images, labels in prepared_data:
        X.append(images)
        y.append(labels)
    
    X = np.array(X)
    y = np.array(y)
    
    #Splitting the data to train/test batches
    training_img, test_img, training_labels, test_labels = train_test_split(X, y, test_size=0.30, random_state=42)
    
    #Saving of the data
    np.save('training_img.npy',training_img, allow_pickle=True)
    np.save('test_img.npy',test_img, allow_pickle=True)
    np.save('training_labels.npy',training_labels, allow_pickle=True)
    np.save('test_labels.npy',test_labels, allow_pickle=True)
    
    print("conversion done")
    
main()