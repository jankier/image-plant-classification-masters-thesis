from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from CNN_architecture import create_model
import numpy as np
import os
import sys
sys.path.insert(0, "E:\Programowanie\master-thesis\image-plant-classification\variable_models")
from variable_models.tomato_gray import npy_directory, test_img_data_name, test_labels_data_name, weights_directory, best_weight

def main():
        
    #Setting parameters of CNN
    input_shape = (224, 224, 3)
    optimizer = Adam(learning_rate=0.001)
    n_classes=10
    fine_tune=0
    
    #Creation of the model with adjusted parameters
    final_model = create_model(input_shape, n_classes, optimizer, fine_tune)
    
    #Loading of weights achieved from training of CNN
    final_model.load_weights(os.path.join(weights_directory, best_weight)) # initialize the best trained weights
    
    #Loading of previously prepared test data
    X_test = np.load(os.path.join(npy_directory, test_img_data_name), allow_pickle=True)
    y_test = np.load(os.path.join(npy_directory, test_labels_data_name), allow_pickle=True)

    #Model predicting
    final_model_predicts = final_model.predict(X_test)
    
    #Acquired predictions of the model
    predicted_classes = np.argmax(final_model_predicts, axis=1)
    
    #Accuracy test of CNN
    final_model_acc = accuracy_score(y_test, predicted_classes)
    print("CNN Image Classification Model Accuracy: {:.2f}%".format(final_model_acc * 100))
    
main()