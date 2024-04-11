from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from architecture import create_model
from confusion_matrix import plot_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(0, "./variable_models")
from variable_models.strawberry_seg import classes, npy_directory, results_title, species_directory, test_img_data_name, test_labels_data_name, weights_directory, best_weight, shape, n_categories, loss_parameter, title, tune, dense_0, dense_1, dropout, lr

def main():
    
    # Setting parameters of CNN
    input_shape = shape
    learning_rate = lr
    optimizer = Adam(learning_rate)
    n_classes = n_categories
    fine_tune = tune
    loss = loss_parameter
    
    # Creation of the model with adjusted parameters
    final_model = create_model(input_shape, n_classes, optimizer, fine_tune, loss, dense_0, dense_1, dropout)
    
    # Loading of weights achieved from training of CNN
    final_model.load_weights(os.path.join(weights_directory, species_directory, best_weight))
    
    # Loading of previously prepared test data
    X_test = np.load(os.path.join(npy_directory, test_img_data_name), allow_pickle=True)
    y_test = np.load(os.path.join(npy_directory, test_labels_data_name), allow_pickle=True)

    # Model predicting
    final_model_predicts = final_model.predict(X_test)
    
    # Acquired predictions of the model
    predicted_classes = np.argmax(final_model_predicts, axis=1)
    
    # Accuracy test of CNN
    final_model_acc = accuracy_score(y_test, predicted_classes)
    accuracy = (title + " - model accuracy: {:.2f}%".format(final_model_acc * 100))
    print(accuracy)
    
    # Creating confusion matrix
    fig = plt.figure(figsize=(20, 10))
    plot_matrix(y_test, predicted_classes, classes, title)
    fig.suptitle("Confusion Matrix Model", fontsize=24)
    fig.tight_layout()
    
    # Creation of new directory for results
    if not os.path.exists(r"./results"):
        os.mkdir(r"./results")
    
    os.chdir(r"./results")
     
    if not os.path.exists(species_directory):
        os.mkdir(species_directory)
    
    os.chdir(species_directory)
    
    # Save accuracy result to txt file
    with open(results_title + '.txt', 'w') as f:
        f.write(accuracy)
    
    # Saving plotted image
    plt.savefig(results_title)
    
    # Plotting confusion matrix
    # plt.show()
    
main()