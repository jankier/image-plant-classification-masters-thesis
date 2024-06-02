from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from architecture import create_model
from confusion_matrix import plot_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(0, "./variable_models")
from variable_models.tomato_seg_tuned import classes, npy_directory, results_title, species_directory, test_img_data_name, test_labels_data_name, weights_directory, best_weight, shape, n_categories, loss_parameter, title, tune, dense_0, dense_1, dropout, lr

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
    
    # Accuracy, precision, recall and f-score of CNN
    final_model_accuracy = accuracy_score(y_test, predicted_classes)
    accuracy = ("accuracy: {:.2f}%".format(final_model_accuracy * 100))

    final_model_precision = precision_score(y_test, predicted_classes, average='macro')
    precision = ("precision: {:.2f}%".format(final_model_precision * 100))
    
    final_model_recall = recall_score(y_test, predicted_classes, average='macro')
    recall = ("recall: {:.2f}%".format(final_model_recall * 100))
    
    final_model_f_score = f1_score(y_test, predicted_classes, average='macro')
    f_score = ("f-measure: {:.2f}%".format(final_model_f_score * 100))
    print(title + ":", accuracy, precision, recall, f_score, sep="\n")
    
    # Creating confusion matrix
    fig = plt.figure(figsize=(10, 10))
    plot_matrix(y_test, predicted_classes, classes, title)
    # fig.suptitle("Confusion Matrix Model", fontsize=24)
    fig.tight_layout()
    
    # Creation of new directory for results
    if not os.path.exists(r"./results"):
        os.mkdir(r"./results")
    
    os.chdir(r"./results")
     
    if not os.path.exists(species_directory):
        os.mkdir(species_directory)
    
    os.chdir(species_directory)
    
    # Save accuracy, precision, recall and f-measure results to txt file
    metrics = [title + ":", accuracy, precision, recall, f_score]
    with open(results_title + '.txt', 'w') as f:
        f.writelines("\n".join(metrics))
    
    # Saving plotted image
    plt.savefig(results_title)
    
    # Plotting confusion matrix
    # plt.show()
    
main()