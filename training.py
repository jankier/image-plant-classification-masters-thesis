from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from livelossplot.inputs.keras import PlotLossesCallback
from architecture import create_model
import time
import neptune
import numpy as np
import os
from dotenv import load_dotenv
import sys
sys.path.insert(0, "./variable_models")
from variable_models.tomato_seg_tuned import npy_directory, training_img_data_name, training_labels_data_name, validation_img_data_name, validation_labels_data_name, weights_directory, best_weight, shape, n_categories, loss_parameter, title, tune, species_directory, dense_0, dense_1, dropout, lr

load_dotenv()

def main():
    
    # Initializing neptune.ai
    run = neptune.init_run(
    tags=[title],
    project=os.getenv("PROJECT"),
    api_token=os.getenv("NEPTUNE_AI_TOKEN"),
    )
    
    # Setting parameters of CNN
    input_shape = shape
    learning_rate = lr
    optimizer = Adam(learning_rate)
    n_classes = n_categories
    fine_tune = tune
    loss = loss_parameter
    
    # Addition of callbacks to improve CNN operation
    
    # ModelCheckpoint to save the best achieved weights
    tl_checkpoint_1 = ModelCheckpoint(
                                filepath = os.path.join(weights_directory, species_directory, best_weight),
                                save_best_only = True,
                                verbose = 1)
    
    # EarlyStopping to monitor metric and in case of stop of improvements stop the learning process
    early_stop = EarlyStopping(
                        monitor = 'val_loss',
                        patience = 5,
                        restore_best_weights = True,
                        mode = 'min')
    
    # PlotLossesCallback to plot the results of current training run
    plot_loss_1 = PlotLossesCallback()
    
    # Creation of the model with adjusted parameters
    final_model = create_model(input_shape, n_classes, optimizer, fine_tune, loss, dense_0, dense_1, dropout)
    
    # Printing of the model summary
    print(final_model.summary())
    
    # Loading of previously prepared training data
    X_train = np.load(os.path.join(npy_directory, training_img_data_name), allow_pickle=True)
    y_train = np.load(os.path.join(npy_directory, training_labels_data_name), allow_pickle=True)
    X_val = np.load(os.path.join(npy_directory, validation_img_data_name), allow_pickle=True)
    y_val = np.load(os.path.join(npy_directory, validation_labels_data_name), allow_pickle=True)
    
    # Setting traning parameters
    batch_size = 10
    n_epochs = 20
    
    # Passing params to neptune.ai
    params = {"name": title, "n_classes": n_classes,
              "learning_rate": lr, "optimizer": "Adam", 
              "loss": loss, "batch_size": batch_size, 
              "n_epochs": n_epochs, "fine-tune": tune,
              "network": {"dense0":dense_0, "dense1":dense_1, "dropout":dropout}}
    
    run["parameters"] = params

    # Training of the model
    model_history = final_model.fit(
                        X_train,y_train,
                        validation_data = (X_val, y_val),      
                        batch_size = batch_size,
                        epochs = n_epochs,
                        callbacks = [tl_checkpoint_1], #, early_stop , plot_loss_1
                        verbose=1)
    
    # Pass results to neptune.ai
    for epoch in range(n_epochs):
        run["epoch/loss"].append(model_history.history["loss"][epoch])
        run["epoch/accuracy"].append(model_history.history["accuracy"][epoch])
        run["epoch/val_loss"].append(model_history.history["val_loss"][epoch])
        run["epoch/val_accuracy"].append(model_history.history["val_accuracy"][epoch])
    
    run.stop()
    
    print(title + ' - learning finished')
    
main()