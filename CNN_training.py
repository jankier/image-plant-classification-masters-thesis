from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
from livelossplot.inputs.keras import PlotLossesCallback
from CNN_architecture import create_model
import numpy as np
import os
import sys
sys.path.insert(0, "E:\Programowanie\master-thesis\image-plant-classification\variable_models")
from variable_models.tomato_color import npy_directory, training_img_data_name, training_labels_data_name, validation_img_data_name, validation_labels_data_name, weights_directory, best_weight

def main():
    
    #Setting parameters of CNN
    input_shape = (224, 224, 3)
    optimizer = Adam(learning_rate=0.001)
    n_classes=10
    fine_tune=0
    
    #Addition of callbacks to improve CNN operation
    
    #PlotLosses to visualize results obtained from CNN
    plot_loss_1 = PlotLossesCallback()
    
    #ModelCheckpoint to save the best achieved weights
    tl_checkpoint_1 = ModelCheckpoint(
                                filepath= os.path.join(weights_directory, best_weight),
                                save_best_only=True,
                                verbose=1)
    
    #EarlyStopping to monitor metric and in case of stop of improvements stop the learning process
    early_stop = EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True,
                        mode='min')
    
    #Creation of the model with adjusted parameters
    final_model = create_model(input_shape, n_classes, optimizer, fine_tune)
    
    #Plotting and printing of the model summary
    # plot_model(final_model, to_file='model.png')
    print(final_model.summary())
    
    #Loading of previously prepared training data
    X_train = np.load(os.path.join(npy_directory, training_img_data_name), allow_pickle=True)
    y_train = np.load(os.path.join(npy_directory, training_labels_data_name), allow_pickle=True)
    X_val = np.load(os.path.join(npy_directory, validation_img_data_name), allow_pickle=True)
    y_val = np.load(os.path.join(npy_directory, validation_labels_data_name), allow_pickle=True)
    
    #Setting traning parameters
    batch_size = 10
    n_epochs = 1
    
    #Training of the model
    model_history = final_model.fit(
                            X_train,y_train,
                            validation_data=(X_val, y_val),      
                            batch_size=batch_size,
                            epochs=n_epochs,
                            callbacks=[tl_checkpoint_1, early_stop, plot_loss_1],
                            verbose=1)
    
    #Saving the model
    # final_model.save('CNN_image_classification.h5')
    
    print('learning finished')
    
main()