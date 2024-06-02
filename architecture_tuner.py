from keras.optimizers import Adam
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten
import keras_tuner
import numpy as np
import os
from dotenv import load_dotenv
import sys
sys.path.insert(0, "./variable_models")
from variable_models.tomato_seg import npy_directory, species_directory, training_img_data_name, training_labels_data_name, validation_img_data_name, validation_labels_data_name, shape, n_categories, loss_parameter, tune, title

load_dotenv()

# Loading of previously prepared training data
X_train = np.load(os.path.join(npy_directory, training_img_data_name), allow_pickle=True)
y_train = np.load(os.path.join(npy_directory, training_labels_data_name), allow_pickle=True)
X_val = np.load(os.path.join(npy_directory, validation_img_data_name), allow_pickle=True)
y_val = np.load(os.path.join(npy_directory, validation_labels_data_name), allow_pickle=True)

# Creation of new directory for prepared data
if not os.path.exists(r"./tuner_results"):
    os.mkdir(r"./tuner_results")

os.chdir(r"./tuner_results")

# Specifying log directory
LOG_DIR = species_directory

def create_model(hp):

    # Importing VGG16 pre-trained convolutional layers without VGG16 fully-connected layers
    conv_model = VGG16(
        include_top = False,
        weights = 'imagenet',
        input_shape = shape,
    )
    
    # Creating definition of how many layers should be frozen during training
    if tune > 0:
        for layer in conv_model.layers[:-tune]:
            layer.trainable = False
    else:
        for layer in conv_model.layers:
            layer.trainable = False
            
    # Creation of custom fully-connected layers to suit data (with hyperparameters)
    top_model = conv_model.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(hp.Int('dense_0', min_value=256, max_value=4096, step=256), activation='relu')(top_model)
    top_model = Dense(hp.Int('dense_1', min_value=100, max_value=1000, step=100), activation='relu')(top_model)
    top_model = Dropout(hp.Float("dropout", min_value=0.1, max_value=0.6, step=0.1))(top_model)
    output_layer = Dense(n_categories, activation='softmax')(top_model)
    
    # Grouping of convolutional layers and newly created fully-connected layers into new model
    model = Model(inputs=conv_model.input, outputs=output_layer)
    
    # Defining optimizer learning rate (with hyperparameter)
    lr = hp.Float("lr", min_value=0.0001, max_value=0.01, sampling="log")
    
    # Compilation of the model for training
    model.compile(
        optimizer = Adam(learning_rate=lr),
        loss = loss_parameter,
        metrics = ['accuracy'],
    )

    return model

# Defining tuner type and its parameters
tuner = keras_tuner.RandomSearch( # BayesianOptimization - other option
    create_model,
    objective='val_accuracy',
    max_trials=40, # how many model variations to test
    executions_per_trial=2, # how many trials per variation (same model could perform differently)
    directory=LOG_DIR)

# List hyperparameters
tuner.search_space_summary()

# Running tuner search with specified parameters
tuner.search(x=X_train,
             y=y_train,
             epochs=3,
             batch_size=10,
             validation_data=(X_val,y_val))

# Printing results
tuner.results_summary()