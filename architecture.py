from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten

def create_model(input_shape, n_classes, optimizer, fine_tune, loss, dense_0, dense_1, dropout):
    
    # Importing VGG16 pre-trained convolutional layers without VGG16 fully-connected layers
    conv_model = VGG16(
        include_top = False,
        weights = 'imagenet',
        input_shape = input_shape,
    )
    
    # Creating definition of how many layers should be frozen during training
    if fine_tune > 0:
        for layer in conv_model.layers[:-fine_tune]:
            layer.trainable = False
    else:
        for layer in conv_model.layers:
            layer.trainable = False
            
    # Creation of custom fully-connected layers to suit data
    top_model = conv_model.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(dense_0, activation='relu')(top_model) # 4096
    top_model = Dense(dense_1, activation='relu')(top_model) # 1000
    top_model = Dropout(dropout)(top_model) # 0.2
    output_layer = Dense(n_classes, activation='softmax')(top_model)
    
    # Grouping of convolutional layers and newly created fully-connected layers into new model
    model = Model(inputs=conv_model.input, outputs=output_layer)
    
    # Compilation of the model for training
    model.compile(
        optimizer = optimizer,
        loss = loss,
        metrics = ['accuracy'],
    )

    return model