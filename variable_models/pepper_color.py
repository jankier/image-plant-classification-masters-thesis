img_directory = "./plantvillage_dataset\color"

classes = ["Pepper,_bell___Bacterial_spot", 
                "Pepper,_bell___healthy"]

npy_directory = r"./test_train_validation_datasets/pepper_color"
tuner_directory = "pepper_color"

training_img_data_name = "training_img_pepper_col.npy"
training_labels_data_name = "training_labels_pepper_col.npy"
test_img_data_name = "test_img_pepper_col.npy"
test_labels_data_name = "test_labels_pepper_col.npy"
validation_img_data_name = "validation_img_pepper_col.npy"
validation_labels_data_name = "validation_labels_pepper_col.npy"

shape = (224, 224, 3)

n_categories = len(classes)

loss_parameter = 'sparse_categorical_crossentropy'

weights_directory = r"./weights"

tune = 1

if tune > 0:
    best_weight = "pepper_color_fine_tune.weights.best.hdf5"
    title = "Pepper color (fine tune)"
else:
    best_weight = "pepper_color.weights.best.hdf5"
    title = "Pepper color"