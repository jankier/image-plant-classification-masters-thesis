img_directory = "./plantvillage_dataset\color"

classes = ["Pepper,_bell___Bacterial_spot", 
                "Pepper,_bell___healthy"]

npy_directory = r"./test_train_validation_datasets/pepper_color"
species_directory = "pepper_color"

training_img_data_name = "training_img_pepper_col.npy"
training_labels_data_name = "training_labels_pepper_col.npy"
test_img_data_name = "test_img_pepper_col.npy"
test_labels_data_name = "test_labels_pepper_col.npy"
validation_img_data_name = "validation_img_pepper_col.npy"
validation_labels_data_name = "validation_labels_pepper_col.npy"

shape = (224, 224, 3)

n_categories = len(classes)

dense_0 = 4096
dense_1 = 1000
dropout = 0.2
lr = 0.001

loss_parameter = 'sparse_categorical_crossentropy'

weights_directory = r"./weights"

tune = 0

if tune > 0:
    best_weight = "pepper_color_fine_tune.weights.best.hdf5"
    title = "Pepper color (fine tune)"
    results_title = "pepper_color_prediction_fine_tune"
else:
    best_weight = "pepper_color.weights.best.hdf5"
    title = "Pepper color"
    results_title = "pepper_color_prediction"