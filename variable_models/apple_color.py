img_directory = "./plantvillage_dataset\color"

classes = ["Apple___Apple_scab", 
                "Apple___Black_rot",
                "Apple___Cedar_apple_rust",
                "Apple___healthy"]

npy_directory = r"./test_train_validation_datasets/apple_color"
species_directory = "apple_color"

training_img_data_name = "training_img_apple_col.npy"
training_labels_data_name = "training_labels_apple_col.npy"
test_img_data_name = "test_img_apple_col.npy"
test_labels_data_name = "test_labels_apple_col.npy"
validation_img_data_name = "validation_img_apple_col.npy"
validation_labels_data_name = "validation_labels_apple_col.npy"

shape = (224, 224, 3)

n_categories = len(classes)

dense_0 = 4096
dense_1 = 1000
dropout = 0.2
lr = 0.001

loss_parameter = 'sparse_categorical_crossentropy'

weights_directory = r"./weights"

tune = 1

if tune > 0:
    best_weight = "apple_color_fine_tune.weights.best.hdf5"
    title = "Apple color (fine tune)"
    results_title = "apple_color_prediction_fine_tune"
else:
    best_weight = "apple_color.weights.best.hdf5"
    title = "Apple color"
    results_title = "apple_color_prediction"