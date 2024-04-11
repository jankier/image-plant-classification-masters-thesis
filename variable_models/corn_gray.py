img_directory = "./plantvillage_dataset\grayscale"

classes = ["Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", 
                "Corn_(maize)___Common_rust_",
                "Corn_(maize)___healthy",
                "Corn_(maize)___Northern_Leaf_Blight"]

npy_directory = r"./test_train_validation_datasets/corn_gray"
species_directory = "corn_gray"

training_img_data_name = "training_img_corn_gray.npy"
training_labels_data_name = "training_labels_corn_gray.npy"
test_img_data_name = "test_img_corn_gray.npy"
test_labels_data_name = "test_labels_corn_gray.npy"
validation_img_data_name = "validation_img_corn_gray.npy"
validation_labels_data_name = "validation_labels_corn_gray.npy"

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
    best_weight = "corn_gray_fine_tune.weights.best.hdf5"
    title = "Corn grayscale (fine tune)"
    results_title = "corn_gray_prediction_fine_tune"
else:
    best_weight = "corn_gray.weights.best.hdf5"
    title = "Corn grayscale"
    results_title = "corn_gray_prediction"