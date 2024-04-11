img_directory = "./plantvillage_dataset\grayscale"

classes = ["Strawberry___healthy", 
                "Strawberry___Leaf_scorch"]

npy_directory = r"./test_train_validation_datasets/strawberry_gray"
species_directory = "strawberry_gray"

training_img_data_name = "training_img_strawberry_gray.npy"
training_labels_data_name = "training_labels_strawberry_gray.npy"
test_img_data_name = "test_img_strawberry_gray.npy"
test_labels_data_name = "test_labels_strawberry_gray.npy"
validation_img_data_name = "validation_img_strawberry_gray.npy"
validation_labels_data_name = "validation_labels_strawberry_gray.npy"

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
    best_weight = "strawberry_gray_fine_tune.weights.best.hdf5"
    title = "Strawberry grayscale (fine tune)"
    results_title = "strawberry_gray_prediction_fine_tune"
else:
    best_weight = "strawberry_gray.weights.best.hdf5"
    title = "Strawberry grayscale"
    results_title = "strawberry_gray_prediction"