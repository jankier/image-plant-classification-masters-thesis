img_directory = "./plantvillage_dataset\grayscale"

classes = ["Cherry_(including_sour)___healthy", 
                "Cherry_(including_sour)___Powdery_mildew"]

npy_directory = r"./test_train_validation_datasets/cherry_gray"
species_directory = "cherry_gray"

training_img_data_name = "training_img_cherry_gray.npy"
training_labels_data_name = "training_labels_cherry_gray.npy"
test_img_data_name = "test_img_cherry_gray.npy"
test_labels_data_name = "test_labels_cherry_gray.npy"
validation_img_data_name = "validation_img_cherry_gray.npy"
validation_labels_data_name = "validation_labels_cherry_gray.npy"

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
    best_weight = "cherry_gray_fine_tune.weights.best.hdf5"
    title = "Cherry grayscale (fine tune)"
    results_title = "cherry_gray_prediction_fine_tune"
else:
    best_weight = "cherry_gray.weights.best.hdf5"
    title = "Cherry grayscale"
    results_title = "cherry_gray_prediction"