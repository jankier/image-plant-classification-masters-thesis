img_directory = "./plantvillage_dataset\grayscale"

classes = ["Pepper,_bell___Bacterial_spot", 
                "Pepper,_bell___healthy"]

npy_directory = r"./test_train_validation_datasets/pepper_gray"
species_directory = "pepper_gray"

training_img_data_name = "training_img_pepper_gray.npy"
training_labels_data_name = "training_labels_pepper_gray.npy"
test_img_data_name = "test_img_pepper_gray.npy"
test_labels_data_name = "test_labels_pepper_gray.npy"
validation_img_data_name = "validation_img_pepper_gray.npy"
validation_labels_data_name = "validation_labels_pepper_gray.npy"

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
    best_weight = "pepper_gray_fine_tune.weights.best.hdf5"
    title = "Pepper grayscale (fine tune)"
    results_title = "pepper_gray_prediction_fine_tune"
else:
    best_weight = "pepper_gray.weights.best.hdf5"
    title = "Pepper grayscale"
    results_title = "pepper_gray_prediction"