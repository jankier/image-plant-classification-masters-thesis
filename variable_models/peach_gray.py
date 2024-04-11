img_directory = "./plantvillage_dataset\grayscale"

classes = ["Peach___Bacterial_spot", 
                "Peach___healthy"]

npy_directory = r"./test_train_validation_datasets/peach_gray"
species_directory = "peach_gray"

training_img_data_name = "training_img_peach_gray.npy"
training_labels_data_name = "training_labels_peach_gray.npy"
test_img_data_name = "test_img_peach_gray.npy"
test_labels_data_name = "test_labels_peach_gray.npy"
validation_img_data_name = "validation_img_peach_gray.npy"
validation_labels_data_name = "validation_labels_peach_gray.npy"

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
    best_weight = "peach_gray_fine_tune.weights.best.hdf5"
    title = "Peach grayscale (fine tune)"
    results_title = "peach_gray_prediction_fine_tune"
else:
    best_weight = "peach_gray.weights.best.hdf5"
    title = "Peach grayscale"
    results_title = "peach_gray_prediction"