img_directory = "./plantvillage_dataset\segmented"

classes = ["Apple___Apple_scab", 
                "Apple___Black_rot",
                "Apple___Cedar_apple_rust",
                "Apple___healthy"]

npy_directory = r"./test_train_validation_datasets/apple_seg"
species_directory = "apple_seg"

training_img_data_name = "training_img_apple_seg.npy"
training_labels_data_name = "training_labels_apple_seg.npy"
test_img_data_name = "test_img_apple_seg.npy"
test_labels_data_name = "test_labels_apple_seg.npy"
validation_img_data_name = "validation_img_apple_seg.npy"
validation_labels_data_name = "validation_labels_apple_seg.npy"

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
    best_weight = "apple_seg_fine_tune.weights.best.hdf5"
    title = "Apple segmented (fine tune)"
    results_title = "apple_seg_prediction_fine_tune"
else:
    best_weight = "apple_seg.weights.best.hdf5"
    title = "Apple segmented"
    results_title = "apple_seg_prediction"