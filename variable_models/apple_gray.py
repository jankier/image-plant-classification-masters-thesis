img_directory = "./plantvillage_dataset\grayscale"

classes = ["Apple___Apple_scab", 
                "Apple___Black_rot",
                "Apple___Cedar_apple_rust",
                "Apple___healthy"]

npy_directory = r"./test_train_validation_datasets/apple_gray"

training_img_data_name = "training_img_apple_gray.npy"
training_labels_data_name = "training_labels_apple_gray.npy"
test_img_data_name = "test_img_apple_gray.npy"
test_labels_data_name = "test_labels_apple_gray.npy"
validation_img_data_name = "validatio_img_apple_gray.npy"
validation_labels_data_name = "validation_labels_apple_gray.npy"

shape = (224, 224, 3)

n_categories = len(classes)

loss_parameter = 'sparse_categorical_crossentropy'

weights_directory = r"./weights"

best_weight = "apple_gray.weights.best.hdf5"

title = "Apple grayscale"