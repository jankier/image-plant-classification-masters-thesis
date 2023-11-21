img_directory = "E:\Programowanie\master-thesis\plantvillage_dataset\color"

classes = ["Apple___Apple_scab", 
                "Apple___Black_rot",
                "Apple___Cedar_apple_rust",
                "Apple___healthy"]

npy_directory = r"E:\Programowanie\master-thesis\test_train_validation_datasets\apple_color"

training_img_data_name = "training_img_apple_col.npy"
training_labels_data_name = "training_labels_apple_col.npy"
test_img_data_name = "test_img_apple_col.npy"
test_labels_data_name = "test_labels_apple_col.npy"
validation_img_data_name = "validatio_img_apple_col.npy"
validation_labels_data_name = "validation_labels_apple_col.npy"

shape = (224, 224, 3)

n_categories = len(classes)

loss_parameter = 'sparse_categorical_crossentropy'

weights_directory = r"E:\Programowanie\master-thesis\weights"

best_weight = "apple_color.weights.best.hdf5"