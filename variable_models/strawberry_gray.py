img_directory = "./plantvillage_dataset\grayscale"

classes = ["Strawberry___healthy", 
                "Strawberry___Leaf_scorch"]

npy_directory = r"./test_train_validation_datasets/strawberry_gray"

training_img_data_name = "training_img_strawberry_gray.npy"
training_labels_data_name = "training_labels_strawberry_gray.npy"
test_img_data_name = "test_img_strawberry_gray.npy"
test_labels_data_name = "test_labels_strawberry_gray.npy"
validation_img_data_name = "validatio_img_strawberry_gray.npy"
validation_labels_data_name = "validation_labels_strawberry_gray.npy"

shape = (224, 224, 3)

n_categories = len(classes)

loss_parameter = 'sparse_categorical_crossentropy'

weights_directory = r"./weights"

best_weight = "strawberry_gray.weights.best.hdf5"

title = "Strawberry grayscale"