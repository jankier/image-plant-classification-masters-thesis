img_directory = "E:\Programowanie\master-thesis\plantvillage_dataset\grayscale"

classes = ["Peach___Bacterial_spot", 
                "Peach___healthy"]

npy_directory = r"E:\Programowanie\master-thesis\test_train_validation_datasets\peach_gray"

training_img_data_name = "training_img_peach_gray.npy"
training_labels_data_name = "training_labels_peach_gray.npy"
test_img_data_name = "test_img_peach_gray.npy"
test_labels_data_name = "test_labels_peach_gray.npy"
validation_img_data_name = "validatio_img_peach_gray.npy"
validation_labels_data_name = "validation_labels_peach_gray.npy"

shape = (224, 224, 1)

n_categories = len(classes)

loss_parameter = 'sparse_categorical_crossentropy'

weights_directory = r"E:\Programowanie\master-thesis\weights"

best_weight = "peach_gray.weights.best.hdf5"