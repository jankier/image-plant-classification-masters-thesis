img_directory = "E:\Programowanie\master-thesis\plantvillage_dataset\grayscale"

classes = ["Potato___Early_blight", 
                "Potato___healthy",
                "Potato___Late_blight"]

npy_directory = r"E:\Programowanie\master-thesis\test_train_validation_datasets\potato_gray"

training_img_data_name = "training_img_potato_gray.npy"
training_labels_data_name = "training_labels_potato_gray.npy"
test_img_data_name = "test_img_potato_gray.npy"
test_labels_data_name = "test_labels_potato_gray.npy"
validation_img_data_name = "validatio_img_potato_gray.npy"
validation_labels_data_name = "validation_labels_potato_gray.npy"

shape = (224, 224, 1)

n_categories = len(classes)

loss_parameter = 'sparse_categorical_crossentropy'

weights_directory = r"E:\Programowanie\master-thesis\weights"

best_weight = "potato_gray.weights.best.hdf5"