img_directory = "E:\Programowanie\master-thesis\plantvillage_dataset\segmented"

classes = ["Potato___Early_blight", 
                "Potato___healthy",
                "Potato___Late_blight"]

npy_directory = r"E:\Programowanie\master-thesis\test_train_validation_datasets\potato_seg"

training_img_data_name = "training_img_potato_seg.npy"
training_labels_data_name = "training_labels_potato_seg.npy"
test_img_data_name = "test_img_potato_seg.npy"
test_labels_data_name = "test_labels_potato_seg.npy"
validation_img_data_name = "validatio_img_potato_seg.npy"
validation_labels_data_name = "validation_labels_potato_seg.npy"

shape = (224, 224, 3)

n_categories = len(classes)

loss_parameter = 'sparse_categorical_crossentropy'

weights_directory = r"E:\Programowanie\master-thesis\weights"

best_weight = "potato_seg.weights.best.hdf5"