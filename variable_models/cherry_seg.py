img_directory = "E:\Programowanie\master-thesis\plantvillage_dataset\segmented"

classes = ["Cherry_(including_sour)___healthy", 
                "Cherry_(including_sour)___Powdery_mildew"]

npy_directory = r"E:\Programowanie\master-thesis\test_train_validation_datasets\cherry_seg"

training_img_data_name = "training_img_cherry_seg.npy"
training_labels_data_name = "training_labels_cherry_seg.npy"
test_img_data_name = "test_img_cherry_seg.npy"
test_labels_data_name = "test_labels_cherry_seg.npy"
validation_img_data_name = "validatio_img_cherry_seg.npy"
validation_labels_data_name = "validation_labels_cherry_seg.npy"

shape = (224, 224, 3)

n_categories = len(classes)

loss_parameter = 'sparse_categorical_crossentropy'

weights_directory = r"E:\Programowanie\master-thesis\weights"

best_weight = "cherry_seg.weights.best.hdf5"