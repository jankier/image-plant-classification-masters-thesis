img_directory = "./plantvillage_dataset\segmented"

classes = ["Pepper,_bell___Bacterial_spot", 
                "Pepper,_bell___healthy"]

npy_directory = r"./test_train_validation_datasets/pepper_seg"

training_img_data_name = "training_img_pepper_seg.npy"
training_labels_data_name = "training_labels_pepper_seg.npy"
test_img_data_name = "test_img_pepper_seg.npy"
test_labels_data_name = "test_labels_pepper_seg.npy"
validation_img_data_name = "validatio_img_pepper_seg.npy"
validation_labels_data_name = "validation_labels_pepper_seg.npy"

shape = (224, 224, 3)

n_categories = len(classes)

loss_parameter = 'sparse_categorical_crossentropy'

weights_directory = r"./weights"

best_weight = "pepper_seg.weights.best.hdf5"

title = "Pepper segmented"