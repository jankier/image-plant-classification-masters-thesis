img_directory = "./plantvillage_dataset\segmented"

classes = ["Potato___Early_blight", 
                "Potato___healthy",
                "Potato___Late_blight"]

npy_directory = r"./test_train_validation_datasets/potato_seg"
tuner_directory = "potato_seg"

training_img_data_name = "training_img_potato_seg.npy"
training_labels_data_name = "training_labels_potato_seg.npy"
test_img_data_name = "test_img_potato_seg.npy"
test_labels_data_name = "test_labels_potato_seg.npy"
validation_img_data_name = "validation_img_potato_seg.npy"
validation_labels_data_name = "validation_labels_potato_seg.npy"

shape = (224, 224, 3)

n_categories = len(classes)

loss_parameter = 'sparse_categorical_crossentropy'

weights_directory = r"./weights"

tune = 1

if tune > 0:
    best_weight = "potato_seg_fine_tune.weights.best.hdf5"
    title = "Potato segmented (fine tune)"
else:
    best_weight = "potato_seg.weights.best.hdf5"
    title = "Potato segmented"