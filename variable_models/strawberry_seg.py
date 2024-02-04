img_directory = "./plantvillage_dataset\segmented"

classes = ["Strawberry___healthy", 
                "Strawberry___Leaf_scorch"]

npy_directory = r"./test_train_validation_datasets/strawberry_seg"
tuner_directory = "strawberry_seg"

training_img_data_name = "training_img_strawberry_seg.npy"
training_labels_data_name = "training_labels_strawberry_seg.npy"
test_img_data_name = "test_img_strawberry_seg.npy"
test_labels_data_name = "test_labels_strawberry_seg.npy"
validation_img_data_name = "validation_img_strawberry_seg.npy"
validation_labels_data_name = "validation_labels_strawberry_seg.npy"

shape = (224, 224, 3)

n_categories = len(classes)

loss_parameter = 'sparse_categorical_crossentropy'

weights_directory = r"./weights"

tune = 1

if tune > 0:
    best_weight = "strawberry_seg_fine_tune.weights.best.hdf5"
    title = "Strawberry segmented (fine tune)"
else:
    best_weight = "strawberry_seg.weights.best.hdf5"
    title = "Strawberry segmented"