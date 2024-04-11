img_directory = "./plantvillage_dataset\segmented"

classes = ["Peach___Bacterial_spot", 
                "Peach___healthy"]

npy_directory = r"./test_train_validation_datasets/peach_seg"
species_directory = "peach_seg"

training_img_data_name = "training_img_peach_seg.npy"
training_labels_data_name = "training_labels_peach_seg.npy"
test_img_data_name = "test_img_peach_seg.npy"
test_labels_data_name = "test_labels_peach_seg.npy"
validation_img_data_name = "validation_img_peach_seg.npy"
validation_labels_data_name = "validation_labels_peach_seg.npy"

shape = (224, 224, 3)

n_categories = len(classes)

dense_0 = 4096
dense_1 = 1000
dropout = 0.2
lr = 0.001

loss_parameter = 'sparse_categorical_crossentropy'

weights_directory = r"./weights"

tune = 1

if tune > 0:
    best_weight = "peach_seg_fine_tune.weights.best.hdf5"
    title = "Peach segmented (fine tune)"
    results_title = "peach_seg_prediction_fine_tune"
else:
    best_weight = "peach_seg.weights.best.hdf5"
    title = "Peach segmented"
    results_title = "peach_seg_prediction"