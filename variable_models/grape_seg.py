img_directory = "./plantvillage_dataset\segmented"

classes = ["Grape___Black_rot", 
                "Grape___Esca_(Black_Measles)",
                "Grape___healthy",
                "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"]

npy_directory = r"./test_train_validation_datasets/grape_seg"
species_directory = "grape_seg"

training_img_data_name = "training_img_grape_seg.npy"
training_labels_data_name = "training_labels_grape_seg.npy"
test_img_data_name = "test_img_grape_seg.npy"
test_labels_data_name = "test_labels_grape_seg.npy"
validation_img_data_name = "validation_img_grape_seg.npy"
validation_labels_data_name = "validation_labels_grape_seg.npy"

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
    best_weight = "grape_seg_fine_tune.weights.best.hdf5"
    title = "Grape segmented (fine tune)"
    results_title = "grape_seg_prediction_fine_tune"
else:
    best_weight = "grape_seg.weights.best.hdf5"
    title = "Grape segmented"
    results_title = "grape_seg_prediction"