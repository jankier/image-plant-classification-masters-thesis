img_directory = "./plantvillage_dataset\segmented"

classes = ["Tomato___Bacterial_spot", 
                "Tomato___Early_blight",
                "Tomato___healthy",
                "Tomato___Late_blight",
                "Tomato___Leaf_Mold",
                "Tomato___Mosaic_virus",
                "Tomato___Septoria_leaf_spot",
                "Tomato___Spider_mites",
                "Tomato___Target_Spot",
                "Tomato___Yellow_Leaf_Curl_Virus"]

npy_directory = r"./test_train_validation_datasets/tomato_seg"
species_directory = "tomato_seg"

training_img_data_name = "training_img_tomato_seg.npy"
training_labels_data_name = "training_labels_tomato_seg.npy"
test_img_data_name = "test_img_tomato_seg.npy"
test_labels_data_name = "test_labels_tomato_seg.npy"
validation_img_data_name = "validation_img_tomato_seg.npy"
validation_labels_data_name = "validation_labels_tomato_seg.npy"

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
    best_weight = "tomato_seg_fine_tune_" + str(tune) + ".weights.best.hdf5"
    title = "Tomato segmented (fine tune = " + str(tune) + ")"
    results_title = "tomato_seg_prediction_fine_tune_" + str(tune)
else:
    best_weight = "tomato_seg.weights.best.hdf5"
    title = "Tomato segmented"
    results_title = "tomato_seg_prediction"