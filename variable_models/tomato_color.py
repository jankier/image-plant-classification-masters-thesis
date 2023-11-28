img_directory = "./plantvillage_dataset\color"

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

npy_directory = r"./test_train_validation_datasets/tomato_color"

training_img_data_name = "training_img_tomato_col.npy"
training_labels_data_name = "training_labels_tomato_col.npy"
test_img_data_name = "test_img_tomato_col.npy"
test_labels_data_name = "test_labels_tomato_col.npy"
validation_img_data_name = "validatio_img_tomato_col.npy"
validation_labels_data_name = "validation_labels_tomato_col.npy"

shape = (224, 224, 3)

n_categories = len(classes)

loss_parameter = 'sparse_categorical_crossentropy'

weights_directory = r"./weights"

best_weight = "tomato_color.weights.best.hdf5"

title = "Tomato color"