img_directory = "E:\Programowanie\master-thesis\plantvillage_dataset\segmented"

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

npy_directory = r"E:\Programowanie\master-thesis\test_train_validation_datasets\tomato_seg"

training_img_data_name = "training_img_tomato_seg"
training_labels_data_name = "training_labels_tomato_col"
test_img_data_name = "test_img_tomato_seg"
test_labels_data_name = "test_labels_tomato_seg"
validation_img_data_name = "validatio_img_tomato_seg"
validation_labels_data_name = "validation_labels_tomato_seg"

weights_directory = r"E:\Programowanie\master-thesis\weights"

best_weight = "tomato_seg.weights.best.hdf5"