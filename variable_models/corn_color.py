img_directory = "E:\Programowanie\master-thesis\plantvillage_dataset\color"

classes = ["Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", 
                "Corn_(maize)___Common_rust_",
                "Corn_(maize)___healthy",
                "Corn_(maize)___Northern_Leaf_Blight"]

npy_directory = r"E:\Programowanie\master-thesis\test_train_validation_datasets\corn_color"

training_img_data_name = "training_img_corn_col.npy"
training_labels_data_name = "training_labels_corn_col.npy"
test_img_data_name = "test_img_corn_col.npy"
test_labels_data_name = "test_labels_corn_col.npy"
validation_img_data_name = "validatio_img_corn_col.npy"
validation_labels_data_name = "validation_labels_corn_col.npy"

shape = (224, 224, 3)

n_categories = len(classes)

loss_parameter = 'sparse_categorical_crossentropy'

weights_directory = r"E:\Programowanie\master-thesis\weights"

best_weight = "corn_color.weights.best.hdf5"