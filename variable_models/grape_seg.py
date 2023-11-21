img_directory = "E:\Programowanie\master-thesis\plantvillage_dataset\segmented"

classes = ["Grape___Black_rot", 
                "Grape___Esca_(Black_Measles)",
                "Grape___healthy",
                "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"]

npy_directory = r"E:\Programowanie\master-thesis\test_train_validation_datasets\grape_seg"

training_img_data_name = "training_img_grape_seg.npy"
training_labels_data_name = "training_labels_grape_col.npy"
test_img_data_name = "test_img_grape_seg.npy"
test_labels_data_name = "test_labels_grape_seg.npy"
validation_img_data_name = "validatio_img_grape_seg.npy"
validation_labels_data_name = "validation_labels_grape_seg.npy"

shape = (224, 224, 3)

n_categories = len(classes)

loss_parameter = 'sparse_categorical_crossentropy'

weights_directory = r"E:\Programowanie\master-thesis\weights"

best_weight = "grape_seg.weights.best.hdf5"