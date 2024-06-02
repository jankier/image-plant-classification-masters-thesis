img_directory = "./plantvillage_dataset\grayscale"

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

npy_directory = r"./test_train_validation_datasets/tomato_gray"
species_directory = "tomato_gray"

training_img_data_name = "training_img_tomato_gray.npy"
training_labels_data_name = "training_labels_tomato_gray.npy"
test_img_data_name = "test_img_tomato_gray.npy"
test_labels_data_name = "test_labels_tomato_gray.npy"
validation_img_data_name = "validation_img_tomato_gray.npy"
validation_labels_data_name = "validation_labels_tomato_gray.npy"

shape = (224, 224, 3)

n_categories = len(classes)

loss_parameter = 'sparse_categorical_crossentropy'

weights_directory = r"./weights"

tune = 1

tuner_version = 1 # 0 - Random Search | 1 - Baysian Optimization

if tuner_version == 1:
    
    if tune > 0:
        dense_0 = 1792
        dense_1 = 1000
        dropout = 0.30000000000000004
        lr = 0.0001
        
        best_weight = "tomato_gray_tuned_baysian_optimization_fine_tune_" + str(tune) + ".weights.best.hdf5"
        title = "Tomato grayscale (baysian optimization + fine tune = " + str(tune) + ")"
        results_title = "tomato_gray_prediction_tuned_baysian_optimization_fine_tune_" + str(tune)
    else:
        dense_0 = 256
        dense_1 = 300
        dropout = 0.2
        lr = 0.0001406653245871662
        
        best_weight = "tomato_gray_tuned_baysian_optimization.weights.best.hdf5"
        title = "Tomato grayscale tuned (baysian optimization)"
        results_title = "tomato_gray_prediction_tuned_baysian_optimization"

else:
    
    if tune > 0:
        dense_0 = 1536
        dense_1 = 800
        dropout = 0.5
        lr = 0.00033235390013150126
        
        best_weight = "tomato_gray_tuned_random_search_fine_tune_" + str(tune) + ".weights.best.hdf5"
        title = "Tomato grayscale (random search + fine tune = " + str(tune) + ")"
        results_title = "tomato_gray_prediction_tuned_random_search_fine_tune_" + str(tune)
    else:
        dense_0 = 1792
        dense_1 = 1000
        dropout = 0.2
        lr = 0.00027196868413302354
        
        best_weight = "tomato_gray_tuned_random_search.weights.best.hdf5"
        title = "Tomato grayscale tuned (random search)"
        results_title = "tomato_gray_prediction_tuned_random_search"