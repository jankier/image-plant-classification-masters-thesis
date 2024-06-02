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

loss_parameter = 'sparse_categorical_crossentropy'

weights_directory = r"./weights"

tune = 1

tuner_version = 1 # 0 - Random Search | 1 - Baysian Optimization 

if tuner_version == 1:
    
    if tune > 0:
        dense_0 = 256
        dense_1 = 1000
        dropout = 0.5
        lr = 0.0001
        
        best_weight = "tomato_seg_tuned_baysian_optimization_fine_tune_" + str(tune) + ".weights.best.hdf5"
        title = "Tomato segmented (baysian optimization + fine tune = " + str(tune) + ")"
        results_title = "tomato_seg_prediction_tuned_baysian_optimization_fine_tune_" + str(tune)
    else:
        dense_0 = 768
        dense_1 = 700
        dropout = 0.1
        lr = 0.0001
        
        best_weight = "tomato_seg_tuned_baysian_optimization.weights.best.hdf5"
        title = "Tomato seg tuned (baysian optimization)"
        results_title = "tomato_seg_prediction_tuned_baysian_optimization"

else:
    
    if tune > 0:
        dense_0 = 512
        dense_1 = 500
        dropout = 0.1
        lr = 0.00010367464190947009
        
        best_weight = "tomato_seg_tuned_random_search_fine_tune_" + str(tune) + ".weights.best.hdf5"
        title = "Tomato segmented (random search + fine tune = " + str(tune) + ")"
        results_title = "tomato_seg_prediction_tuned_random_search_fine_tune_" + str(tune)
    else:
        dense_0 = 2048
        dense_1 = 800
        dropout = 0.1
        lr = 0.00019682683962500205
        
        best_weight = "tomato_seg_tuned_random_search.weights.best.hdf5"
        title = "Tomato seg tuned (random search)"
        results_title = "tomato_seg_prediction_tuned_random_search"