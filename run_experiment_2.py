from utils_svm import compute_svm_experiment, compute_original_experiment, create_dirs
from constant import DATASETS_PATHS # Import constant needed



def main():

    create_dirs()

    num_outputs = [10, 256, 1000, "flatten"]

    for dataset_model in DATASETS_PATHS.keys():
        compute_original_experiment(2, dataset_model) # Compute the results of the original model with cross model-dataset for experiment 2

    for dataset_model in DATASETS_PATHS.keys():
        for num_output in num_outputs:

            print("DATASET MODEL: " + str(dataset_model))
            print("NUM OUTPUTS: " + str(num_output))
            compute_svm_experiment(2, num_output, dataset_model)


    
        
if __name__ == "__main__":

    main()