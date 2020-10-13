from utils_svm import compute_svm_experiment, compute_original_experiment, create_dirs_experiment
from config import DATASETS_PATHS, FIX_LEN # Import config needed
from utils_dataset import create_datasets



def main():

    create_datasets(DATASETS_PATHS, FIX_LEN) # Extract frames from the raw videos

    create_dirs_experiment(2) # # Create the experiment directory and subdirectories

    num_outputs = [10, 256]

    for dataset_model in DATASETS_PATHS.keys():
        compute_original_experiment(2, dataset_model) # Compute the results of the original model with cross model-dataset for experiment 2
        for num_output in num_outputs:
            compute_svm_experiment(2, num_output, dataset_model)
        


    
        
if __name__ == "__main__":

    main()