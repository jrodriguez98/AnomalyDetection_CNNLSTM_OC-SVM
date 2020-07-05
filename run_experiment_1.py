from utils_svm import compute_svm_experiment, compute_original_experiment, create_dirs_experiment
from utils_dataset import create_datasets
from sklearn.metrics import accuracy_score, classification_report
from constant import DATASETS_PATHS, FIX_LEN # Import constant needed
import os

def main():

    create_datasets(DATASETS_PATHS, FIX_LEN) # Extract frames from the raw videos

    create_dirs_experiment(1) # Create the experiment directory and subdirectories

    # Compute all original results 
    compute_original_experiment(1)

    list_possible_outputs = [10, 256, 1000, "flatten"]

    # Compute all SVMs results over different output layers
    for num_outputs in list_possible_outputs:
        compute_svm_experiment(1, num_outputs)
        
        
        
if __name__ == "__main__":
    main()