from utils_svm import compute_svm_experiment, compute_original_experiment, create_dirs
from sklearn.metrics import accuracy_score, classification_report
from constant import * # Import constant needed

def main():

    create_dirs()

    list_possible_outputs = [10, 256, 1000, "flatten"]

    # Compute all SVMs results over different output layers
    for num_outputs in list_possible_outputs:
        compute_svm_experiment(1, num_outputs)
        
        
    # Compute all original results 
    compute_original_experiment(1)
        


if __name__ == "__main__":
    main()