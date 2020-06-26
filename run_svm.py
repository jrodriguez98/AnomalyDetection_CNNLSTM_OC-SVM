from utils_svm import compute_all_cross
from constant import * # Import constant needed



def main():

    create_dirs()

    num_outputs = [10, 256, 1000, "flatten"]

    for num_output in num_outputs:
        for dataset_model in DATASETS_PATHS.keys()
                compute_all_cross(num_output, dataset_model)


if __name__ == "__main__":

    main()
    
    
    
