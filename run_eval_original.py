from utils_svm import compute_representation_model_dataset
from sklearn.metrics import accuracy_score, classification_report
from constant import * # Import constant needed

def main():

    for dataset_name in DATASETS_PATHS.keys():
        for dataset_model in DATASETS_PATHS.keys():
            _, pred_y, test_y = compute_representation_model_dataset(dataset_model, dataset_name, "all", False)
            result = eval_model(pred_y, test_y)
            if (dataset_model != dataset_name):
                pd.DataFrame(data=result, dtype=np.float).to_csv("experiment_2/results_cross_original/dataset_{}_model_{}.csv".format(dataset_name, dataset_model))
            else:
                pd.DataFrame(data=result, dtype=np.float).to_csv("experiment_1/results_original/dataset_{}_model_{}.csv".format(dataset_name, dataset_model))


if __name__ == "__main__":
    main()