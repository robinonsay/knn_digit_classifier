import argparse
import time
from typing import Dict, List, Tuple

from sklearn.neighbors import KNeighborsClassifier

from kNN_digit_classifier import data
from kNN_digit_classifier.data import KDTreeDataWrapper
from kNN_digit_classifier.kd_tree import KDTree, KDNode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--training_csv",
                        default="data/train.csv",
                        help="CSV training value to use with kNN classifier")
    parser.add_argument("--test_csv",
                        default="data/test.csv",
                        help="CSV test value to use with kNN classifier")
    args = parser.parse_args()
    print("Training...")
    training_data = data.import_training_data(args.training_csv)
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(*training_data)
    test_data = data.import_test_data(args.test_csv)
    print("Making Predictions...")
    predictions = knn_classifier.predict(test_data)
    with open("predictions.csv", "w") as pred_file:
        pred_file.write("ImageID,Label\n")
        i = 1
        for prediction in predictions:
            pred_file.write(f"{i},{prediction}\n")
            i += 1
    print("Done")
