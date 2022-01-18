import argparse
from typing import Dict, List, Tuple
from kNN_digit_classifier import data
from kNN_digit_classifier.data import KDTreeDataWrapper
from kNN_digit_classifier.kd_tree import KDTree, KDNode


def train(data_dict: Dict) -> KDTree:
    data_list = []
    for key, values in data_dict.items():
        for val in values:
            data_list.append(KDTreeDataWrapper(key, val))
    return KDTree(data_list)


def classify(k_nn: List[KDNode]) -> Tuple[int, int]:
    class_dict = {}
    for wrapped_data in k_nn:
        point = wrapped_data.value
        if point.key not in class_dict:
            class_dict[point.key] = 1
        else:
            class_dict[point.key] += 1
    pred = class_dict.popitem()
    for key, cnt in class_dict.items():
        if cnt > pred[1]:
            pred = (key, cnt)
    return pred


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
    kd_tree = train(training_data)
    test_data = data.import_test_data(args.test_csv)
    print("Making Predictions...")
    print("ImageID, Prediction, Count")
    with open("predictions.txt", "w") as pred_file:
        i = 1
        for data_point in test_data:
            k_nn = kd_tree.kNN_search(data_point)
            prediction, count = classify(k_nn)
            pred_file.write(f"{i}, {prediction}")
            print(f"{i}, {prediction}, {count}")
            i += 1

