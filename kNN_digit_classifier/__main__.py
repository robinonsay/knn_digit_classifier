import argparse
from kNN_digit_classifier import data
from kNN_digit_classifier.knn_classifier import KNNClassifier


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
    test_data = data.import_test_data(args.test_csv)
    knn_classifier = KNNClassifier(*training_data, k=10)
    print("Making Predictions...")
    predictions = knn_classifier.predict(test_data[:5])
    with open("predictions.csv", "w") as pred_file:
        pred_file.write("ImageID,Label\n")
        i = 1
        for prediction in predictions:
            pred_file.write(f"{i},{prediction}\n")
            i += 1
    print("Done")
