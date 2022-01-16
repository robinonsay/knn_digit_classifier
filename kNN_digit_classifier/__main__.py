import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--training_csv",
                        default="data/train.csv",
                        help="CSV training data to use with kNN classifier")
    parser.add_argument("--test_csv",
                        default="data/test.csv",
                        help="CSV test data to use with kNN classifier")
    args = parser.parse_args()
