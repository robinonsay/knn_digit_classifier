import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--training_csv",
                        default="value/train.csv",
                        help="CSV training value to use with kNN classifier")
    parser.add_argument("--test_csv",
                        default="value/test.csv",
                        help="CSV test value to use with kNN classifier")
    args = parser.parse_args()
