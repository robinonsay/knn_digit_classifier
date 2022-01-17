from typing import Dict, List, TypeVar
from numpy.typing import ArrayLike
import pandas as pd


def import_training_data(training_filename: str) -> Dict:
    training_data = {}
    training_df = pd.read_csv(training_filename)
    for _, img_series in training_df.iterrows():
        key = img_series.get("label")
        img = img_series.drop("label").to_numpy()
        if key in training_data:
            training_data[key].append(img)
        else:
            training_data[key] = [img]
    return training_data


def import_test_data(test_filename: str) -> List:
    test_data = []
    training_df = pd.read_csv(test_filename)
    for _, img_series in training_df.iterrows():
        test_data.append(img_series.to_numpy())
    return test_data


class KDTreeDataWrapper:
    __slots__ = ["key", "value"]

    def __init__(self, key: int, value: ArrayLike) -> None:
        self.key = key
        self.value = value

    def __repr__(self) -> str:
        return f"key: {self.key}, value: {self.value}"

    def __str__(self) -> str:
        return f"key: {self.key}, value: {self.value}"

    def __len__(self):
        return len(self.value)

    def __getitem__(self, item):
        return self.value[item]
