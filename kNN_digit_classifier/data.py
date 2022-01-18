from typing import Dict, List, TypeVar, Tuple
from numpy.typing import ArrayLike
import pandas as pd


def import_training_data(training_filename: str) -> Tuple[ArrayLike, ArrayLike]:
    data = []
    keys = []
    training_df = pd.read_csv(training_filename)
    for _, img_series in training_df.iterrows():
        key = img_series.get("label")
        img = img_series.drop("label").to_numpy()
        data.append(img)
        keys.append(key)
    return data, keys


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
