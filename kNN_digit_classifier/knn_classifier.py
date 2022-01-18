from numpy.typing import ArrayLike
from kNN_digit_classifier.kd_tree import KDTree


class DataWrapper:
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


class KNNClassifier:
    __slots__ = ["kd_tree"]

    def __init__(self, data: ArrayLike, keys: ArrayLike, k: int = 5):
        if len(data) != len(keys):
            raise ValueError("len(data) != len(keys)")
        data_list = []
        for i in range(len(data)):
            data_list.append(DataWrapper(key=keys[i], value=data[i]))
        self.kd_tree = KDTree(data_list)

    def predict(self, test_data: ArrayLike):
        pass
