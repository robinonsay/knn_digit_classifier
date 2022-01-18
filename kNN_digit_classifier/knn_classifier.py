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

    def __iter__(self):
        return iter(self.value)


class KNNClassifier:
    __slots__ = ["kd_tree", "k"]

    def __init__(self, data: ArrayLike, keys: ArrayLike, k: int = 5):
        if len(data) != len(keys):
            raise ValueError("len(data) != len(keys)")
        data_list = []
        for i in range(len(data)):
            data_list.append(DataWrapper(key=keys[i], value=data[i]))
        self.kd_tree = KDTree(data_list)
        self.k = k

    def predict(self, test_data: ArrayLike):
        predictions = []
        i = 1
        for data_point in test_data:
            print(f"Data Point: {i}", end=" | ")
            k_neighbors = self.kd_tree.kNN_search(data_point, k=self.k)
            classes = {}
            for node in k_neighbors:
                if node.key in classes:
                    classes[node.key] += 1
                else:
                    classes[node.key] = 1
            best_key, best_count = classes.popitem()
            for key, count in classes.items():
                if best_count < count:
                    best_key = key
                    best_count = count
            print(f"Prediction: {best_key} | Count: {best_count}")
            predictions.append(best_key)
            i += 1
        return predictions
