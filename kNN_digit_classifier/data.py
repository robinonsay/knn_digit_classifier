from typing import Dict, List
from kNN_digit_classifier import const
import numpy
import pandas as pd


def import_training_data(training_filename: str) -> Dict:
    training_data = {}
    training_df = pd.read_csv(training_filename)
    for _, img_series in training_df.iterrows():
        key = img_series.get("label")
        img_1d = img_series.drop("label").to_numpy()
        img = numpy.reshape(img_1d, (const.IMAGE_HEIGHT, const.IMAGE_WIDTH))
        if key in training_data:
            training_data[key].append(img)
        else:
            training_data[key] = [img]
    return training_data


def import_test_data(test_filename: str) -> List:
    training_data = []
    training_df = pd.read_csv(test_filename)
    for _, img_series in training_df.iterrows():
        img = numpy.reshape(img_series.to_numpy(), (const.IMAGE_HEIGHT, const.IMAGE_WIDTH))
        training_data.append(img)
    return training_data
