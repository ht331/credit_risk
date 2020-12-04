
import pandas as pd
import numpy as np
import json


def load_feature_group():
    with open('feature_group.json', 'r') as f:
        a = f.readline()
    return json.loads(a)


class WoeFeatures:

    def __init__(self):
        pass

    def fit(self):
        pass

    def fit_transform(self):
        pass

    def transform(self):
        pass


def woe_train(feature_group_dict):
    pass

