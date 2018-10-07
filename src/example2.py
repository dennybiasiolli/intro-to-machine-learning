from functools import reduce
import math
import numpy as np


class MyGaussianNB():

    def __init__(self):
        self.known_values = []
        self.known_labels = []

    def fit(self, values, labels):
        if len(values) != len(labels):
            raise Exception(
                'values count has a differeNt length compared to labels count')
        for v in values:
            self.known_values.append(v)
        for l in labels:
            self.known_labels.append(l)
        return

    def __get_distances__(self, value):

        def obtain_distance(known_value):
            if len(value) != len(known_value):
                raise Exception(
                    'value has a different dimension compared to known_value')
            distances = (map(
                lambda kv, v:
                    v - kv, value, known_value
            ))
            return math.sqrt(
                reduce(
                    lambda acc, elem: acc + (elem**2), distances, 0
                )
            )

        return obtain_distance

    def predict_value(self, value):
        distances = map(self.__get_distances__(value), self.known_values)
        labeled_distances = map(lambda d, l: (
            d, l), distances, self.known_labels)
        return sorted(labeled_distances, key=lambda i: i[0])[0][1]


X = [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]
Y = [1, 1, 1, 2, 2, 2]

clf = MyGaussianNB()
clf.fit(X, Y)
print(clf.predict_value([-0.8, -1]))
print(clf.predict_value([2, 2]))


clf2 = MyGaussianNB()
clf2.fit([
    [-7, -7, -7, -7],
    [-6, -6, -6, -6],
    [-5, -5, -5, -5],
    [-4, -4, -4, -4],
    [-3, -3, -3, -3],
    [-2, -2, -2, -2],
    [-1, -1, -1, -1],
    [0, 0, 0, 0],
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3],
    [4, 4, 4, 4],
    [5, 5, 5, 5],
    [6, 6, 6, 6],
    [7, 7, 7, 7],
], [
    'antipatico',
    'antipatico',
    'antipatico',
    'antipatico',
    'antipatico',
    'simpatico',
    'simpatico',
    'simpatico',
    'simpatico',
    'simpatico',
    'simpatico',
    'antipatico',
    'antipatico',
    'antipatico',
    'antipatico',
])
print(clf2.predict_value([-2, -3, 0, -3]))
