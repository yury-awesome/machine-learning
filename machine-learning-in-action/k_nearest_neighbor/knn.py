import numpy as np


def calculate_distance(input, dataset):
    """
    Calculate the Euclidean distance between each point in group and the input vector.
    """
    diff = np.tile(input, (dataset.shape[0], 1)) - dataset
    return np.sum(diff ** 2, axis=1) ** 0.5


def classify(input, dataset, labels, k):
    """
    Classify the input vector inX using the k-nearest neighbors algorithm.
    """
    sorted_indices = np.argsort(calculate_distance(input, dataset))
    class_count = {}

    for i in range(k):
        label = labels[sorted_indices[i]]
        class_count[label] = class_count.get(label, 0) + 1

    sorted_class_count = sorted(class_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_class_count[0][0]
