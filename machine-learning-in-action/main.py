import numpy as np

from k_nearest_neighbor.dataset import load
from k_nearest_neighbor.knn import classify


def main():
    group, labels = load()

    print("Group:\n", group)
    print("Labels:\n", labels)
    print("Classifying input [0., 0.]: ", classify(np.array([0., 0.]), group, labels, 3))


if __name__ == "__main__":
    main()
