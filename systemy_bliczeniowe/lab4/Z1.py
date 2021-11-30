# https://www.kth.se/blogs/pdc/2019/08/parallel-programming-in-python-mpi4py-part-1/
# https://www.kth.se/blogs/pdc/2019/11/parallel-programming-in-python-mpi4py-part-2/

import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.datasets import make_classification

N_SAMPLES = 1000
N_CLUSTERS = 3
RND_SEED = 42

sns.set()


def generate_data():
    data, _ = make_classification(
        N_SAMPLES, n_features=2, n_redundant=0, random_state=RND_SEED
    )

    return data


def init_centroids(data):
    idx = random.sample(range(N_SAMPLES), N_CLUSTERS)

    return data[idx]


def get_classes(data, centroids):
    dist = cdist(data, centroids)
    classes = np.argmin(dist, axis=1)

    return classes


def update_centroids(data, classes):
    centroids = []
    for c in range(N_CLUSTERS):
        class_data = data[classes == c]
        new_centroid = class_data.mean(axis=0)
        centroids.append(new_centroid)

    return centroids


def plot(data, centroids, classes):
    centroids = np.array(centroids)
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=classes)
    sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], color='red')
    plt.show()
    plt.clf()


data = generate_data()
centroids = init_centroids(data)
classes = get_classes(data, centroids)
done = False

while not done:
    centroids = update_centroids(data, classes)
    new_classes = get_classes(data, centroids)

    if any(classes != new_classes):
        classes = new_classes
    else:
        done = True

plot(data, centroids, classes)


# %%
