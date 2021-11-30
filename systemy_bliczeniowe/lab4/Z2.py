# https://www.kth.se/blogs/pdc/2019/08/parallel-programming-in-python-mpi4py-part-1/
# https://www.kth.se/blogs/pdc/2019/11/parallel-programming-in-python-mpi4py-part-2/

import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpi4py import MPI
from scipy.spatial.distance import cdist
from sklearn.datasets import make_classification

N_SAMPLES = 1000
N_CLUSTERS = 3
RND_SEED = 42

sns.set()
random.seed(RND_SEED)

comm = MPI.COMM_WORLD

proc_num = comm.Get_rank()
total_proc = comm.Get_size()


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

    return np.array(centroids)


def plot(data, centroids, classes):
    centroids = np.array(centroids)
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=classes)
    sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], color='red')
    plt.show()
    plt.clf()


if proc_num == 0:
    data = generate_data()
    centroids = init_centroids(data)

    ave, res = divmod(data.shape[0], total_proc)
    counts = [ave + 1 if p < res else ave for p in range(total_proc)]

    starts = [sum(counts[:p]) for p in range(total_proc)]
    ends = [sum(counts[: p + 1]) for p in range(total_proc)]

    split_data = [data[starts[p] : ends[p]] for p in range(total_proc)]
else:
    split_data = None
    centroids = None
    done = None


done = False
split_data = comm.scatter(split_data, root=0)
while not done:
    centroids = comm.bcast(centroids, root=0)

    new_classes = get_classes(split_data, centroids)
    new_centroids = update_centroids(split_data, new_classes)
    new_centroids = comm.gather(new_centroids, root=0)
    if proc_num == 0:
        new_centroids = np.array(new_centroids)
        new_centroids = new_centroids.mean(axis=0)

        if not np.array_equal(centroids, new_centroids):
            centroids = new_centroids
        else:
            done = True

    done = comm.bcast(done, root=0)

classes = comm.gather(new_classes, root=0)
if proc_num == 0:
    classes = np.concatenate(tuple(classes))
    plot(data, centroids, classes)
