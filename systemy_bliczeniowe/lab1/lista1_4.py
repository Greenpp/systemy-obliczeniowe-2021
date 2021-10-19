# %%
import itertools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 42
DATA_PATH = Path('data')
SPLITS = 5
REPEATS = 2
N_DATASETS = 28

CLASSIFIERS = {
    'linear': RidgeClassifier(random_state=RANDOM_SEED),
    'svm': SVC(random_state=RANDOM_SEED),
    'tree': DecisionTreeClassifier(random_state=RANDOM_SEED),
}


def run_experiment(classifiers: dict, dataset: np.ndarray, data_i: int) -> np.ndarray:
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    scores = np.zeros((len(CLASSIFIERS), N_DATASETS, SPLITS * REPEATS))

    for fold_i, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        for clf_id, clf_name in enumerate(classifiers):
            clf = clone(classifiers[clf_name])
            clf.fit(X[train_idx], y[train_idx])
            y_pred = clf.predict(X[test_idx])
            scores[clf_id, data_i, fold_i] = balanced_accuracy_score(
                y[test_idx], y_pred
            )

    return scores


def load_datasets() -> list[np.ndarray]:
    data_files = DATA_PATH.glob('*.csv')

    datasets = []
    for file in data_files:
        df = pd.read_csv(file)

        datasets.append(df.to_numpy().astype(float))

    return datasets


if __name__ == '__main__':
    d = load_datasets()

    kfold = RepeatedKFold(n_splits=SPLITS, n_repeats=REPEATS, random_state=RANDOM_SEED)

    with ThreadPoolExecutor() as exec:
        scores = exec.map(run_experiment, itertools.repeat(CLASSIFIERS), d, range(len(d)))
# %%
    np.sum(scores)

# %%
