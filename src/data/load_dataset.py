from os.path import dirname, join
from pathlib import Path

import numpy as np
from graspy.utils import pass_to_ranks, symmetrize


def load_COBRE(ptr=None):
    # Load data and wrangle it
    module_path = dirname(__file__)
    path = Path(module_path).parents[1] / "data/raw/COBRE.npz"

    X, y = _load_dataset(path=path, n_nodes=263, ptr=ptr)
    return X, y


def load_UMich(ptr=None):
    module_path = dirname(__file__)
    path = Path(module_path).parents[1] / "data/raw/UMich.npz"

    X, y = _load_dataset(path=path, n_nodes=264, ptr=ptr)
    return X, y


def _load_dataset(path, n_nodes, ptr=None):
    file = np.load(path)
    X = file["X"]
    y = file["y"].astype(int)

    n_samples = X.shape[0]

    y[y == -1] = 0

    idx = np.triu_indices(n_nodes, k=1)

    X_graphs = np.zeros((n_samples, n_nodes, n_nodes))

    for i, x in enumerate(X):
        X_graphs[i][idx] = x
        X_graphs[i] = symmetrize(X_graphs[i], "triu")

    if ptr is not None:
        X_graphs = X_graphs - X_graphs.min(axis=(1, 2)).reshape(-1, 1, 1)

        for i, x in enumerate(X_graphs):
            X_graphs[i] = pass_to_ranks(X_graphs[i])

    return X_graphs, y
