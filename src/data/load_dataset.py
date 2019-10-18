from os.path import dirname, join
from pathlib import Path

import numpy as np
import pandas as pd
from graspy.utils import pass_to_ranks, symmetrize, import_edgelist

MODULE_PATH = dirname(__file__)

def load_COBRE(ptr=None):
    # Load data and wrangle it
    path = Path(MODULE_PATH).parents[1] / "data/raw/COBRE.npz"

    X, y = _load_dataset(path=path, n_nodes=263, ptr=ptr)
    return X, y


def load_UMich(ptr=None):
    path = Path(MODULE_PATH).parents[1] / "data/raw/UMich.npz"

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


def load_HNU1(ptr=None, return_subid=False):
    path = Path(MODULE_PATH).parents[1] / "data/raw/HNU1/"

    f = sorted(path.glob('*.ssv'))
    df = pd.read_csv(path / "HNU1.csv")
    subid = [int(fname.stem.split('_')[0].split('-')[1][-5:]) for fname in f]

    y = np.array([np.unique(df.SEX[df.SUBID == s])[0] - 1 for s in subid])
    g = np.array(import_edgelist(f, 'ssv'))

    if ptr is not None:
        g = np.array([pass_to_ranks(x) for x in g])

    if return_subid:
        return g, y, subid
    else:
        return g, y

def load_SWU4(ptr=None, return_subid=False):
    path = Path(MODULE_PATH).parents[1] / "data/raw/SWU4/"

    f = sorted(path.glob('*.ssv'))
    df = pd.read_csv(path / "SWU4.csv")
    subid = [int(fname.stem.split('_')[0].split('-')[1][-5:]) for fname in f]

    y = np.array([np.unique(df.SEX[(df.SUBID == s) & (df.SESSION == 'Baseline')])[0] for s in subid]).astype(int)
    y -= 1
    g = np.array(import_edgelist(f, 'ssv'))

    if ptr is not None:
        g = np.array([pass_to_ranks(x) for x in g])

    if return_subid:
        return g, y, subid
    else:
        return g, y


def load_BNU1(ptr=None):
    path = Path(MODULE_PATH).parents[1] / "data/raw/BNU1/"

    f = sorted(path.glob('*.ssv'))
    df = pd.read_csv(path / "BNU1_phenotypic_data.csv")
    subid = [int(fname.stem.split('_')[0].split('-')[1][-5:]) for fname in f]

    y = np.array([np.unique(df.SEX[(df.SUBID == s) & (df.SESSION == 'Baseline')])[0] for s in subid]).astype(int)
    y -= 1
    g = np.array(import_edgelist(f, 'ssv'))

    if ptr is not None:
        g = np.array([pass_to_ranks(x) for x in g])

    return g, y
