
import warnings
import numpy as np
from scipy import stats

def corrcoef(labels : np.ndarray, X : np.ndarray) ->np.ndarray:
    """
    Computes the correlation coefficients for class-feature and feature-feature pairs.
    It is a less general version of Matlab's corrcoef function.
    """
    matSize = X.shape[1] + 1 if len(X.shape) > 1 else 2
    coefmat = np.zeros([matSize, matSize])
    datasets = np.vstack([labels.T, X.T]).T
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(datasets.shape[1]):
            for j in range(datasets.shape[1]):
                corr, _ = stats.pearsonr(datasets[:, i], datasets[:, j])
                coefmat[i, j] = corr
    return coefmat

def parseTextFile(fileName : str) -> np.ndarray:
    """
    Parses a text file containg 2D table of integers separated by whitespaces and returns
    a numpy array containing the table.
    """
    with open(fileName) as f:
        lines = f.readlines()
        data = [line.split() for line in lines]
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = int(data[i][j])
    return np.asarray(data)
