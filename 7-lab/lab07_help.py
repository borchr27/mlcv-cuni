
import numpy as np
from typing import List, Sequence, Tuple

def loadStringData(fileName : str) -> List[List[str]]:
    with open(fileName) as f:
        lines = f.readlines()
        data = [line.split(",") for line in lines]
    return data

def stringCategoriesToInt(stringData : Sequence[Sequence[str]]) -> Tuple[np.ndarray, List[np.ndarray]]:
    stringData = np.asarray(stringData)
    uniqueLines = []
    valueNames = []
    for i in range(stringData.shape[1]):
        values, indices = np.unique(stringData[:, i], return_inverse=True)
        uniqueLines.append(indices)
        valueNames.append(values)
    data = np.vstack(uniqueLines).T
    
    return data, valueNames

def loadShrooms(fileName : str) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
    stringData = loadStringData(fileName)
    data, names = stringCategoriesToInt(stringData)
    featureNames = [
        "Cap shape", "Cap surface", "Cap color", "Has bruises", "Odor", "Gill attachment", "Gill spacing",
        "Gill size", "Gill color", "Stalk shape", "Stalk root", "Stalk surf. above ring", "Stalk surf. below ring",
        "Stalk col. above ring", "Stalk col. below ring", "Veil type", "Veil color", "Ring number", "Ring type",
        "Spore print color", "Population", "Habitat"
    ]
    return data, names, featureNames

def labelSummary(predictions : np.ndarray, labels : np.ndarray, names : Sequence[np.ndarray]) -> None:
    misclassified = predictions != labels
    print("Misclassification error (Error rate): {:.4f}".format(np.sum(misclassified) / labels.shape[0]))
    # NOTE: 'names[0]' are the class names: EDIBLE, POISONOUS.
    classes, counts = np.unique(names[0][labels[misclassified]], return_counts=True)
    for cls, count in zip(classes, counts):
        print("The class \"{}\" was misclassified {} times.".format(cls, count))

def generateData(generator : np.random.RandomState, N : int, mus : Sequence[Sequence[float]], sigmas : Sequence[np.ndarray], labelMultipliers : Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    if len(mus) != len(sigmas):
        raise ValueError("Data generation received different number of means than covariance matrices!")
    data = []
    labels = []
    for mu, sigma, labelMul in zip(mus, sigmas, labelMultipliers):
        data.append(generator.multivariate_normal(mu, sigma, size=N))
        labels.append(np.ones([N]) * labelMul)
    
    return np.vstack(data), np.hstack(labels)
