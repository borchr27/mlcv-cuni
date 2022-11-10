
import numpy as np
import matplotlib.pyplot as plt
import typing
import sklearn.metrics

def getConfusionMatrix(classiferData : np.ndarray, trueClass : np.ndarray, threshold : float) -> np.ndarray:
    """
    Returns a confusion matrix computed from classifier data and true classification according to
    the given threshold. The returned matrix is ordered [positive, negative] assuming that class 1
    marks positive examples and class 0 negative ones.

    Arguments:
    - 'classifierData' - 1D vector of probabilities of belonging to class 1.
    - 'trueClass' - 1D vector of gold labels (true classes of the data).
    - 'threshold' - The threshold for class prediction computation.

    Returns:
    - A square confusion matrix denoting the number of correctly/incorrectly classified samples.
    """
    
    # Compute classification for the selected threshold.
    classes = np.asarray(classiferData >= threshold, np.int32)

    # Compute the confusion matrix for the chosen classifier.
    # We want to order the items in the matrix as [positive, negative] hence the argument label=[1, 0].
    confusionMatrix = sklearn.metrics.confusion_matrix(trueClass, classes, labels=[1, 0])

    return confusionMatrix

def displayMatrix(axes : plt.Axes, matrix : np.ndarray, logColor : bool = False) -> None:
    # Assumes ordering [positive, negative] in the matrix
    axes.matshow(np.log(matrix + np.spacing(1)) if logColor else matrix, cmap=plt.cm.Blues, alpha=0.5)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axes.text(x=j, y=i, s=matrix[i, j], va='center', ha='center')
    axes.set_xticks([0, 1])
    axes.set_yticks([0, 1])
    axes.set_xticklabels(["positive", "negative"])
    axes.set_yticklabels(["positive", "negative"], rotation=90)
    axes.set_xlabel("Predicted")
    axes.set_ylabel("True")
    axes.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)
    axes.xaxis.set_label_position("top")

def drawMatrices(matrices : typing.Union[np.ndarray, typing.Sequence[np.ndarray]], logColor : bool = False) -> None:
    """
    Draws one or more matrices in a single row of matplotlib subplots. The matrix cells can be
    coloured logarithmically to highlight smaller differences.

    Arguments:
    - 'matrices' - One np.ndarray matrix or a list of them intended for visualisation.
    - 'logColor' - Whether to assign colours according to the logarithm of the matrix values.
    """
    isList = isinstance(matrices, typing.Sequence)
    count = len(matrices) if isList else 1 
    _, ax = plt.subplots(1, count, figsize=(count * 6, 7), subplot_kw={'aspect': 'auto'})
    if count > 1:
        for i in range(len(matrices)):
            displayMatrix(ax[i], matrices[i], logColor)
    else:
        displayMatrix(ax, matrices[0] if isList else matrices, logColor)
    plt.tight_layout()
    plt.show()

def setAxes(axes : plt.Axes, title : str, xLabel : str, yLabel : str):
    axes.set_title(title)
    axes.set_xlabel(xLabel)
    axes.set_ylabel(yLabel)
    axes.legend()

def drawAPPlot(axes : plt.Axes, precisions : np.ndarray, apPrecisions : np.ndarray, recalls : np.ndarray, apValue : float, apName : str):
    axes.step(recalls, apPrecisions, where='post', label="AP")
    axes.plot(recalls, precisions, label="PR")
    #axes.scatter(recalls, precisions, s=3)
    #axes.scatter(recalls, apPrecisions, s=3)
    axes.set_title("{}: {:.6f}".format(apName, apValue))
    axes.set_xlabel("Recall")
    axes.set_ylabel("Precision")
    axes.legend()
