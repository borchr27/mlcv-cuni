
import argparse
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import sklearn.decomposition
import lab03a_pca_eigen
import lab03b_pca_svd

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--task", default="scree", type=str, help="Task to execute: 'scree' or 'pcselection'.")

def screeGraph() -> None:
    """This is the main task of this exercise."""
    # Load data
    hald = scipy.io.loadmat("hald.mat")
    data = hald["ingredients"]

    # Identify the number of 'highly' contributing components from a scree graph.
    # Use all 3 methods and compare their results.
    #
    # Plot the scree graph of explained variances
    # Plot the graph of cumulative explained variances (numpy.cumsum
    # might be useful in this regard)
    #
    # TODO: Uncomment the lines starting with 'ax.plot' and replace
    # underscores '_' by appropriate arrays.

    # Manual PCA
    B, D, mu, projected = lab03a_pca_eigen.eigPCA(data)

    _, ax = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={'aspect': 'auto'})    
    ax[0].set_title("Manual PCA (eigen decomposition)")
    ax[0].axhline(color='black', ls='--')
    ax[0].axvline(1.0, color='black', ls='--')
    ax[0].xaxis.get_major_locator().set_params(integer=True)
    #ax[0].plot(range(1, D.size + 1), _, color='red', marker='o', label="Explained") # Explained
    #ax[0].plot(range(1, D.size + 1), _, color='blue', marker='o', label="Cumulative") # Cumulative
    ax[0].legend()

    # Scikit-learn PCA
    pca = sklearn.decomposition.PCA(n_components=None)
    pca.fit(data)

    ax[1].set_title("Scikit-learn PCA")
    ax[1].axhline(color='black', ls='--')
    ax[1].axvline(1.0, color='black', ls='--')
    ax[1].xaxis.get_major_locator().set_params(integer=True)
    #ax[1].plot(range(1, D.size + 1), _, color='red', marker='o', label="Explained") # Explained
    #ax[1].plot(range(1, D.size + 1), _, color='blue', marker='o', label="Cumulative") # Cumulative
    ax[1].legend()

    # SVD PCA
    V, explained, mu, projected = lab03b_pca_svd.svdPCA(data)

    ax[2].set_title("SVD PCA")
    ax[2].axhline(color='black', ls='--')
    ax[2].axvline(1.0, color='black', ls='--')
    ax[2].xaxis.get_major_locator().set_params(integer=True)
    #ax[2].plot(range(1, D.size + 1), _, color='red', marker='o', label="Explained") # Explained
    #ax[2].plot(range(1, D.size + 1), _, color='blue', marker='o', label="Cumulative") # Cumulative
    ax[2].legend()
    plt.tight_layout()
    plt.show()

def pcSelection() -> None:
    """This is a secondary task of this exercise."""
    # TODO: Find the number of 'highly' contributing PCs, use PCA implemented
    # in scikit-learn (sklearn.decomposition.PCA)
    # - The threshold is generally 95% of explained variance.
    # Use the following data:
    ovariancancer = scipy.io.loadmat("ovariancancer.mat")
    data = ovariancancer["obs"]


def main(args : argparse.Namespace) -> None:
    # Select task to execute.
    if args.task == "scree":
        screeGraph()
    elif args.task == "pcselection":
        pcSelection()
    else:
        raise ValueError("Unrecognised task: '{}'.".format(args.task))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
