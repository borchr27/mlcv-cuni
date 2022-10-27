
import argparse
import scipy.io
import numpy as np
import sklearn.decomposition
import lab03a_pca_eigen

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--data", default="hald", type=str, help="Name of the dataset to use: 'hald' or 'cancer'.")


def svdPCA(data : np.ndarray) -> tuple:
    """
    Computes the singular value decomposition on data such that the result
    represents the components and explained variances of PCA.

    Arguments:
    - 'data' - NxD data matrix where N is the number of observation and D is the number of features.

    Returns:
    - Computed principal components.
    - Explained variances.
    - The mean of the data.
    - Projection of the data onto the computed PCs.
    """

    # TODO: Compute the mean for each feature.
    mu = None

    # TODO: Construct the matrix Y from the lecture.
    Y = None

    # TODO: Compute SVD of Y.
    # - The right eigenvectors are situated in rows.
    U, S, V = np.linalg.svd(Y, full_matrices=False)

    # TODO: Project the data onto PCs.
    projected = None

    # TODO: Return the matrix of principal components, the explained variances,
    # the mean of the data and the projection of the data.
    return None, None, None, None

def main(args : argparse.Namespace) -> None:
    if args.data == "hald":
        hald = scipy.io.loadmat("hald.mat")
        data = hald["ingredients"]
    elif args.data == "cancer":
        # TODO: Compute the time for each method using a different dataset
        # - This dataset is much larger than the first so it will be slower
        #   to compute.
        # - NOTE: There are more dimensions in the data than there are examples,
        #   therefore we can get only that many principal components - this results
        #   in 'random' vectors and angles in the last dimensions.
        ovariancancer = scipy.io.loadmat("ovariancancer.mat")
        data = ovariancancer["obs"]

    # TODO: ===== Task =====
    # Measure the time of each PCA computation method
    # - You can use package 'time' or 'timeit'.
    # First, make sure that the methods do what you want, only then start
    # measuring the time.

    # (EIG) TODO: PCA using eigenvalues of covariance matrix
    # - Take your solution to the exercise in lab03a (eigPCA function)
    B, _, _, _ = lab03a_pca_eigen.eigPCA(data)

    # (SVD) TODO: Implement the body of the function 'svdPCA' at the top of the script.
    # - The principal components produced by np.linalg.svd are in rows.
    V, _, _, _ = svdPCA(data)

    # (PCA) TODO: Analyse the scikit-learn PCA method
    # - The computed principal components are in rows
    # - n_components=None means that we want all of the components
    pca = sklearn.decomposition.PCA(n_components=None)
    pca.fit(data)
    coef = pca.components_
    score = pca.transform(data)

    # TODO: Try to figure out what the attributes 'pca.components_',
    # 'pca.explained_variance_' and the variable 'score' represent.
    # - look at (and compare with) the outputs of our eigPCA function.


    # Compare the results

    # TODO: Find the mutual angles between vectors B, V and coef.
    # How is the scalar product of vectors related to their angle?
    # - You should know this from linear algebra or any CG/geometry course.
    with np.printoptions(precision=3, suppress=True):
        # TODO: Uncomment the print lines and replace underscores '_' with
        # the appropriate matrices.
        print("Angles between B and V")
        #print(np.abs(np.arccos(np.clip(_ @ _, -1, 1))) / np.pi * 180)
        print("Angles between V and coef")
        #print(np.abs(np.arccos(np.clip(_ @ _, -1, 1))) / np.pi * 180)
        print("Angles between B and coef")
        #print(np.abs(np.arccos(np.clip(_ @ _, -1, 1))) / np.pi * 180)

    pass

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
