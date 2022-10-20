
import argparse
import numpy as np
import lab03_help

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--seed", default=42, type=int, help="Seed for RNG.")
parser.add_argument("--points", default=100, type=int, help="Number of points to generate.")
parser.add_argument("--no_draw_generated", default=False, action="store_true", help="Stops drawing the generated points.")


def eigPCA(data : np.ndarray) -> tuple:
    """
    This function is our manual implementation of the principal component analysis.
    It can be used to find the components from a 2D matrix of observations x features.

    Arguments:
    - 'data' - NxD data matrix representing N observations with D features.

    Returns:
    - Ordered principal components.
    - Explained variances.
    - The mean of the data.
    - Projection of the data onto the computed PCs.
    """

    # -> PCA step 1

    # TODO: Make sure that the data has the shape NxD where N is the number
    # of observations and D is the number of features (dimensionality of the data).
    # The data passed to this function should be of that shape already.
    # Store the NxD data matrix in the variable 'X'.
    X = None

    # TODO: Compute the mean value for each feature and store it in
    # the variable 'mu' (should be a vector).
    mu = None

    # TODO: Centre the matrix X (Centering means subtracting mean).

    # TODO: Compute the covariance matrix and name it 'sig'.
    sig = None

    # -> PCA step 2

    # Compute eigenvectors V and eigenvalues D (let's use numpy.linalg.eig).
    D, V = np.linalg.eig(sig)
    # Sort the eigenvalues in descending order.
    descOrder = np.argsort(-D)
    D = D[descOrder]
    # Create the basis vector matrix 'B', note that principal components are in the columns of V.
    B = V[:, descOrder]

    # -> PCA step 3

    # TODO: Project the data onto PCs (principal components), name the variable 'projected'.
    projected = None

    # TODO: Return the basis vectors, explained variances(eigenvalues), the mean
    # and the projected data.
    return B, D, mu, projected

def main(args : argparse.Namespace) -> None:
    # Random number generator for reproducibility.
    generator = np.random.RandomState(args.seed)

    # Prepare 2D data
    # Data of class 1.
    N1 = args.points
    mu1 = [1, 1]
    sigma1 = [[1, 1.5], [1.5, 5]]
    dat1 = generator.multivariate_normal(mu1, sigma1, N1)

    # Data of class 2.
    N2 = args.points
    mu2 = [4, 6]
    sigma2 = [[1, 0], [0, 1]]
    dat2 = generator.multivariate_normal(mu2, sigma2, N2)

    # Combined data.
    data = np.vstack([dat1, dat2])
    labels = np.hstack([np.ones(N1), 2 * np.ones(N2)])

    # Display our dataset.
    if not args.no_draw_generated:
        lab03_help.plotData(data, labels)

    # Let us use our implementation of PCA.
    # TODO: Implement the body of eigPCA at the top of the script.
    B, _, mu, projected = eigPCA(data)

    # Plot the new coordinate axes (principal components).
    lab03_help.plotDataWithPCs(data, labels, mu, B)

    # Look at histograms of projections onto PC1 and PC2.
    lab03_help.plotHistograms(projected, labels)

    # ===== Task =====
    # TODO: Find the principal components when dat2 is generated using the following
    # parameters.
    # - Change the values at the beginning of the exercise and run the script again.
    mu2_alt = [4, 1]
    sigma2_alt = [[1, 1.5], [1.5, 3]]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
