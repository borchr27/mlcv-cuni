
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sklearn.mixture
from matplotlib.colors import LogNorm
from matplotlib import cm

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--seed", default=42, type=int, help="Seed for RNG.")
parser.add_argument("--points", default=3000, type=int, help="Total number of points generated.")

def main(args : argparse.Namespace):
    # Gaussian mixture model exercise.
    # - There is no need to implement anything, except for a visualisation at the end of this function.
    # - The point of this exercise is to investigate the results of GMM computed on a generated set of data.

    # Generate data from 3 gaussians.
    generator = np.random.RandomState(args.seed)
    count = args.points

    mu1 = [0, 0]
    sigma1 = [[1, 0.5], [0.5, 0.5]]
    data1 = generator.multivariate_normal(mu1, sigma1, int(0.35 * count))

    mu2 = [3, 1.5]
    sigma2 = [[0.5, -0.2], [-0.2, 0.5]]
    data2 = generator.multivariate_normal(mu2, sigma2, int(0.15 * count))

    mu3 = [7, 0]
    sigma3 = [[3, 0], [0, 1]]
    data3 =  generator.multivariate_normal(mu3, sigma3, int(0.5 * count))

    data = np.vstack([data1, data2, data3])
    labels = np.hstack([[0] * data1.shape[0], [1] * data2.shape[0], [2] * data3.shape[0]])

    # Let us look at the data distribution in 2D.
    _, ax = plt.subplots(1, 1, figsize=(8, 5), subplot_kw={'aspect': 'equal'})
    ax.set_title("Generated data distribution")
    ax.scatter(data[:, 0], data[:, 1], c=labels)
    plt.tight_layout()
    plt.show()

    # Let us look at the histogram of the data, notice the three peaks in the data.
    hist, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], 50, range=[[-5, 15], [-10, 10]])  
    
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-5, 15)
    ax.set_ylim3d(-10, 10)
    xe = np.repeat(xedges[:-1], yedges.shape[0] - 1)
    ye = np.repeat(yedges[:-1], xedges.shape[0] - 1)
    _xx, _yy = np.meshgrid(xedges[:-1], yedges[:-1])
    xe = _xx.ravel()
    ye = _yy.ravel()
    width = xedges[1] - xedges[0]
    depth = yedges[1] - yedges[0]
    flathist = hist.ravel()
    ax.set_title("Histogram of the data")
    ax.bar3d(xe[flathist > 0], ye[flathist > 0], np.zeros(flathist.shape)[flathist > 0], width, depth, flathist[flathist > 0], shade=True)
    plt.tight_layout()
    plt.show()

    # Use python functions to compute the GMM of three gaussians.
    gmm = sklearn.mixture.GaussianMixture(3, max_iter=500)
    gmm.fit(data)

    # TODO: Compare the computed model with the real distribution defined above.
    print("======================================================")
    print("GMM model with 3 components.")
    print("Weights of the GMM components:")
    print(gmm.weights_)
    print("Means of the GMM components - each row is one mean:")
    print(gmm.means_)
    print("Covariance matrices of the GMM components:")
    for i in range(gmm.covariances_.shape[0]):
        print(gmm.covariances_[i])
    print("Number of iterations until convergence: {}".format(gmm.n_iter_))
    print()

    # Let us consider that we do not know the optimal number of gaussian mixture
    # components. Let's compare distributions with different number of components.
    
    # AIC is an estimator of the relative quality of statistical models for a given dataset.
    aics = np.zeros([5])
    gmm_models = [sklearn.mixture.GaussianMixture(k, max_iter=500) for k in range(1, 6)]
    for k in range(len(gmm_models)):
        gmm_models[k].fit(data)
        aics[k] = gmm_models[k].aic(data)

    minAicIdx = np.argmin(aics)
    gmm_best = gmm_models[minAicIdx]

    # TODO: Compare the computed model with the real distribution defined above.
    print("======================================================")
    print("The best GMM model according to AIC metric.")
    print("- Number of components: {}".format(gmm_best.n_components))
    print("- Weights of the components:")
    print(gmm_best.weights_)
    print("- Means of the components:")
    print(gmm_best.means_)
    print("- Covariances of the components:")
    for i in range(gmm_best.covariances_.shape[0]):
        print(gmm_best.covariances_[i])
    print("Number of iterations until convergence: {}".format(gmm_best.n_iter_))
    print()

    # Let's specify the starting parameters and look at how many iterations it will take
    # to converge.
    start_weights = np.asarray([1/2, 1/4, 1/4])
    start_means = np.asarray([[1, 1], [2, 2], [3, 3]])
    start_sigma = np.zeros([3, 2, 2])
    start_sigma[0, :, :] = np.asarray([[1, 1], [1, 2]])
    start_sigma[1, :, :] = 2 * np.asarray([[1, 1], [1, 2]])
    start_sigma[2, :, :] = 3 * np.asarray([[1, 1], [1, 2]])
    gmm_start = sklearn.mixture.GaussianMixture(
        3, max_iter=500, weights_init=start_weights, means_init=start_means, precisions_init=start_sigma
    )
    gmm_start.fit(data)
    print("======================================================")
    print("GMM with the manually chosen starting parameters.")
    print("Number of iterations until convergence: {}".format(gmm_start.n_iter_))
    print()

    # Let's compare the distributions visually.
    x, y = np.meshgrid(np.linspace(-5, 15), np.linspace(-10, 10))
    xx = np.vstack([x.ravel(), y.ravel()]).T

    fig, ax = plt.subplots(2, 3, figsize=(16, 9), subplot_kw={'aspect': 'equal'})
    ax[0, 0].scatter(data[:, 0], data[:, 1], color=(0.66, 0.66, 0.66), marker='o', s=0.5)
    ax[0, 0].set_xlim([-5, 15])
    ax[0, 0].set_ylim([-10, 10])
    heights = -gmm_best.score_samples(xx) # Returns log-likelihood
    heights = np.reshape(heights, x.shape)
    ax[0, 0].contour(x, y, heights, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 1, 10))
    ax[0, 0].set_title("The best model (AIC)")
    for k in range(1, 6):
        ax[k // 3, k % 3].scatter(data[:, 0], data[:, 1], color=(0.66, 0.66, 0.66), marker='o', s=1)
        ax[k // 3, k % 3].set_xlim([-5, 15])
        ax[k // 3, k % 3].set_ylim([-10, 10])
        heights = -gmm_models[k - 1].score_samples(xx)
        heights = np.reshape(heights, x.shape)
        ax[k // 3, k % 3].contour(x, y, heights, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 1, 10))
        ax[k // 3, k % 3].set_title("Model with {} components".format(k))
    plt.tight_layout()
    plt.show()

    # TODO: Visualise the 3 different models:
    # - Extend the following code section so that you can see all 3 models at the same time (they can
    #   be in separate Axes or windows).
    # gmm
    # gmm_best
    # gmm_start

    # 3D visualisation of the PDF (compare with the histogram).
    plt.figure()
    ax = plt.axes(projection='3d')
    x, y = np.meshgrid(np.linspace(-5, 15, 200), np.linspace(-10, 10, 200))
    # Uncomment for a closer look at the largest peak (less distorted surface).
    #x, y = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-3, 3, 200))
    xx = np.vstack([x.ravel(), y.ravel()]).T
    heights = np.exp(gmm_best.score_samples(xx))
    heights = np.reshape(heights, x.shape)
    ax.set_title("Visualisation of the PDF")
    ax.plot_surface(x, y, heights, cmap=cm.coolwarm)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
