
import os
import re
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import lab03b_pca_svd

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--path", default="./centered/", type=str, help="Path to the 'face' dataset with root directory 'centered'.")
parser.add_argument("--compare_idx", default=22, type=int, help="Index of the iamge selected for comparison.")
parser.add_argument("--reconstruct_idx", default=11, type=int, help="Index of the image selected for gradual reconstruction.")

def main(args : argparse.Namespace) -> None:
    # TODO: Make sure that 'args.path' contains a correct path towards folder 'centered' containing
    # face dataset downloaded from moodle.
    trainingFolder = args.path
    regex = re.compile('.*\..*\.pgm$')
    trainFiles = []
    dirs = os.listdir(args.path)
    for f in dirs:
        if regex.match(f):
            trainFiles.append(f)
    trainFiles = [trainingFolder + f for f in trainFiles]

    # Get the resolution of the images (from the first one).
    with Image.open(trainFiles[0]) as imHandle:
        im = np.asarray(imHandle)
    H = im.shape[0]
    W = im.shape[1]
    # How many images are there?
    M = len(trainFiles)
    # Number of classes
    C = 15

    # Read all images and transform them to vectors (create a matrix and name it X).
    flattenedImgs = np.zeros([M, H * W])
    for i in range(M):
        with Image.open(trainFiles[i]) as imHandle:
            flattenedImgs[i] = np.asarray(imHandle).flatten()

    # TODO: Compute eigenvectors forming the matrix B using SVD.
    # - you can use either 'svdPCA' which you implemented in 'lab03b_pca_svd.py'
    #   or compute it again using 'np.linalg.svd'
    #
    # Name the variables as follows (so that the rest of the code works):
    # mx - the mean of the data
    # B - the eigenvectors in row form (1 row = 1 eigenvector)
    # X - the centred data (centred 'flattenedImgs')
    # S - explained variances

    B = None
    S = None
    mx = None
    X = None

    # TODO: Show one of the eigenfaces
    pc = 0
    plt.figure()
    plt.title("PC {} reshaped back into an image".format(pc))
    plt.imshow(np.reshape(B[pc, :], [H, W]), cmap='Greys_r')
    plt.tight_layout()
    plt.show()

    # TODO: Determine the number of PCs to keep, look at the cumulative variances.
    # - Either plot the graph and read the value from there or use an automatic method.
    k = 1
    
    # Keep k principal components.
    Bk = B[:k, :]

    # Compute the new coordinates in k dimensions.
    Xk = X @ Bk.T

    # Reconstruct the reduced data back to D dimensions.
    rec = mx + Xk @ Bk

    # Compare the original image and the reconstruction from principal components.
    idx = args.compare_idx
    _, ax = plt.subplots(1, 2, subplot_kw={'aspect': 'equal'})
    ax[0].set_title("Original image")
    ax[0].imshow(np.reshape(flattenedImgs[idx, :], [H, W]), cmap='Greys_r')
    ax[1].set_title("Reconstruction from {} PCs".format(k))
    ax[1].imshow(np.reshape(rec[idx, :], [H, W]), cmap='Greys_r')
    plt.tight_layout()
    plt.show()

    # Now, let's visualise the reconstructions from different number of principal components.
    # You can change the value of 'imgIdx' to view the reconstruction of other images.
    # Also, you can modify the 'counts' list to see the reconstructions which interest
    # you.
    imgIdx = args.reconstruct_idx
    counts = [1, 2, 5, 10, 20, 40, 80, 120]
    fig, ax = plt.subplots(1, len(counts) + 1, figsize=(20, 3), subplot_kw={'aspect': 'equal'})
    for i in range(len(counts)):
        count = counts[i]
        projected = X[imgIdx, :] @ B[:count, :].T
        reconstructed = mx + projected @ B[:count, :]
        ax[i].imshow(np.reshape(reconstructed, [H, W]), cmap='Greys_r')
        ax[i].set_axis_off()
        ax[i].set_title("{} PCs".format(count))
    ax[len(counts)].imshow(np.reshape(flattenedImgs[imgIdx, :], [H, W]), cmap='Greys_r')
    ax[len(counts)].set_axis_off()
    ax[len(counts)].set_title("Original")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
