
import load_mnist
from matplotlib import pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def main():
    SHOW_IMAGES = False
    SAVE = False
    # 1. Use load_mnist.loadMnist function to read 60000 training images and class labels
    #    of MNIST digits (files train-images.idx3-ubyte and train-labels.idx1-ubyte).
    train_images, train_labels = load_mnist.loadMnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
    print(f"train images shape: {train_images.shape}")
    print(f"train labels shape: {train_labels.shape}")

    # 2. Preview one image from each class.
    if SHOW_IMAGES:
        for i in range(10):
            item_indexes = np.where(train_labels == i)
            plt.figure()
            plt.imshow(train_images[item_indexes[0][0]])
            plt.show()

    # 3. Transform the image data, such that each image forms a row vector,
    #    alternatively, you can transform it such that each image is a column vector. Nevertheless,
    #    remember which transformation you choose.
    #    - NOTE: Math in lectures assumes the column-format, exercises will assume the row-format
    #      (Row-format is used by most Python libraries).
    #    - NOTE: You need to use np.transpose to rearrange dimensions. Now, n x m x 3 is to be converted to 3 x 
    #    (n*m), so send the last axis to the front and shift right the order of the remaining axes (0,1).
    #    Finally , reshape to have 3 rows.
    train_images = np.reshape(train_images, [60000,784])
    print(f"train images shape: {train_images.shape}")
    
    # 4. Save the image matrix and the labels in a numpy .npy or .npz files.
    #    numpy.save / numpy.savez / numpy.savez_compressed, loading is done using numpy.load
    
    if SAVE: np.savez("trainFile", train_images, train_labels)

    # 5. Do the same for 10000 test digits (files t10k-images.idx3-ubyte and
    #    t10k-labels.idx1-ubyte)
    #    - Both files (training and testing set) will be used during the semester.
    print("-"*80)
    k_images, k_labels = load_mnist.loadMnist("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
    print(f"t10k images shape: {k_images.shape}")
    print(f"t10k labels shape: {k_labels.shape}")

    if SHOW_IMAGES:
        for i in range(10):
            item_indexes = np.where(k_labels == i)
            plt.figure()
            plt.imshow(k_images[item_indexes[0][0]])
            plt.show()
    
    k_images = np.reshape(k_images, [10000,784])
    print(f"t10k images shape: {k_images.shape}")
    if SAVE: np.savez("t10kFile", k_images, k_labels)
    print("-"*80)

    # 6. Now, try to load the created files, display some of the images and print their respective
    #    labels.
    train_file = np.load("trainFile.npz")
    print(train_file["arr_0"].shape)  # images with default name because none was assigned
    print(train_file["arr_1"].shape)

    print(f'the value of the image is {train_file["arr_1"][0]}')
    plt.figure()
    plt.imshow(np.reshape(train_file["arr_0"][0], [28,28])) # we have to reshape the array bc its (784,)
    plt.show()


if __name__ == "__main__":
    main()
