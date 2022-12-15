
import os
from typing import Tuple, Sequence, List
import numpy as np
from pathlib import Path
from PIL import Image
import skimage.transform
import tensorflow as tf

class MnistDataset:
    """
    Loads the MNIST data saved in .npy or .npz files.

    If the 'labels' argument is left as None then the class assumes that the file
    in 'data' is .npz and creates attributes, with the same name as specified
    during the file creation, containing the respective numpy arrays.

    If the 'labels' argument is set to a string path then the class assumes that
    the files were saved as .npy and it will create two attributes: 'imgs' which
    contains the contents of the 'data' file and 'labels' with the contents of
    the 'labels' file.

    If you chose to save the arrays differently then you might have to modify
    this class or write your own loader.
    """

    def __init__(self, data : str = "mnist_train.npz", labels : str = None):

        if not os.path.exists(data):
            raise ValueError("Requested mnist data file not found!")
        if (labels is not None) and (not os.path.exists(labels)):
            raise ValueError("Requested mnist label file not found!")

        if labels is None:
            dataset = np.load(data)
            for key, value in dataset.items():
                setattr(self, key, value)
        else:
            self.imgs = np.load(data)
            self.labels = np.load(labels)

def prepareMnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Loads the MNIST dataset from tensorflow repository."""
    
    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()
    label_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    return train_data, test_data, train_labels, test_labels, label_names

def prepareFashionMnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Loads a different dataset, similar to MNIST, which contains images of clothes."""

    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    label_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    return train_data, test_data, train_labels, test_labels, label_names

def prepareCifar10() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Loads a different dataset, similar to MNIST, which contains coloured images of various things."""

    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.cifar10.load_data()
    label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    return train_data, test_data, train_labels, test_labels, label_names

def loadCaltech(relative_path : str, target_resolution : Tuple[int, int] = None, select : Sequence[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Loads Caltech101 images from the given directory extracted from the downloaded archive.
    This method can optionally resize all images to the same shape. This happens if 'target_resolution'
    is not None and it will also extend all monochromatic images to three channels by setting
    all three to the same matrix.

    Additionally, this method can return only a subset of the dataset (selecting only several
    of the classes) by specifying 'select' argument with a list of names of requested classes.

    Arguments:
    - 'relative_path' - Path to the Caltech101 directory.
    - 'target_resolution' - Resizes all images to this size if not None.
    - 'select' - List of categories which should be retrieved, returns all of them if None.

    Returns:
    - Caltech images (optionally resized).
    - Labels of the images.
    - Label names.
    """
    labeled_dirs = os.listdir(relative_path)
    images = []
    labels = []
    label_names = []

    for labeled_dir in labeled_dirs:
        name = Path(labeled_dir).stem
        if select is not None and labeled_dir not in select:
            continue
        label_names.append(name)
        label = len(label_names) - 1

        image_paths = os.listdir(os.path.join(relative_path, labeled_dir))
        for image_path in image_paths:
            image_path = os.path.join(relative_path, labeled_dir, image_path)
            with Image.open(image_path) as imHandle:
                img = np.asarray(imHandle)
            images.append(img)
            labels.append(label)

    if target_resolution is not None:
        # Rescale the images.
        scaled = []
        for image in images:
            image = skimage.transform.resize(image, target_resolution)
            # Extend monochromatic images to RGB.
            if len(image.shape) < 3:
                image = np.dstack([image, image, image])
            scaled.append(image)
        images = scaled
    
    return np.asarray(images), np.asarray(labels), label_names
