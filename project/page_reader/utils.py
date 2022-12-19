
import os
import numpy as np

class EMnistDataset:
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

    def __init__(self, data : str = "emnist_train.npz", labels : str = None):

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

class Phrases:

    phrases = [
        "move forward", "go forward", "go straight", "roll forward", "stumble on", "shamble on", "walk forward",
        "turn left", "wheel left", "rotate left", "spin left",
        "turn right", "wheel right", "rotate right", "spin right",
        "turn back", "wheel back", "rotate back", "spin back", "turn around", "wheel around",
        "move left", "walk left", "shamble left", "go leftish", "stumble left", "skulk left",
        "move right", "walk right", "roll right", "go rightish", "skulk right",
        "move back", "walk back", "shamble back", "go backward", "stumble back", "skulk back",
    ]

    commands = ["F", "L", "R", "B", "ML", "MR", "MB"]

    phraseToCommand = ["F"] * 7 + ["L"] * 4 + ["R"] * 4 + ["B"] * 6 + ["ML"] * 6 + ["MR"] * 5 + ["MB"] * 6
