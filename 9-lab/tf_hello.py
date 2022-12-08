
# TODO: Setup for tensorflow on university computers.
#
# This script contains examples of using tensorflow (some math and neural network training).
# Your goal is to execute this script on your (or a university) computer without errors or
# crashing. The following approach has to be used on university computers.
#
# - The university computers do not have tensorflow installed, and therefore, we will have to do
#   a couple of tricks to get it working.
#
# - If you are not using university computers, but your own, then you can ignore this, however,
#   tensorflow might be quite sensitive about dependency versions. So, in case, you have problems running
#   this script then you might need to upgrade/downgrade some packages (Ideally in a virtual environment :)).
#
# TODO: Creating a virtual python environment with tensorflow.
# - We have to work with commandline, I am not aware of any GUI for this.
# - Python webpage with detailed information: https://docs.python.org/3/tutorial/venv.html
#
# 1) Navigate to some directory where you want to create the environment.
#    - The environment will be a directory with some name (such as tf-env) containing a copy of the python installation.
#    - For example: $HOME/python-envs
# 
# 2) Create virtual python environment by executing:
#    >>> python -m venv tf-env
#    - You may choose different name and make sure that you are using python 3.
#    - NOTE: I am testing this on python 3.10, previously, it worked on python 3.9 and 3.8 but lower versions might have a problem I haven't encountered.
#
# 3) Activate the environment:
#    >>> source tf-env/bin/activate             [Unix/MacOS]
#    >>> .\tf-env\Scripts\activate.bat          [Windows - CMD, PowerShell has some problems]
#    - Your shell should change to something like '(tf-env) $ ...'
#
# 4) Install required python packages:
#    >>> python -m pip install tensorflow numpy scipy matplotlib scikit-image scikit-learn
#
# 5) Now try running this script. Make sure you run it from the command-line with active virtual
#    environment - I am not sure how do different editors react to virtual environments.
#    - NOTE: You might see a CUDA related error saying there is no available GPU - ignore this, the training should still run fine.
#
# TODO: After you finish your tensorflow exercises:
# 6) To deactivate an active virtual environment, simply call:
#    >>> deactivate
#
# 7) To delete/remove deactivated virtual environment, use:
#    >>> rm -rf tf-env                          [Unix/MaxOs]
#    >>> rmdir /s tf-env                        [Windows]
#    - Or simply remove the directory through file explorer.

import argparse
import datetime
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics
import tensorflow as tf
from tensorflow import keras
from keras import Sequential, optimizers, losses, metrics, layers
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--tol", default=1e-4, type=float, help="Tolerance for value comparisons.")
parser.add_argument("--seed", default=None, type=int, help="Seed for RNG.")
parser.add_argument("--validation_size", default=0.2, type=float, help="Test size for dataaset examples.")
parser.add_argument("--test_size", default=0.2, type=float, help="Test size for dataaset examples.")
parser.add_argument("--hidden_layer_size", default=64, type=int, help="Size of the MLP hidden layer.")
parser.add_argument("--activation", default="relu", type=str, help="Activation function for hidden layers.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate of the network.")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size for the training algorithm.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs during training.")
parser.add_argument("--tensorboard", default=False, action="store_true", help="Whether we should compute logs for tensorboard.")


def runMathTest(args : argparse.Namespace):
    """
    This method tries running some basic math functions of tensorflow.
    """
    print("Basic math test:")
    values = np.asarray([1, 6, 5, 9, 2, 3, 5, 4, 7], float)

    # Try computing mean, sum, min, max using tensorflow to see whether the library is included correctly.
    mean_np, mean_tf = np.mean(values), tf.reduce_mean(values)
    sum_np, sum_tf = np.sum(values), tf.reduce_sum(values)
    min_np, min_tf = np.min(values), tf.reduce_min(values)
    max_np, max_tf = np.max(values), tf.reduce_max(values)
    print("Mean: {}".format(np.abs(mean_np - mean_tf) < args.tol))
    print("Sum: {}".format(np.abs(sum_np - sum_tf) < args.tol))
    print("Min: {}".format(np.abs(min_np - min_tf) < args.tol))
    print("Max: {}".format(np.abs(max_np - max_tf) < args.tol))

    indices = [2, 3, 6]
    np_one_hot = np.zeros([3, 7])
    np_one_hot[np.arange(3), indices] = 1
    tf_one_hot = tf.one_hot(indices, 7)
    print("One hot: {}".format(np.abs(np.sum(np_one_hot - tf_one_hot)) < args.tol))
    print()


def main(args : argparse.Namespace):
    runMathTest(args)

    # We are going to train a simple network here.
    print("Basic Sequential model test.")
    # Get dataset from tensorflow API
    # - It will download and extract the file into several numpy arrays.
    # - We are also going to split the data into training and testing group.
    bh_train, bh_test = tf.keras.datasets.boston_housing.load_data("boston_housing.npz", test_split=args.test_size, seed=args.seed)
    train_data, test_data, train_labels, test_labels = bh_train[0], bh_test[0], bh_train[1], bh_test[1]

    # Let's define a simple neural network with one hidden layer.
    model = Sequential()
    # We have to define input layer with shape of our features (Boston dataset has 13 features in each sample).
    model.add(layers.Input(shape=[13]))
    # Our data is already one dimensional but if we were working with images we would want to flatten them
    # before using a fully connected layer (== Dense). There is no Flatten layer before convolutions.
    model.add(layers.Flatten())
    # Let's add one hidden layer with 'args.hidden_layer_units' and activation function 'args.activation'.
    # - activation can be for example 'sigmoid', 'tanh', 'relu'
    # - number of hidden units is always an integer.
    model.add(layers.Dense(args.hidden_layer_size, activation=args.activation))
    # Output layer has one value and since we are computing regression, we do not have to set an activation.
    model.add(layers.Dense(1))

    # In tensorflow, we have to compile the model before we can use it.
    # We also have to specify several attributes of the model.
    # - optimizer - Updates weights of the model according to the gradient of the loss.
    # - loss - Loss computed on the outputs of the model, for regression, we will simply use MSE.
    # - metrics - (Optional) metric which is reported by the mdoel during training. Let's use MSE.
    model.compile(
        optimizer = optimizers.Adam(args.learning_rate),
        loss = losses.MeanSquaredError(),
        metrics = metrics.MeanSquaredError()
    )

    # Let's split the training data into validation set as well.
    tr_data, val_data, tr_labels, val_labels = train_test_split(train_data, train_labels, test_size=args.validation_size)

    # Let's create a callback for Tensorboard, which will allow us to log information about our model
    # and visualise it.
    # TODO: To visualise information about the model, you have to:
    # - Run in the commandline: >>> tensorboard --logdir logs
    #   It will start a server and tell you the address and port (something like http://localhost:6006/)
    #   - Navigate to this address in your internet browser.
    if args.tensorboard:
        log_dir = "logs/" + "tf_hello_boston_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        model_callbacks = [tensorboard_callback]
    else:
        model_callbacks = None

    # Finally, let's train the model for a set number of epochs (one epoch = one iteration over the whole training dataset)
    # with the specified batch size - the gradient will be computed from the whole batch to reduce variance of the loss.
    model.fit(
        tr_data, tr_labels, batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(val_data, val_labels),
        callbacks=model_callbacks,
    )

    # Let's evaluate the model on the test set.
    test_predictions = model.predict(test_data)
    print()
    print("MSE on the test set:  {:7.3f}".format(sklearn.metrics.mean_squared_error(test_labels, test_predictions)))
    print("RMSE on the test set: {:7.3f}".format(sklearn.metrics.mean_squared_error(test_labels, test_predictions, squared=False)))

    _, ax = plt.subplots(1, 1, figsize=(6, 7), subplot_kw={'aspect': 'equal'})
    ax.scatter(test_predictions, test_labels)
    ax.set_xlabel("Predicted values")
    ax.set_ylabel("True housing values")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
