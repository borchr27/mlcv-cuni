
import argparse
import datetime
from typing import Sequence
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics
import tensorflow as tf
import lab10_help

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--validation_size", default=0.2, type=float, help="Test size for dataaset examples.")
# Layer definition parameters:
parser.add_argument("--hidden_layer_size", default=64, type=int, help="Size of the MLP hidden layer.")
parser.add_argument("--activation", default="relu", type=str, help="Activation function for hidden layers.")
parser.add_argument("--L2", default=None, type=float, help="Float describing L2 regularisation or None.")
# Optimization parameters:
parser.add_argument("--optimizer", default="sgd", type=str, help="Defines which optimizer the network should use.")
parser.add_argument("--lr_schedule", default="exponential", type=str, help="Type of learning rate decay.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate of the network.")
parser.add_argument("--decay_steps", default=100, type=int, help="Number of batches after which the learning rate changes.")
parser.add_argument("--decay_rate", default=0.99, type=float, help="Rate at which decay schedule reduces the learning rate.")
# Batch and training iterations parameters:
parser.add_argument("--batch_size", default=16, type=int, help="Batch size for the training algorithm.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs during training.")
# General parameters:
parser.add_argument("--tensorboard", default=False, action="store_true", help="Whether we should compute logs for tensorboard.")
parser.add_argument("--cross_validation", default=False, action="store_true", help="Whether we should perform cross_validation.")
parser.add_argument("--data", default="digits", type=str, help="Which dataset we should use 'digits'/'fashion'.")


def oneHotLabels(labels : np.ndarray, args : argparse.Namespace) -> np.ndarray:
    # TODO: Create one-hot encoding of the labels.
    # - We are classifying into 10 classes so every label should turn into a vector of size 10 which
    #   contains only 1 value '1' at the position given by the label, the remaining values are '0'.
    #   - 10 classes work for both 'digits' and 'fashion' data.
    # NOTE: The result should be a numpy integer array. ('np.asarray(values, dtype=np.int32)' should always do the trick.)
    return None

def defineOptimizer(args : argparse.Namespace) -> tf.keras.optimizers.Optimizer:
    """
    Defines the optimizer together with its learning rate.
    """
    # TODO: Create learning rate schedule and store it in a variable called 'learning_rate'.
    # Based on 'args.lr_schedule' define the appropriate decay or set the value to a constant.
    # - For 'args.lr_schedule == "exponential"' create an instance of 'tf.keras.optimizers.schedules.ExponentialDecay'
    #   with parameters:
    #   - 'initial_learning_rate = args.learning_rate' which specifies the value to start with.
    #   - 'decay_steps = args.decay_steps' which specifies after how many batches we decrease the learning rate.
    #   - 'decay_rate = args.decay_rate' which specifies the multiplicative factor with which we reduce the learning rate (rate < 1).
    # - For 'args.lr_schedule == "inverse_time"' create an instance of 'tf.keras.optimizers.schedules.InverseTimeDecay'
    #   with parameters:
    #   - 'initial_learning_rate = args.learning_rate' which specifies the value to start with.
    #   - 'decay_steps = args.decay_steps' which specifies after how many batches we decrease the learning rate.
    #   - 'decay_rate = args.decay_rate' which specifies the factor by which the lr changes (lr is divided by the product of the rate and 'steps/decay_steps').
    # - For 'args.lr_schedule == "constant"' simply set learning rate to the cosntant 'args.learning_rate'
    #
    # For exact formulas and more information, look at tensorflow documentation.
    # NOTE: The 'decay_rate' parameter has different meaning (and should be set to different values) for exponetial and inverse time decay.
    learning_rate = None

    # TODO: Create and return an optimizer based on the 'args.optimizer' argument.
    # - For 'args.optimizer == "adam"' return an instance of 'tf.keras.optimizers.Adam' with the given learning rate.
    #   - You can try changing parameters 'beta_1' and 'beta_2' as well. This is not really necessary but it can help in some cases.
    # - For 'args.optimizer == "sgd"' return an instance of 'tf.keras.optimizers.SGD' with the given learning rate.
    #   - You can try specifying 'momentum' which turns abrupt changes in gradient to a smoother transition by introducing momentum
    #     to the gradient direction.
    # - For 'args.optimizer == "rmsprop"' return an instance of 'tf.keras.optimizers.RMSprop' with the given learning rate.
    optimizer = None

    return optimizer

def defineRegularizer(args : argparse.Namespace) -> tf.keras.regularizers.Regularizer:
    # TODO: If 'args.L2' is not None (it is a floating point number) then create a regularizer
    # by instantiating the class 'tf.keras.regularizers.L2' with a single parameter 'args.L2'.
    # Otherwise return 'None'
    return None


def defineModel(args : argparse.Namespace, input_shape : Sequence[int]) -> tf.keras.Model:
    """
    Creates multi-layered perceptron neural network model (MLP) using tensorflow functional API.
    """
    # NOTE: 'lab10_tf_hello.py' shows model definition with 'Sequential' API which is the simplest
    # possible way of defining the model.
    # A slightly more complex definition of the model (but much more expressive) is through tensorflow
    # functional API. It allows us to connect any layers together by "calling" the layer with its predecessor.
    # - An example of how to use functional API is shown below.

    # NOTE: There is not really a good way to define the model properly from the arguments without
    # the need to specify lots of them, therefore the model topology is modified manually within the script.

    # Let's define regularization - it forces weights to be smaller which reduces over-fitting
    # - Large weights usually mean that the model learned some very specific case which usually isn't in the data distribution.
    regularizer = defineRegularizer(args)

    # Define the input layer.
    inputs = tf.keras.layers.Input(shape=input_shape)
    # Flatten layer in case of >1D data.
    hidden = tf.keras.layers.Flatten()(inputs)

    # TODO: Add 1 or more 'tf.keras.layers.Dense' layers with args.hidden_layer_size units (or custom numbers)
    # with 'activation' specified by 'args.activation' (or different activation for every layer.)
    # and 'kernel_regularizer' specified by regularizer.
    hidden = tf.keras.layers.Dense(args.hidden_layer_size, activation=args.activation, kernel_regularizer=regularizer)(hidden)

    # Let's add the output layer classifying into 10 classes using softmax activation.
    outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(hidden)

    # Let's create the model using the functional API.
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Let's define the optimizer for our model.
    optimizer = defineOptimizer(args)

    # Let's compile the model so that it is ready for training.
    model.compile(
        optimizer = optimizer,
        loss = tf.keras.losses.CategoricalCrossentropy(), # For classification using one-hot labels.
        metrics = tf.keras.metrics.CategoricalAccuracy()
    )

    return model

def main(args : argparse.Namespace):
    # Let's load our data:
    # TODO: Draw several images from the 'Fashion MNIST' dataset to see how it looks like.
    dataset = {
        "digits" : lab10_help.prepareMnist,
        "fashion" : lab10_help.prepareFashionMnist,
    }
    if args.data not in dataset.keys():
        raise ValueError("Unknown dataset: '{}'".format(args.data))
    all_train_data, test_data, all_train_labels, test_labels, label_names = dataset[args.data]()

    # Let's convert the training labels to one-hot representation.
    all_train_labels_1h = oneHotLabels(all_train_labels, args)

    # Let's define our model.
    model = defineModel(args, all_train_data[0].shape)
    # Print an ASCII decription of the model.
    model.summary()

    # NOTE: To evaluate the performance of our model, we should do full cross validation (meaning that we should run the training several times)
    # and compare the average error. Then we should select the best hyper-parameters based on these results.
    # However, the training process is quite slow, and therefore, try training with only a single validation split. Then, if you have
    # time, you can do the full cross-validation and investigate whether the single split gave you a good estimation of the performance.
    if not args.cross_validation:
        # TODO: Train the model using a single validation split with various hyperparameters from 'args' and different network
        # topologies.
        # Try changing at least the following hyper-parameters:
        # - args.learning_rate - values such as 0.1, 0.01, 0.001, 0.0001 (lower values need longer training)
        # - args.activation - at least 'relu' and 'tanh' (In general, ReLU should perform better.)
        # - args.optimizer - at least 'adam' and 'sgd'
        # - args.lr_schedule - at least 'exponential' and 'constant'
        # - args.batch_size - try powers of two, e.g., 8, 16, 32, ..., 256 (Number of images processed in one update)
        # - args.epochs - try lower numbers because it will significantly increase the training time. (The number of passes through the whole dataset.)
        # - also try to change the topology of the network - add more layers and change the number of hidden units.
        # You are welcome to try and modify any parameter or piece of code in this script - the above mentioned options
        # are the main ones with which you can see the difference immediately.

        # TODO: Use 'train_test_split' function with 'test_size=args.validation_size' to split the training data into
        # training and validation sets. Store them in variables 'train_data', 'val_data', 'train_labels_1h', 'val_labels_1h'
        # - Make sure that you are splitting the one-hot labels and not the original ones.
        train_data, val_data, train_labels_1h, val_labels_1h = None

        # Let's define tensorboard callbacks so that we are able to look at the training curves and graph of the model.
        if args.tensorboard:
            log_dir = "logs/" + "tf_" + args.data + "_mlp_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            model_callbacks = [tensorboard_callback]
        else:
            model_callbacks = None

        # Train the model
        model.fit(
            train_data, train_labels_1h,
            batch_size = args.batch_size,
            epochs = args.epochs,
            validation_data = (val_data, val_labels_1h),
            callbacks = model_callbacks,
        )

    else:
        # TODO: Perform a 5-fold cross-validation to find the best hyper-parameters and model topology.
        # - Do this only if you have time at the end because, although simple, it will take a lot of time to evaluate.
        pass

    print()
    print("=" * 50)
    print("Model evaluation")
    print("=" * 50)
    print()
    # TODO: Use the trained network in 'model' to predict the test_data and compute the test accuracy of the model.
    # - method 'predict' as in scikit-learn, however, it returns probabilities for all classes instead of the label itself.
    test_accuracy = None
    print("Test set accuracy is {:>5.2f}%".format(test_accuracy * 100))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
