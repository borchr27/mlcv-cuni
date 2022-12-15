
import argparse
import datetime
from typing import Sequence
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import lab10_help

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--validation_size", default=0.2, type=float, help="Test size for dataaset examples.")
# Layer definition parameters:
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of the MLP hidden layer.")
parser.add_argument("--activation", default="relu", type=str, help="Activation function for hidden layers.")
parser.add_argument("--L2", default=None, type=float, help="Float describing L2 regularisation or None.")
# Optimization parameters:
parser.add_argument("--optimizer", default="adam", type=str, help="Defines which optimizer the network should use.")
parser.add_argument("--lr_schedule", default="exponential", type=str, help="Type of learning rate decay.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate of the network.")
parser.add_argument("--decay_steps", default=100, type=int, help="Number of batches after which the learning rate changes.")
parser.add_argument("--decay_rate", default=0.99, type=float, help="Rate at which decay schedule reduces the learning rate.")
# Batch and training iterations parameters:
parser.add_argument("--batch_size", default=32, type=int, help="Batch size for the training algorithm.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs during training.")
# General parameters:
parser.add_argument("--tensorboard", default=False, action="store_true", help="Whether we should compute logs for tensorboard.")
parser.add_argument("--cross_validation", default=False, action="store_true", help="Whether we should perform cross_validation.")
parser.add_argument("--data", default="digits", type=str, help="Which dataset we should use 'digits'/'fashion'.")


def defineOptimizer(args : argparse.Namespace) -> tf.keras.optimizers.Optimizer:
    # TODO: Use exactly the same definition of optimizer as in 'lab10a_tf_mlp' exercise.
    # - You can copy your code. This function defaults to Adam with exponential decay.

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, args.decay_steps, args.decay_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    return optimizer

def defineRegularizer(args : argparse.Namespace) -> tf.keras.regularizers.Regularizer:
    # TODO: The same as in 'lab10a_tf_mlp'.
    return None

def defineModel(args : argparse.Namespace, input_shape : Sequence[int]) -> tf.keras.Model:
    # TODO: Define convolutional neural network.
    # We are going to use the tensorflow functional API to define the convolutional model.
    # - It is done the same way as in the mlp exercise but with new types of layers.

    # Let's define the regularization.
    regularizer = defineRegularizer(args)

    # Define the input layer.
    if len(input_shape) == 2:
        input_shape = list(input_shape) + [1] # Input shape has to contain the number of channels as well.
    inputs = tf.keras.layers.Input(shape=input_shape)
    # NOTE: We do not flatten the data because we are going to use convolutions.
    hidden = inputs

    # TODO: Add 1 or more convolutional and pooling layers to create your CNN.
    # - 2D convolution is created using 'tf.keras.layers.Conv2D'.
    # - Max pooling layer is created using 'tf.keras.layers.MaxPool2D', it has 'pool_size', 'strides' and 'padding' with
    #   the same meaning as 'kernel_size', 'strides' and 'padding' in convolution, but it pools.
    # - 'tf.keras.layers' package contains various layers you can use in your model definition, you are free to experiment.
    hidden = tf.keras.layers.Conv2D(
        filters = 8,            # Number of resulting channels (The number of chunnels should increase with decreasing image size)
        kernel_size = 3,        # Size of the kernel (3 == 3x3), you can specify rectangular kernel as well by writing a tuple, e.g. (4, 3).
        strides = 1,            # Stride of the convolution, if >1 then it reduces the size of the result. Can be used instead of pooling.
        padding = "valid",      # 'valid' - convolution only within the image -> smaller result, 'same' - image with borders -> result has the same size as input.
        activation = tf.nn.relu,# Activation function, 'tf.nn' contains all activations defined within tensorflow.
        kernel_regularizer = regularizer
    )(hidden)

    # TODO: Add a fully connected layer after we are finished with convolutions.
    # - First, we have to flatten the remaining 2D data (use tf.keras.layers.Flatten).
    # - We can add multiple layers here as well.
    

    # Let's add the output layer classifying into 10 classes using softmax activation.
    outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(hidden)

    # Let's create the model.
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Let's define the optimizer for our model.
    optimizer = defineOptimizer(args)

    # Let's compile the model so that it is ready for training.
    # NOTE: SparseCategoricalCrossentropy allows us to use numerical labels directly instead of one-hot encoding.
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=tf.keras.metrics.SparseCategoricalAccuracy()
    )

    return model

def main(args : argparse.Namespace):
    # Let's load our data:
    dataset = {
        "digits" : lab10_help.prepareMnist,
        "fashion" : lab10_help.prepareFashionMnist,
        "cifar10" : lab10_help.prepareCifar10,
    }
    if args.data not in dataset.keys():
        raise ValueError("Unknown dataset: '{}'".format(args.data))
    all_train_data, test_data, all_train_labels, test_labels, label_names = dataset[args.data]()

    # Let's define our model
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
        # Try changing the parameters as in the 'lab10a_tf_mlp' exercise to get the best performance you can.
        # - It should be easier to get better performance than with the MLP.

        # TODO: Use 'train_test_split' function with 'test_size=args.validation_size' to split the training data into
        # training and validation sets. Store them in variables 'train_data', 'val_data', 'train_labels', 'val_labels'
        # - We do not need one-hot labels.
        train_data, val_data, train_labels, val_labels = None

        # Let's define tensorboard callbacks so that we are able to look at the training curves and graph of the model.
        if args.tensorboard:
            log_dir = "logs/" + "tf_" + args.data + "_cnn_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            model_callbacks = [tensorboard_callback]
        else:
            model_callbacks = None

        # Train the model.
        model.fit(
            train_data, train_labels,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(val_data, val_labels),
            callbacks=model_callbacks,
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
    # We can use model.evaluate to compute the model metrics, where we specified categorical accuracy.
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)
    print("Test set accuracy is {:>5.2f}%".format(test_accuracy * 100))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
