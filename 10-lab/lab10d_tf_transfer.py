
import argparse
import datetime
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import lab10_help

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--test_size", default=0.75, type=float, help="Test size for dataaset examples.")
# Layer definition parameters:
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of the MLP hidden layer.")
parser.add_argument("--activation", default="relu", type=str, help="Activation function for hidden layers.")
parser.add_argument("--L2", default=None, type=float, help="Float describing L2 regularisation or None.")
# Optimization parameters:
parser.add_argument("--optimizer", default="adam", type=str, help="Defines which optimizer the network should use.")
parser.add_argument("--lr_schedule", default="exponential", type=str, help="Type of learning rate decay.")
parser.add_argument("--transfer_lr", default=0.001, type=float, help="Learning rate of the network for transfer learning.")
parser.add_argument("--transfer_ds", default=10, type=int, help="Number of batches after which the learning rate changes.")
parser.add_argument("--transfer_dr", default=0.99, type=float, help="Rate at which decay schedule reduces the learning rate.")
parser.add_argument("--fine_tune_lr", default=0.0001, type=float, help="Learning rate of the network for fine-tuning.")
parser.add_argument("--fine_tune_ds", default=10, type=int, help="Number of batches after which the learning rate changes.")
parser.add_argument("--fine_tune_dr", default=0.99, type=float, help="Rate at which decay schedule reduces the learning rate.")
# Batch and training iterations parameters:
parser.add_argument("--batch_size", default=16, type=int, help="Batch size for the training algorithm.")
parser.add_argument("--transfer_epochs", default=2, type=int, help="Number of epochs during training of transfer model.")
parser.add_argument("--fine_tune_epochs", default=1, type=int, help="Number of epochs for fine-tuning.")
# General parameters:
parser.add_argument("--tensorboard", default=False, action="store_true", help="Whether we should compute logs for tensorboard.")


def defineTransferOptimizer(args : argparse.Namespace) -> tf.keras.optimizers.Optimizer:
    # TODO: The same as 'defineOptimizer' in 'lab10b_tf_cnn'.

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(args.transfer_lr, args.transfer_ds, args.transfer_dr)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    return optimizer

def defineFineTuneOptimizer(args : argparse.Namespace) -> tf.keras.optimizers.Optimizer:
    # TODO: The same as "defineOptimizer" in 'lab10b_tf_cnn', however, fine-tuning optimizer should
    # be slower than transfer optimizer because we are optimizing the parmeters of the base model as well
    # and greater gradient values can cause the original model to diverge.
    # - Generally, it is enough to reduce the learning rate of transfering by a factor of 10.

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(args.fine_tune_lr, args.fine_tune_ds, args.fine_tune_dr)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    return optimizer

def defineModel(efficientnet_base, classes : int, args : argparse.Namespace) -> tf.keras.Model:
    # TODO: Finish the last fully connected part of the classification network based on efficientnet for transfer
    # learning of Caltech101 dataset.

    # Define inputs for Caltech101 dataset.
    inputs = tf.keras.layers.Input(shape=[224, 224, 3])
    # Let's use efficientnet as a layer in the network - this is possible with functional API.
    # - training=False is here because the model contains BatchNormalisation which causes it to run in inference mode.
    #   - You don't have to worry about this at the moment.
    hidden = efficientnet_base(inputs, training=False)
    # The last layer in our efficientnet is convolution - GlobalMaxPooling reduces 2D slices into an array
    # by selecting only the maximum from each slice.
    hidden = tf.keras.layers.GlobalMaxPooling2D()(hidden)

    # TODO: Add 1 or more fully-connected (Dense) layers as the classification head for our network.


    # Let's define outputs of our model classifying into the given number of classes.
    outputs = tf.keras.layers.Dense(classes, activation=tf.nn.softmax)(hidden)

    # Create the model and compile it.
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer = defineTransferOptimizer(args),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics = tf.keras.metrics.SparseCategoricalAccuracy(),
    )

    return model

def main(args : argparse.Namespace):
    # NOTE: Transfer learning and fine-tuning exercise.
    # You can learn more about transfer learning on the tensorflow/keras webpages:
    # https://www.tensorflow.org/tutorials/images/transfer_learning

    # TODO: Let's load our data:
    # Download Caltech101 dataset from: https://data.caltech.edu/records/mzrjq-6wc02
    # - The zipped archive contains a series of directories specifying the individual classes.
    # - Make sure that you obtain a directory called '101_ObjectCategories' containing the folders with individual classes
    #   and copy this directory next to this script so that the following method works with the given relative path.
    #
    # TODO: Use only 'airplanes', 'Faces', 'Motorbikes' and 'watch' categories by specifying the 'select' parameter
    # of 'loadCaltech' function.
    # - Using more classes will significantly increase the processing time, setting selection to 'None' will use all classes.
    #
    # - Ensure that the values of pixels are integer in the range from 0 to 255 otherwise efficientnet won't work.
    # - Convert the data returned by the loading function into numpy arrays and rename the 'images' array into 'data'.
    # - The target shape (224, 224) has to match the input shape of the model ~(224, 224, 3).
    selection = ["airplanes", "Faces", "Motorbikes", "watch"]
    images, labels, label_names = lab10_help.loadCaltech("101_ObjectCategories", (224, 224), selection)
    data = None
    
    # TODO (If you have time):
    # Load all Caltech images and store the data and label arrays into .npy/.npz files and use those instead.
    # - It will be much faster and you can experiment on the whole dataset.


    # Let's define our model by using EfficientNetB0 as the main pretrained part.
    efficientnet_base : tf.keras.Model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    # Let's freeze the model so that it's weights will not be updated during transfer learning.
    efficientnet_base.trainable = False
    # Print the structure of EfficientNetB0
    efficientnet_base.summary()

    # TODO: Define the transfer model.
    model = defineModel(efficientnet_base, len(label_names), args)
    # Print an ASCII decription of the model.
    model.summary()

    # Split the data into training and testing set.
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=args.test_size, stratify=labels)

    # Let's define tensorboard callbacks so that we are able to look at the training curves and graph of the model.
    if args.tensorboard:
        log_dir = "logs_transfer/" + "tf_transfer_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        model_callbacks = [tensorboard_callback]
    else:
        model_callbacks = None

    # NOTE: Do the transfer learning. The base efficientnet network remains unchanged during the training
    # but the weights in our classification head are trained to use the features computed by the efficientnet.
    model.fit(
        train_data, train_labels,
        batch_size=args.batch_size,
        epochs=args.transfer_epochs,
        callbacks=model_callbacks,
    )

    # NOTE: And now, let's try fine-tuning.
    # - It is different from transfer learning, in that it trains the weights of the base model as well.
    #   This means that the training procedure has to be careful and slow otherwise the gradients from
    #   new data can cause diveregence of the base model.

    # Let's unfreeze the model so that it's weights are updated.
    efficientnet_base.trainable = True
    # Let's compile the model with custom parameters.
    # - Slow training means in general very low learning rate - try values that are about 1/10 of the transfer
    #   learning rate.
    model.compile(
        optimizer = defineFineTuneOptimizer(args),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics = tf.keras.metrics.SparseCategoricalAccuracy(),
    )

    # NOTE: Finally, let's fine-tune the model by running the training procedure again.
    model.fit(
        train_data, train_labels,
        batch_size=args.batch_size,
        epochs=args.fine_tune_epochs,
        callbacks=model_callbacks,
    )

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
