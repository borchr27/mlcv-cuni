
import argparse
import numpy as np
import sklearn.linear_model
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sklearn.neural_network
import lab09_help


parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--seed", default=None, type=int, help="Seed for RNG.")
parser.add_argument("--points", default=20, type=int, help="Number of points per class.")
parser.add_argument("--test_points", default=20, type=int, help="Number of points for testing.")
parser.add_argument("--hidden_layer_sizes", nargs="*", default=[10, 5], type=int, help="Size of the MLP hidden layer.")
parser.add_argument("--activation", default="relu", type=str, help="Activation function for the network.")
parser.add_argument("--max_iter", default=2000, type=int, help="Maximum number of iterations of the MLP.")
parser.add_argument("--threshold", default=0.5, type=float, help="Classification threshold from probability.")
parser.add_argument("--print_proba", default=False, action="store_true", help="Whether we should print the computed probabilities.")
parser.add_argument("--multiclass", default=False, action="store_true", help="Computes multiclass problem instead of binary one.")


def relu(values : np.ndarray) -> np.ndarray:
    # TODO: Implement the ReLU function working on numpy arrays.
    # -> ReLU(x) = max(0, x)
    raise NotImplementedError()

def logistic(values : np.ndarray) -> np.ndarray:
    # TODO: Implement the logistic sigmoid function working on numpy arrays.
    # -> sigmoid(x) = 1 / (1 + e^-x)
    raise NotImplementedError()

def softmax(values : np.ndarray) -> np.ndarray:
    # TODO: Implement the softmax function working on numpy arrays.
    # - Softmax is used for multi-class classification and for each sample it computes the probability
    #   of belonging to each class. The probability is computed through normalisation of exponentiated values.
    # -> softmax(x)_i = e^x_i / sum_j(e^x_j)
    raise NotImplementedError()

def forward(inputs : np.ndarray, mlp : sklearn.neural_network.MLPClassifier, args : argparse.Namespace) -> np.ndarray:
    # TODO: Implement the forward pass of a MLP which uses ReLU activation in hidden
    # layers and logistic sigmoid (or softmax) for probability computation in the last layer.
    # In detail:
    # - Trained weights of the model are stored in 'mlp.coefs_'
    # - Trained biases of the model are stored in 'mlp.intercepts_'
    # - For every layer except the last one, use ReLU activation by calling the above 'relu' function.
    # - For the last layer:
    #   - Use logistic sigmoid activation by calling the above 'logistic' function in case of binary classification.
    #   - Use softmax activation by calling the above 'softmax' function in case of multiclass classification.
    # - Return the computed probabilities - the result after logistic/softmax activation.
    raise NotImplementedError()

def main(args : argparse.Namespace):
    # NOTE: Manual prediction from trained MLP using ReLU activation.
    
    generator = np.random.RandomState(args.seed)

    # Data generation.
    mu1 = [1, 1]
    sigma1 = [[0.6, 0], [0, 0.3]]
    mu2 = [4, 2]
    sigma2 = [[0.6, 0], [0, 1.2]]
    mu3 = [2.5, 3.5]
    sigma3 = [[0.4, 0.2], [0.2, 0.8]]

    # Create binary or multiclass problem.
    # NOTE: You can add more classes or change parameters after you finish the algorithm.
    if args.multiclass:
        mus = [mu1, mu2, mu3]
        sigmas = [sigma1, sigma2, sigma3]
        labelValues = [0, 1, 2]
    else:
        mus = [mu1, mu2]
        sigmas = [sigma1, sigma2]
        labelValues = [0, 1]

    data, labels = lab09_help.mvnGenerateData(generator, args.points, mus, sigmas, labelValues)

    # Train the MLP.
    mlp = sklearn.neural_network.MLPClassifier(args.hidden_layer_sizes, activation=args.activation, max_iter=args.max_iter, random_state=generator)
    mlp.fit(data, labels)

    # Generate the test data.
    testData, testLabels = lab09_help.mvnGenerateData(generator, args.test_points, mus, sigmas, labelValues)

    # Compute probabilities using our forward function and also with the builtin method.
    manualPredictionsProba = forward(testData, mlp, args)
    builtinPredictionsProba = mlp.predict_proba(testData)

    # Optionally, print the computed probabilities.
    print("="*50)
    if args.print_proba:
        print(manualPredictionsProba)
        print("="*50)
        print(builtinPredictionsProba)
        print("="*50)

    # Compute the final predictions (builtin method returns probability for each class, our method
    # returns only probability for the class '1' in the binary case)
    builtinPredictions = np.argmax(builtinPredictionsProba, axis=1)

    if not args.multiclass:
        manualPredictions = np.asarray(manualPredictionsProba >= args.threshold, np.int32)
        manualPredictions = manualPredictions.ravel()
        # Hack to ensure the same subtraction behaviour for binary and multiclass problem.
        builtinPredictionsProba = np.c_[builtinPredictionsProba[:, 1]]
    else:
        manualPredictions = np.argmax(manualPredictionsProba, axis=1)

    # Verify correctness of the computation.
    print("The following two values compare your forward pass with the builtin one.")
    print("- The first value should be 0.0 (Floating point rounding errors might be possible).")
    print("- For default threshold 0.5, the second value should be exactly 0.")
    print("The absolute sum of prediction proba difference: {}".format(np.sum(np.abs(manualPredictionsProba - builtinPredictionsProba))))
    print("The absolute sum of prediction difference: {}".format(np.sum(np.abs(manualPredictions - builtinPredictions))))
    print()
    print("="*50)

    # Create a meshgrid for visual comparison of the predictions.
    xDots = np.linspace(np.min(testData[:, 0]) - 1.5, np.max(testData[:, 0]) + 1.5, 500)
    yDots = np.linspace(np.min(testData[:, 1]) - 1.5, np.max(testData[:, 1]) + 1.5, 500)
    xx, yy = np.meshgrid(xDots, yDots)
    meshData = np.vstack([xx.ravel(), yy.ravel()]).T

    # Predict the meshgrid for decision boundary visualisation.
    predictions = mlp.predict(meshData)

    # Visualise the results. The two images should be identical for multiclass and binary threshold = 0.5.
    colours = np.asarray(["red", "blue", "green", "yellow", "magenta", "cyan", "black"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'aspect': 'equal'})
    ax[0].pcolormesh(xx, yy, predictions.reshape(xx.shape), cmap=ListedColormap(colours[:len(labelValues)]),  alpha=0.25, shading="auto")
    ax[0].scatter(testData[:, 0], testData[:, 1], edgecolors=colours[manualPredictions.ravel()], marker='s', facecolors='none', s=150, linewidths=1.5, label="Predicted class")
    ax[0].scatter(testData[:, 0], testData[:, 1], c=colours[testLabels], label="True class")
    ax[0].legend()
    ax[0].set_title("Manual predictions of test data")
    ax[1].pcolormesh(xx, yy, predictions.reshape(xx.shape), cmap=ListedColormap(colours[:len(labelValues)]),  alpha=0.25, shading="auto")
    ax[1].scatter(testData[:, 0], testData[:, 1], edgecolors=colours[builtinPredictions], marker='s', facecolors='none', s=150, linewidths=1.5, label="Predicted class")
    ax[1].scatter(testData[:, 0], testData[:, 1], c=colours[testLabels], label="True class")
    ax[1].legend()
    ax[1].set_title("Builtin predictions of test data")
    fig.suptitle("Comparison of results from builtin and manual predictions (They should be identical).")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
