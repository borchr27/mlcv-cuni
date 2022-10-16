
import argparse
import numpy as np
from scipy.spatial.distance import cdist
import lab02_help

# Class Notes
# use numpy.sort
# use max for the covariance

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
# This exercise has 3 tasks:
# - 'measure' - Implement and test 4 fitness measures from the lecture.
# - 'one-step' - Use the implemented fitness measures to compute one-step forward feature selection.
# - 'sequential' - Use the implemented fitness measures to compute sequential forward feature selection.
parser.add_argument("--task", default="measure", type=str, help="Performed task: 'measure', 'one-step', 'sequential'.")
parser.add_argument("--data", default="hriby2.txt", type=str, help="Path to the file with data.")


def taskMeasure(labels : np.ndarray, features : np.ndarray):
    # Create a list with indices of features you want to use.
    selection = [1, 4, 5, 6]

    # X_tilde - set of data for these features.
    X = features[:, selection]

    # TODO: ===== Task 1 ('measure') =====
    # Modify functions computing different fitness measure within this script such that
    # they return the correct value.
    # You can start with one or two functions, continue with the other tasks and then
    # return to the other functions.
    J = fitnessCons(labels, X)
    print("Consistency fitness: {:.4f}".format(J))

    J = fitnessCbfs(labels, X)
    print("Correlation based fitness: {:.4f}".format(J))

    J = fitnessIcd(labels, X)
    print("Interclass distance fitness: {:.4f}".format(J))

    J = fitnessMi(labels, X)
    print("Mutual information fitness: {:.4f}".format(J))

def taskOneStep(labels : np.ndarray, features : np.ndarray):
    # TODO: ===== Task 2 ('one-step') =====
    # Implement the One-step forward selection.
    #
    # Evaluate individual features, select top 3.
    # Compare the sets of features chosen with different fitness functions.

    # labels (8124,)
    # features (8124,22)

    cols = features.shape[1]    # number of features / columns in the data
    fCons, fCbfs, fIcd, fMi = [], [], [], []    # make lists for fitness data
    fit_vals = [fCons, fCbfs, fIcd, fMi]    # make a list of the fit val lists
    functions = [fitnessCons, fitnessCbfs, fitnessIcd, fitnessMi]      # make a list of the functions to process

    for c in range(cols):
        # for each column cycle thru all values
        X = features[:, c]
        for f, v in zip(functions, fit_vals):
            # for each column in the data process it with the different functions and append value to list
            v.append((f(labels, X), c))

    for fit_list in fit_vals:
        # for each list in fit_vals sort it then print the top three values
        fit_list = sorted(fit_list, reverse=True)
        print([i[1] for i in fit_list[:3]])

def taskSequential(labels: np.ndarray, features : np.ndarray):
    # TODO: ===== Task 3 ('sequential') =====
    # Implement the Sequential forward selection.
    #
    # Evaluate individual features, select top 3.
    # Compare the sets of features chosen with different fitness functions.
    
    functions = [fitnessCons, fitnessCbfs, fitnessIcd, fitnessMi]

    def sequential(labels, features, fitness):
        new_X = np.zeros((features.shape[0],0))
        final_indices = []
        num_feat = features.shape[1]
        
        for _ in range(3):
            scores = np.zeros(num_feat)
            for feat in range(num_feat):
                if feat in final_indices:
                    scores[feat] = 0
                    continue    # goes back to start of for loop
                scores[feat] = fitness(labels, np.concatenate( \
                    (new_X, features[:, feat].reshape(-1,1)), axis=1))
            sorted = np.nanargmax(scores)   # use nanargmax here to grab the first value in the list if multiple are equal (not the last like np.argsort())
            new_X = np.concatenate((new_X, features[:, sorted].reshape(-1,1)), axis=1)
            final_indices.append(sorted)
        print(final_indices)

    for f in functions:
        sequential(labels, features, f)

 

def fitnessCons(labels : np.ndarray, X : np.ndarray) -> float:
    """
    Consistency measure
    
    Write the body of this function. It should calculate the fitness (consistency)
    of a subset of features.
    Input: Vector of gold classes and observations of a subset of features.
    Output: Fitness of the feature set.
    """
    # Make sure that the data array is a matrix(2D) and not a vector(1D).
    X = X if len(X.shape) > 1 else np.reshape(X, [-1, 1])
    # print(X.shape)  (8241, 4)

    # Find unique rows
    C, ix, ic, M = np.unique(X, axis=0, return_index=True, return_inverse=True, return_counts=True)
    # print(C.shape)  (31, 4) 

    # C - unique rows
    # ic - index to C for each row in X
    # ix - index to the first occurence in X for each unique row
    # np.max(ic) = len(ix) - 1     -> Indexing starts at 0
    # C = X[ix, :], X = C[ic, :]
    # M - number of occurences for each unique row
    
    # TODO: For each unique row (a loop through ix) find the number of inconsistent classifications.
    uniqueCount = len(C)
    inconsistency_sum = 0
    for i in range(uniqueCount):
        # Classes of each unique row
        clas = labels[ic == i]
        # Find classifications
        classes, jx, jc, classcounts = np.unique(clas, axis=0, return_index=True, return_inverse=True, return_counts=True)
        # TODO: Find the maximum of classcounts
        M_max = np.max(classcounts)
        # TODO: Compute inconsistency of object 'i'
        inconsistency = M[i] - M_max
        inconsistency_sum += inconsistency

    # TODO: Return the final value J according to the formula from the lecture
    return 1 - inconsistency_sum/len(X)

def fitnessCbfs(labels : np.ndarray, X : np.ndarray) -> float:
    """
    Correlation-based Feature selector

    Write the body of this function. It should calculate the fitness
    (Correlation-based Feature Selector) of a subset of features.
    Input: Vector of gold classes and observations of a subset of features.
    Output: Fitness of the feature set.
    """
    # Make sure that the data array is a matrix(2D) and not a vector(1D).
    X = X if len(X.shape) > 1 else np.reshape(X, [-1, 1])
    # Correlation coefficients
    R = lab02_help.corrcoef(labels, X)
    # TODO: Compute the number of columns in X
    K = X.shape[1]

    # Upper triangular matrix of ones
    indT = np.tri(K + 1, k=-1, dtype=bool).T
    
    # Matlab uses lower triangular matrix because it reads values column by column
    # whereas numpy does it row by row. The matrix R is symmetric.
    coefs = R[indT]
    rcf = np.mean(coefs[0 : K])

    # TODO: Return the final value J according to the formula from the lecture.
    if K > 1:
        rff = np.mean(coefs[K :])
        ans = (K*rcf) / np.sqrt(K+K*(K-1)*rff)
        return 0 if np.isnan(ans) else ans
    else:
        return 0 if np.isnan(rcf) else rcf

def fitnessIcd(labels : np.ndarray, X : np.ndarray) -> float:
    """
    Interclass distance

    Write the body of this function. It should calculate the fitness
    (Interclass distance) of a subset of features.
    Input: Vector of gold classes and observations of a subset of features.
    Output: Fitness of the feature set as mean class distance.
    """
    # Make sure that the data array is a matrix(2D) and not a vector(1D).
    X = X if len(X.shape) > 1 else np.reshape(X, [-1, 1])
    # Objects belonging to class 1
    X_1 = X[labels == 1, :]
    # Objects belonging to class 0
    X_0 = X[labels == 0, :]
    # Matrix of pair-wise distances of points in X_1 and X_0
    d_X = cdist(X_1, X_0, "euclidean")
    D_X = np.mean(d_X)

    # TODO: Return the value of interclass distance according to the formula from the lecture.
    return (X_0.shape[0]/len(X)) * (X_1.shape[0]/len(X)) * D_X

def fitnessMi(labels : np.ndarray, X : np.ndarray) -> float:
    """
    Mutual information
    lables.shape = (8124,)
    x.shape = (8124, 4)

    Write the body of this function. It should calculate the fitness
    (Mutual information) of a subset of features.
    Input: Vector of gold classes and observations of a subset of features.
    Output: Fitness of the feature set - mutual information of the set and the classification.
    """
    # Make sure that the data array is a matrix(2D) and not a vector(1D).
    # - Renamed 'X' to 'Y' to match the formula from the slides.
    Y = X if len(X.shape) > 1 else np.reshape(X, [-1, 1])

    # Follow the formula I(Y;X) = H(Y) - H(Y|X)
    # The following line computes entropy of the whole set (Y in our formula).
    # - NOTE: 'X' in code is not the same as 'X' in the formula.
    entr = getEntropy(Y)

    # TODO: Compute the remaining terms of the formula.
    X0 = X[np.nonzero(labels == 0)]
    X1 = X[np.nonzero(labels == 1)]
    ent_given_X = (getEntropy(X0) * len(X0) + getEntropy(X1) * len(X1)) / len(X)

    # TODO: Return the value of mutual information according to the formula from the lecture.
    return entr - ent_given_X

def getEntropy(X : np.ndarray) -> float:
    """
    Computes the entropy of a single data set.
    """
    # The following code computes the entropy of a single variable.
    # Find unique objects of the set.
    C, ix, ic, occur = np.unique(X, axis=0, return_index=True, return_inverse=True, return_counts=True)
    N = X.shape[0]
    # Compute the probability
    prob = occur / N
    # Compute the entropy
    entr = -np.sum(prob * np.log2(prob))
    return entr


def main(args : argparse.Namespace) -> None:
    # Data preparation
    # Load classification data of certain mushrooms.
    data = lab02_help.parseTextFile(args.data)

    # The first column in data matrix is the class.
    # The remaining columns 2:23 are features.
    # Each row is one observation.

    # Split data into two variables: 'labels' (class) and 'features' (features).
    labels = data[:, 0]
    features = data[:, 1:]

    tasks = {
        "measure" : taskMeasure,
        "one-step" : taskOneStep,
        "sequential" : taskSequential
    }
    if args.task not in tasks:
        raise ValueError("Task '{}' is not recognised!".format(args.task))
    tasks[args.task](labels, features)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
