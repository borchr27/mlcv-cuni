
import argparse
import numpy as np
from typing import Sequence
from sklearn import tree
import sklearn.model_selection
import matplotlib.pyplot as plt
import lab07_help

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--task", default="importance", type=str, help="Decision tree showcase, one of: 'importance', 'weighting', 'crossval', 'pruning'")

parser.add_argument("--max_depth", default=None, type=int, help="The maximum depth of the trained trees.")
parser.add_argument("--min_samples_split", default=2, type=int, help="The minimum number of samples required fro a node split.")
parser.add_argument("--min_samples_leaf", default=1, type=int, help="The minimum number of samples, which has to be present in a leaf.")
parser.add_argument("--max_leaf_nodes", default=None, type=int, help="The maximum number of leaf nodes allowed in the tree.")
parser.add_argument("--weight_0", default=1.0, type=float, help="Weight of the class 0(edible) in weighted showcase.")
parser.add_argument("--weight_1", default=100.0, type=float, help="Weight of the class 1(poisonous) in weighted showcase.")
parser.add_argument("--kfold_split", default=5, type=int, help="Number of folds in crossvalidation showcase.")
parser.add_argument("--show_crossval_trees", default=False, action="store_true", help="Starts drawing trees from the crossvalidation example.")
parser.add_argument("--test_size", default=0.2, type=float, help="Fraction of the data chosen as the testing set.")
parser.add_argument("--seed", default=42, type=int, help="Seed for RNG during tree pruning.")
parser.add_argument("--show_prune_trees", default=False, action="store_true", help="Starts drawing trees from the pruning example.")


def importanceShowcase(args : argparse.Namespace, features : np.ndarray, labels : np.ndarray, names : Sequence[np.ndarray], featureNames : Sequence[str]):
    # Fit binary tree:
    # TODO: Experiment with the parameters of the decision tree:
    # - max_depth - The maximum depth of the tree (None - no constraint).
    # - min_samples_split - The minimum number of samples required in a node for splitting.
    # - min_samples_leaf - Every leaf has to have at least this many samples.
    # - max_leaf_nodes - Maximum number of leaves (None - no constraint).
    shroomTree = tree.DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_leaf_nodes=args.max_leaf_nodes
    )
    shroomTree.fit(features, labels)

    tree.plot_tree(shroomTree)
    plt.tight_layout()
    plt.show()

    importances = shroomTree.feature_importances_
    _, ax = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={'aspect': 'auto'})
    ax.bar(range(0, features.shape[1]), importances, width=0.9)
    ax.set_title("Feature importance")
    ax.set_xticks(range(0, features.shape[1]))
    ax.set_xticklabels(featureNames, rotation=90)
    plt.tight_layout()
    plt.show()

    predictions = shroomTree.predict(features)
    print("=" * 50)
    print("Feature importance summary")
    lab07_help.labelSummary(predictions, labels, names)

def weightedClasses(args : argparse.Namespace, features : np.ndarray, labels : np.ndarray, names : Sequence[np.ndarray], featureNames : Sequence[str]):
    # TODO: Try to change class weights together with tree size limitations from 'importanceShowcase'
    # to observe the effect of class weighting on the decision tree classification.
    # - NOTE: scikit-learn is not able to train classifiers with cost matrices (which would specify
    #   the cost of misclassification and allow us to say that FN is worse than FP).
    weights = {
        0: args.weight_0, # EDIBLE
        1: args.weight_1 # POISONOUS
    }
    shroomTree = tree.DecisionTreeClassifier(
        class_weight=weights,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_leaf_nodes=args.max_leaf_nodes
    )
    shroomTree.fit(features, labels)

    tree.plot_tree(shroomTree)
    plt.tight_layout()
    plt.show()

    predictions = shroomTree.predict(features)
    print("=" * 50)
    print("Weighted classes summary")
    lab07_help.labelSummary(predictions, labels, names)

def crossValidation(args : argparse.Namespace, features : np.ndarray, labels : np.ndarray, names : Sequence[np.ndarray], featureNames : Sequence[str]):
    # K-fold crossvalidation with summary printed for the validation set.
    kf = sklearn.model_selection.StratifiedKFold(args.kfold_split, shuffle=True)

    print("=" * 50)
    print("Crossvalidation tree summary")

    for trainIndices, validationIndices in kf.split(features, labels):
        # TODO: Observe the changes in validation performance with different values of arguments.
        shroomTree = tree.DecisionTreeClassifier(
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_leaf_nodes=args.max_leaf_nodes
        )
        shroomTree.fit(features[trainIndices], labels[trainIndices])

        # TODO: Analyse the produced tree on validation data, use 'tree.plot_tree' and 'lab07help.labelSummary'
        # - you can use 'args.show_crossval_trees' to enable/disable plotting of trees

        pass

def pruning(args : argparse.Namespace, features : np.ndarray, labels : np.ndarray, names : Sequence[np.ndarray], featureNames : Sequence[str]):
    # Split the data into a training and testing set.
    features_train, features_test, labels_train, labels_test = sklearn.model_selection.train_test_split(features, labels, test_size=args.test_size, stratify=labels)

    print("=" * 50)
    print("Pruning based on cost-complexity summary")

    # TODO: Try different parameters and class weights.
    # - Start with class weights 1.0 and 1.0, default is 1.0 and 100.0
    generator = np.random.RandomState(args.seed)
    weights = {
        0: args.weight_0, # EDIBLE
        1: args.weight_1 # POISONOUS
    }
    shroomTree = tree.DecisionTreeClassifier(
        class_weight=weights,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_leaf_nodes=args.max_leaf_nodes,
        random_state=generator
    )
    # Find alphas which cause cost complexity pruning.
    path = shroomTree.cost_complexity_pruning_path(features_train, labels_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    for ccp_alpha, impurity in zip(ccp_alphas, impurities):
        clf = tree.DecisionTreeClassifier(
            class_weight=weights,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_leaf_nodes=args.max_leaf_nodes,
            random_state=generator,
            ccp_alpha=ccp_alpha
        )
        clf.fit(features_train, labels_train)

        # TODO: Analyse the produced tree on the test data, use 'tree.plot_tree' and 'lab07help.labelSummary'
        # - you can use 'args.show_prune_trees' to enable/disable plotting of trees
        
        pass

def main(args : argparse.Namespace):
    # Loads the full mushroom dataset from Lab02.
    # - 'shrooms' contains categorical features and labels from the dataset.
    # - 'names' contains names of the feature values.
    shrooms, names, featureNames = lab07_help.loadShrooms("shrooms_string.txt")
    features = shrooms[:, 1:]
    labels = shrooms[:, 0]

    tasks = {
        "importance" : importanceShowcase,
        "weighting" : weightedClasses,
        "crossval" : crossValidation,
        "pruning" : pruning,
    }
    if not args.task in tasks.keys():
        raise ValueError("Unrecognised task: {}!".format(args.task))
    # TODO: Try changing arguments and observe the effect on feature importance (task 'importance').
    # TODO: Try changing the class weight and observe the effect on misclassification rate (task 'weighting').
    # TODO: Observe the effect of tree parameters on cross-validation performance (task 'crossval').
    # TODO: Observe pruning by increasing cost-complexity alpha (task 'pruning').
    # - All nodes with alpha value lower than the given ccp_alpha are pruned.
    tasks[args.task](args, features, labels, names, featureNames)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
