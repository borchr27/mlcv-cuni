
from typing import Tuple, List
import argparse
import numpy as np

# NOTE: Arguments in this source file are only for your convenience. They will not be used during
# evaluation but you can use them for easier experimentation with your solution.
parser = argparse.ArgumentParser()
parser.add_argument("--example", default=None, type=int, help="Example argument.")

class PageReader:

    def __init__(self, note : str) -> None:
        # TODO: Place any initialisation of your method in here. If your method doesn't require
        # initialisation then you can keep this method empty but you have to remove 'raise NotImplementedError()'.
        # Argument 'note' is passed to this method by the evaluator. It is a single string argument that you
        # can use for something, e.g., name of the trained/loaded model in the fit method.
        raise NotImplementedError()

    def fit(self, training : bool) -> None:
        # TODO: Place your training, model saving and model loading code in here.
        # This method will be called once before any evaluation takes place so you should set up all objects
        # required for the task solving.
        # For example, running evaluation of a complete set of tasks will be done by creating an instance
        # of this class, calling this 'fit' method and then calling 'solve' for every task.
        # >>> pr = PageReader()
        # >>> pr.fit(args.training)
        # >>> pr.solve(task1)
        # >>> pr.solve(task2) etc.
        #
        # This method should be able to train your solution on demand. That means, if the argument 'training'
        # is True then you should train your classification models and use the newly trained ones. If the argument
        # is False then you should load models from saved files.
        raise NotImplementedError()

    def solve(self, pages : np.ndarray) -> Tuple[List[List[str]], List[List[str]], List[str]]:
        # TODO: This method should solve a single page reading task.
        # It gets a stack of page images on its input. You have to process and classify all pages
        # and return the text you extracted from each page, phrases which you matched on each page
        # (both in the input order of the pages) and the final list of movement instructions in the correct order
        # according to the page numbers.
        raise NotImplementedError()

def main(args : argparse.Namespace) -> None:
    # NOTE: You can run any test or visualisation that you want here or anywhere else.
    # However, you must not change the signature of 'PageReader.__init__', 'PageReader.fit' or 'PageReader.solve'.
    #
    # Your solution will be evaluated using 'evaluator.py' as it was given to you. This means
    # that you should not change anything in 'evaluator.py'. Also, you should make sure that
    # your solution can be evaluated with on-demand training.
    #
    # Evaluation of your solution through the commandline can look like this:
    # >>> python .\evaluator.py --type=single --set=python_train --name=005 --verbose=2 --note=something
    # >>> python .\evaluator.py --type=full --set=python_train --verbose=1 --training
    # >>> python .\evaluator.py --type=full --set=python_validation --verbose=1
    raise NotImplementedError()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
