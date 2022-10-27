
import numpy as np
from matplotlib import pyplot as plt

def main():
    """
    This function describes several exercises which help getting accustomed
    to python and libraries necessary for machine learning and computer vision.
    These exercises focus on array manipulation, whereas training models and applying
    algorithms is part of the regular practicals.

    The exercises require the relevant python libraries to be already installed.
    Information about what is necessary to install (and what is optional) is
    available in python-setup.txt provided together with this script.
    """

    # Create a list of numbers from 1 to 45.
    
    # Turn this list into numpy ndarray. (numpy.asarray)
    
    # Print out the shape of this array. (attribute 'shape') [should be one number]
    # - The size of the array in all dimensions is called shape - it is represented by
    #   a tuple of numbers.
    
    # Reshape the array into the shape [5, 9] (numpy.reshape)
    # - You can use '-1' to fill one dimension, e.g., [-1, 9] and [5, -1] will achieve
    #   the same as [5, 9] on an array with 45 elements.
    
    # Print out the array and check the ordering of the numbers.
    
    # Print out row and vector slices of the array (3rd row and 5th column)
    # - Using ':' as an index will select all elements in a particular dimension
    #   (so called 'slice')
    
    # Transpose the array and check its shape and the ordering of its numbers.
    # - For 2D arrays, calling T on the array is enough, e.g. 'numbers.T', but
    #   'numpy.transpose' is a general method for transposition.
    #   The result should have shape [9, 5]
    
    # Reshape the [5, 9] array into the shape [5, 3, 3] and print out 2D slices
    # of the array (mainly blocks with dimensions 1x3x3)
    
    # Transpose the three dimensional array such that the first dimension becomes
    # last and the other two are moved backwards, i.e., [1, 2, 0].
    
    # Print out 2D slices of the array (mainly blocks with dimensions 1x3x5 and 3x3x1)
    # - Compare with the slices before transposition.
    
    # Return to the 5x9 array.
    # Compute and print out the column-wise and row-wise sum, maximum, minimum,
    # mean and median (numpy.sum/max/min/mean/median). Use 'axis' argument of the functions.
    # - The results should be vectors containing one value for each row/column
    
    # Create a 2x3 matrix filled with 2 and 3x4 matrix filled with 3. (still as numpy ndarray)
    # - Function 'numpy.ones' might be usefull.
    
    # Multiply these matrices together and print out the result. Symbol '@' is a shortcut
    # for matrix multiplciation (numpy.matmul).

    # Create a 500x500 image with zeros (numpy.zeros). Fill the inner 300x300 pixels
    # with ones using array slicing. Display the image using pyplot (plt.figure, plt.imshow,
    # plt.show)

    # Prepare an array of 50 numbers in the range from 0 to 4*PI (numpy.pi, numpy.linspace). Compute
    # the values of function 2*sin(x+PI/4)+0.5 from the prepared array of x-values. (numpy.sin)
    # plot the function. (plt.plot instead of plt.imshow)

if __name__ == "__main__":
    main()
