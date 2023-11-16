# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    train_ds = read_data('dataset/LAB11_part2/laptop14_train.txt')
    test_ds = read_data('dataset/LAB11_part2/laptop14_test.txt')

    