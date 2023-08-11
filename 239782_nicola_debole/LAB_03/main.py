# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    train_data,test_data = createDataset()
    stupid_backoff_NLTK(train_data,test_data,4,40)
    model = stupid_backoff_MYVERSION(4,40)
    model.fit(train_data)
    print(f'Perplexity of my stupid backoff implementation: {model.compute_ppl(test_data)}')
