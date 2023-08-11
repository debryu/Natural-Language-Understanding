# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from sklearn.datasets import fetch_20newsgroups
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SKLEARN_STOP_WORDS

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    print('Initializing the data...')
    data = fetch_20newsgroups(subset='all')
    num_split = 2
    print('------------------------------[COUNT VECTORIZE]------------------------------')
    count_vectorize(data,num_split) 
    print('------------------------------[TFIDF]------------------------------') 
    tfidf(data,num_split)
    print('------------------------------[TFIDF with MIN/MAX]------------------------------')
    tfidf(data,num_split,True,None,0.1,0.85)
    print('------------------------------[TFIDF without stopwords]------------------------------')
    tfidf(data,num_split,True,SKLEARN_STOP_WORDS,0.0,1.0)
    print('------------------------------[TFIDF without lowercase]------------------------------')
    tfidf(data,num_split,False,None,0.0,1.0)