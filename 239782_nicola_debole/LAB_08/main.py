# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results

    # Initialize the classifier, create the vector with the features and the ground truth labels
    # and split the data into train and test sets
    classifier, BOWvector, labels, stratified_split = initialize_model()
    # Compute the score using only Bag Of Words
    scores = compute_score(classifier, BOWvector, labels, stratified_split)
    print("Only BOW score:", scores)
    # Produce the collocational features and compute the score
    # The features are extended with POS and NGRAMS
    posngram_colloc_feat_vector = produce_collocational_features('POS+NGRAMS')
    scores = compute_score(classifier, posngram_colloc_feat_vector, labels, stratified_split)
    print("Only extended coll. feat. (POS+NGRAMS):", scores)
    # Produce the collocational features and then concatenate them with the BOW features
    new_colloc_feat_vector = produce_collocational_features()
    concatenated_vectors = concatenate_features(BOWvector, new_colloc_feat_vector)
    # Compute the score using the concatenated vector of features
    scores = compute_score(classifier, concatenated_vectors, labels, stratified_split)
    print("BOW and new collocational feature vectors score:", scores)

    # Run only the first split to make it faster
    print("Evaluating metrics on just the first split:")
    # Split the data into train and test sets
    for i,data in enumerate(stratified_split.split(BOWvector, labels)):
        print(f"-.-.-.-.-.-.-[  Split {i+1}  ] -.-.-.-.-.-.-")
        test,train = data
        print('Computing Lesk original... (takes around 8 minutes)')
        print('Lesk original:', evaluate_metrics('original', mapping, test))
        print('Computing Lesk similarity... (takes around 8 minutes)')
        print('Lesk similarity:', evaluate_metrics('lesk_similarity', mapping, test))
        # Remove the break to evaluate all the splits
        break

    