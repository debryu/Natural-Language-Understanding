# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    baseline_prediction = predict(trn_baseline_feats, trn_label, tst_baseline_feats)
    baseline_results = create_results_table(tst_sents, baseline_prediction)
    print("Baseline results:")
    print(baseline_results)
    print("--------------------------------------")

    withSuffix_prediction = predict(trn_withSuffix_feats, trn_label, tst_withSuffix_feats)
    withSuffix_results = create_results_table(tst_sents, withSuffix_prediction)
    print("Adding suffix to features:")
    print(withSuffix_results)
    print("--------------------------------------")

    all_prediction = predict(trn_all_feats, trn_label, tst_all_feats)
    all_results = create_results_table(tst_sents, all_prediction)
    print("Using all features from the tutorial:")
    print(all_results)
    print("--------------------------------------")

    withContext1_prediction = predict(trn_withContext1_feats, trn_label, tst_withContext1_feats)
    withContext1_results = create_results_table(tst_sents, withContext1_prediction)
    print("With features window size 1:")
    print(withContext1_results)
    print("--------------------------------------")

    withContext2_prediction = predict(trn_withContext2_feats, trn_label, tst_withContext2_feats)
    withContext2_results = create_results_table(tst_sents, withContext2_prediction)
    print("With features window size 2:")
    print(withContext2_results)
    print("--------------------------------------")

