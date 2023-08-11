# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
# Import tqdm for the progress bar
from tqdm import tqdm
import pandas as pd

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    # First create the dataset from the dependency treebank
    data = create_dataset()
    
    print('Comparing Spacy and Stanza to see if the dependency tags are the same:')
    spacy_dependency_graphs = []
    stanza_dependency_graphs = []
    # Then preprocess the data (join the words in a sentence) and parse it with spacy and stanza
    for sentence in tqdm(preprocess_data(data)):
        spacy_dataframe, spacy_depgraph = spacy_parsing(sentence)
        spacy_dependency_graphs.append(spacy_depgraph)
        stanza_dataframe, stanza_depgraph = stanza_parsing(sentence)
        stanza_dependency_graphs.append(stanza_depgraph)
        # Merge the two dataframes in a nicer way with just the minimal information
        results = pd.concat([spacy_dataframe[['FORM','DEPREL']], stanza_dataframe[['DEPREL']]], axis=1)
        # Rename the columns
        new_headers = ['FORM', 'Spacy', 'Stanza']
        results.columns = new_headers
        print(results)
    print('As visible from the previous lines, the dependency tags are not the same')
    print('--------------------------------[EVALUATION]--------------------------------')
    print('Evaluating Spacy:')
    spacy_las, spacy_uas = evaluate_parses(spacy_dependency_graphs)
    print('LAS: ', spacy_las)
    print('UAS: ', spacy_uas)
    print('Evaluating Stanza:')
    stanza_las, stanza_uas = evaluate_parses(stanza_dependency_graphs)
    print('LAS: ', stanza_las)
    print('UAS: ', stanza_uas)
