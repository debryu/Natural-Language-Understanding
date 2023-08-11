# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    data_train, data_test, total_size = get_dataset()
    print(f"Total: {total_size}; Train: {len(data_train)}; Test: {len(data_test)}")
    for order in range(1,5):
        print(f'Trying NLTK tagger order {order} with NO BACKOFF and NO CUTOFF')
        NLTK_noBACKOFF_noCUTOFF = NLTK_tagger(data_train, order=3, sel_backoff=None, sel_cutoff=0)
        print(f'   -> accuracy {NLTK_noBACKOFF_noCUTOFF.accuracy(data_test)}')
        print(f'Trying NLTK tagger order {order} with BACKOFF and NO CUTOFF')
        NLTK_BACKOFF_noCUTOFF = NLTK_tagger(data_train,order=3, sel_backoff=aug_re_tagger, sel_cutoff=0)
        print(f'   -> accuracy {NLTK_BACKOFF_noCUTOFF.accuracy(data_test)}')
        print(f'Trying NLTK tagger order {order} with NO BACKOFF and CUTOFF = 2')
        NLTK_noBACKOFF_CUTOFF = NLTK_tagger(data_train,order=3, sel_backoff=None, sel_cutoff=2)
        print(f'   -> accuracy {NLTK_noBACKOFF_CUTOFF.accuracy(data_test)}')
        print(f'Trying NLTK tagger order {order} with BACKOFF and CUTOFF = 2')
        NLTK_BACKOFF_CUTOFF = NLTK_tagger(data_train,order=3, sel_backoff=aug_re_tagger, sel_cutoff=2)
        print(f'   -> accuracy {NLTK_BACKOFF_CUTOFF.accuracy(data_test)}')
        print(f'Trying NLTK tagger order {order} with NO BACKOFF and CUTOFF = 5')
        NLTK_noBACKOFF_CUTOFF = NLTK_tagger(data_train,order=3, sel_backoff=None, sel_cutoff=5)
        print(f'   -> accuracy {NLTK_noBACKOFF_CUTOFF.accuracy(data_test)}')
        print(f'Trying NLTK tagger order {order} with BACKOFF and CUTOFF = 5')
        NLTK_BACKOFF_CUTOFF = NLTK_tagger(data_train,order=3, sel_backoff=aug_re_tagger, sel_cutoff=5)
        print(f'   -> accuracy {NLTK_BACKOFF_CUTOFF.accuracy(data_test)}')
        print('------------------------------------------------------')

    NLTK_pos_tagger = NLTK_tagger(data_train)
    accuracyNLTK = NLTK_pos_tagger.accuracy(data_test)
    Spacy_pos_tagger = Spacy_tagger()
    tagged_dataset = Spacy_pos_tagger.tag()
    
    tokens_flat_list, tags_flat_list, tags_flat_list_NLTK, corpus, corpus_pos = Spacy_pos_tagger.flatten_and_convert_to_NLTK(tagged_dataset)
   
    accuracySpaCy = Spacy_pos_tagger.compute_accuracy(tags_flat_list_NLTK,tokens_flat_list)
    print('--------------------[ACCURACY]------------------------')
    print(f'NLTK: {accuracyNLTK} SpaCy: {accuracySpaCy}')

