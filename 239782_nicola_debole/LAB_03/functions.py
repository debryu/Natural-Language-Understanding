# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

# NLTK StupidBackoff
from nltk.lm import StupidBackoff
from nltk.corpus import gutenberg
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
import math
from itertools import chain
from nltk.lm import Vocabulary
from nltk.util import everygrams, pad_sequence
flatten = chain.from_iterable
from nltk.lm import NgramCounter
import numpy as np

# Dataset
def createDataset(train_test_ratio=0.9):
    macbeth_sents = [[w.lower() for w in sent] for sent in gutenberg.sents('shakespeare-macbeth.txt')]
    total_size = len(macbeth_sents)
    train_indx = math.ceil(total_size * train_test_ratio)
    trn_data = macbeth_sents[:train_indx]
    tst_data = macbeth_sents[train_indx:]
    return trn_data, tst_data

# Custom perplexity
def compute_ppl_nltk(model, ngrams):
    scores = [] 
    # Convert the ngrams to a list
    ngrams = list(ngrams)
    for w in ngrams:
        scores.extend([-1 * model.logscore(w[-1], w[0:-1])])
    return math.pow(2.0, np.asarray(scores).mean())
  

#STUPID BACKOFF NLTK
def stupid_backoff_NLTK(trn_data, tst_data, N = 4, cutoff = 40):
    # Initialize the model
    stupidoff = StupidBackoff(0.4,N)
    # Flatten the data
    macbeth_words = flatten(trn_data)
    # Create the vocabulary
    lex = Vocabulary(macbeth_words, unk_cutoff=cutoff)
    # Replace Out of Vocabulary words with UNK
    macbeth_oov_sents = [list(lex.lookup(sent)) for sent in trn_data]
    # Create the ngrams
    padded_ngrams, flat_text = padded_everygram_pipeline(N, macbeth_oov_sents)
    # Fit the model
    stupidoff.fit(padded_ngrams, flat_text)
    # Prepare the test set using the same vocabulary
    # I am using the train set vocabulary as it is the bigger one (0.9 of the dataset)
    test_oov_sents = [list(lex.lookup(sent)) for sent in tst_data]
    ngrams_test, flat_text_test = padded_everygram_pipeline(stupidoff.order, test_oov_sents)
    ngrams_testset = [w for x in ngrams_test for w in x if len(w) == stupidoff.order]

    print(f'Perplexity NLTK stupid backoff: {stupidoff.perplexity(ngrams_testset)}')
    print(f'Custom ppl function for NLTK stupid backoff: {compute_ppl_nltk(stupidoff, ngrams_testset)}')


class stupid_backoff_MYVERSION:
    def __init__(self, model_order = 4, cutoff = 40):
        self.model_order = model_order
        self.alpha = 0.4
        self.cutoff = cutoff
        self.counter = None
        self.lex = None

    def fit(self,trn_data):
        # Flatten the data
        macbeth_words = flatten(trn_data)
        # Create the vocabulary
        self.lex = Vocabulary(macbeth_words, unk_cutoff = self.cutoff)
        # Replace Out of Vocabulary words with UNK
        train_oov_sents = [list(self.lex.lookup(sent)) for sent in trn_data]
        # Create the ngrams
        padded_ngrams, flat_text = padded_everygram_pipeline(self.model_order, train_oov_sents)
        # Count all the ngrams
        self.counter = NgramCounter(padded_ngrams)
        
    
    def probability(self, word, context):
        # If there is no context, use the unigram probability
        if len(context) == 0:
            return self.counter[word] / sum(self.counter.unigrams.values())
        else:
            # If the ngram exists, return the probability
            if self.counter[context][word] > 0:
                # Check if the context is not a unigram 
                if len(context) > 1:
                    # If the context is not a unigram, divide by the count of that ngram
                    return self.counter[context][word] / self.counter[context[:-1]][context[-1]]
                else:
                    # If the context is a unigram, divide by that unigram count
                    return self.counter[context][word] / self.counter[context[0]]
            else:
                # If the ngram does not exist, use the stupid backoff algorithm to compute the probability
                return self.alpha * self.probability(word, context[1:])

    def logscore(self, word, context):
        return math.log(self.probability(word, context), 2)
    
    def compute_ppl(self,tst_data):
        # Prepare the test set using the same vocabulary, removing the OOV words
        test_oov_sents = [list(self.lex.lookup(sent)) for sent in tst_data]
        # Create the ngrams
        ngrams_test, flat_text_test = padded_everygram_pipeline(self.model_order, test_oov_sents)
        # Only keep the ngrams of the same order as the model
        ngrams_testset = [w for x in ngrams_test for w in x if len(w) == self.model_order]
        
        scores = [] 
        ngrams = list(ngrams_testset)
        for w in ngrams:
            scores.extend([-1 * self.logscore(w[-1], w[0:-1])])
        return math.pow(2.0, np.asarray(scores).mean())

