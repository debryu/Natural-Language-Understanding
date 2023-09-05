# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

# Import the dataset
from nltk.corpus import conll2002
import spacy
from spacy.tokenizer import Tokenizer
import es_core_news_sm
nlp = es_core_news_sm.load()
from tqdm import tqdm
from sklearn_crfsuite import CRF
from conll import *
import pandas as pd

# nlp = spacy.load("es_core_news_sm")
nlp.tokenizer = Tokenizer(nlp.vocab)  # to use white space tokenization (generally a bad idea for unknown data)

# Extract the baseline features
def sent2spacy_features(sent):
    # Parse the document
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    feats = []
    for token in spacy_sent:
        # Add only some features
        token_feats = {
            'bias': 1.0,
            'word.lower()': token.lower_,
            'pos': token.pos_,
            'lemma': token.lemma_
        }
        feats.append(token_feats)
    
    return feats

# Extract the baseline features with suffix
def sent2spacy_features_withSuffix(sent):
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    feats = []
    for token in spacy_sent:
        token_feats = {
            'bias': 1.0,
            'word.lower()': token.lower_,
            'pos': token.pos_,
            'lemma': token.lemma_,
            # Add the suffix feature
            'suffix': token.suffix_
        }
        feats.append(token_feats)
    return feats

# Extract the baseline features with context 
# meaning we also add the features of the previous and next tokens
# Based on the context size 
def sent2spacy_features_withContext(sent, context_size=1):
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    feats = []
    for k,token in enumerate(spacy_sent):
        # Add the features of the current token
        token_feats = {
            'bias': 1.0,
            'word.lower()': token.lower_,
            'pos': token.pos_,
            'lemma': token.lemma_,
        }
        # If the context size is greater than 0 we add the features of the previous and next tokens
        if context_size > 0:
            for i in range(context_size):
                selected_token = i + 1
                # If the token is not the last one
                # Add the features of the next tokens
                if(k + selected_token < len(spacy_sent)):
                    token_feats.update({
                    f'bias{selected_token}': 1.0,
                    f'word.lower(){selected_token}': spacy_sent[k + selected_token].lower_,
                    f'pos{selected_token}': spacy_sent[k + selected_token].pos_,
                    f'lemma{selected_token}': spacy_sent[k + selected_token].lemma_,
                    })
                # If the token is the last one
                # Change the features to EOS (End Of Sentence)
                # As a way to indicate that the next token is not available 
                # and is just a padding token
                else:
                    token_feats.update({
                    f'biasPAD{selected_token}': 1.0,
                    f'word.lower()PAD{selected_token}': 'EOS',
                    f'posPAD{selected_token}': 'EOS',
                    f'lemmaPAD{selected_token}': 'EOS',
                    })
                
                selected_token = - selected_token
                # If the token is not the first one
                # Add the features of the previous tokens
                if(k + selected_token >= 0):
                    token_feats.update({
                    f'bias{selected_token}': 1.0,
                    f'word.lower(){selected_token}': spacy_sent[k + selected_token].lower_,
                    f'pos{selected_token}': spacy_sent[k + selected_token].pos_,
                    f'lemma{selected_token}': spacy_sent[k + selected_token].lemma_,
                    })
                # If the token is the first one
                # Change the features to BOS (Beginning Of Sentence)
                # As a way to indicate that the previous token is not available 
                # and is just a padding token
                else:
                    token_feats.update({
                    f'biasPAD{selected_token}': 1.0,
                    f'word.lower()PAD{selected_token}': 'BOS',
                    f'posPAD{selected_token}': 'BOS',
                    f'lemmaPAD{selected_token}': 'BOS',
                    })
        feats.append(token_feats)
    return feats

# Extract all the features from the conll tutorial
def all_features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],        
    }
    
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
                
    return features

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

# let's get only word and iob-tag
trn_sents = [[(text, iob) for text, pos, iob in sent] for sent in conll2002.iob_sents('esp.train')]
#print(trn_sents[0])

tst_sents = [[(text, iob) for text, pos, iob in sent] for sent in conll2002.iob_sents('esp.testa')]
#print(tst_sents[0])


# Extract all the different sets of features for the training set
trn_baseline_feats = []
trn_withSuffix_feats = []
trn_all_feats = []
trn_withContext1_feats = []
trn_withContext2_feats = []
trn_label = []
trn_token = []
for s in tqdm(trn_sents):
    trn_baseline_feats.append(sent2spacy_features(s))
    trn_withSuffix_feats.append(sent2spacy_features_withSuffix(s))
    trn_all_feats.append([all_features(s, i) for i in range(len(s))])
    trn_withContext1_feats.append(sent2spacy_features_withContext(s,1))
    trn_withContext2_feats.append(sent2spacy_features_withContext(s,2))
    trn_label.append(sent2labels(s))
    trn_token.append(sent2tokens(s))
print(trn_all_feats[0])

# Extract all the different sets of features for the test set
tst_baseline_feats = []
tst_withSuffix_feats = []
tst_all_feats = []
tst_withContext1_feats = []
tst_withContext2_feats = []
for s in tqdm(tst_sents):
    tst_baseline_feats.append(sent2spacy_features(s))
    tst_withSuffix_feats.append(sent2spacy_features_withSuffix(s))
    tst_all_feats.append([all_features(s, i) for i in range(len(s))])
    tst_withContext1_feats.append(sent2spacy_features_withContext(s,1))
    tst_withContext2_feats.append(sent2spacy_features_withContext(s,2))
print(tst_all_feats[0])

# This function will first train the CRF model and then predict the labels for the input
# and then return the predicted labels for the input dataset.
# Here features and labels are for training the CRF model,
# while input is the test set for which we want to predict the labels
def predict(features,labels,input):    
    crf = CRF(
        algorithm='lbfgs', 
        c1=0.1, 
        c2=0.1, 
        max_iterations=100, 
        all_possible_transitions=True
    )

    # workaround for scikit-learn 1.0
    try:
        crf.fit(features, labels)
    except AttributeError:
        pass

    # Predict the labels of the input dataset
    pred = crf.predict(input)

    # Convert the predicted labels to the correct format
    hyp = [[(input[i][j], t) for j, t in enumerate(tokens)] for i, tokens in enumerate(pred)]

    return hyp

def create_results_table(gt, hypo):
    results = evaluate(gt, hypo)

    pd_tbl = pd.DataFrame().from_dict(results, orient='index')
    pd_tbl = pd_tbl.round(decimals=3)
    return pd_tbl


