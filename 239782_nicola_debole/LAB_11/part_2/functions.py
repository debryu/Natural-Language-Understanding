# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from sklearn.cluster import KMeans
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        '''
            Returns:
                glove_embedded_sent: the sentence embedded with GloVe
                label: the label of the sentence (One hot encoding of NONE, POS, NEG, NEU)
                cosine: the cosine similarity of the sentence with each cluster
                cluster_labels: the label of each word (ASPECT or NO ASPECT)
        '''
        glove_embedded_sent = torch.tensor(self.dataset[index]['emb']).to(dtype=torch.float32) #[seq_len, 300]
        label = self.dataset[index]['label'].to(dtype=torch.float32) #[seq_len, 4]
        cosine = self.dataset[index]['cosine'].to(dtype=torch.float32) #[seq_len, 2]
        cluster_labels = self.dataset[index]['cluster'].to(dtype=torch.float32) #[seq_len]
        return glove_embedded_sent, label, cosine, cluster_labels
    

def batchify(list_of_samples):
    '''
        Makes the length of each sentence the same by padding with zeros inside the batch
        Returns:
            output: the sentences embedded with GloVe
            labels: the labels of the sentences (One hot encoding of NONE, POS, NEG, NEU)
            cosine_aspects: the cosine similarity of the sentences with each cluster
            cluster_labels: the labels of each word (ASPECT or NO ASPECT)
            lenghts: the lenghts of each sentence (for packing later)
    '''
    #list_of_samples = samples x [len, 300]
    lenghts = []
    for i, (emb,label,aspects,cluster_labels) in enumerate(list_of_samples):
        lenghts.append(label.shape[0])

    max_len = max(lenghts)
    output = torch.zeros(len(list_of_samples), max_len, 300)
    labels = torch.zeros(len(list_of_samples), max_len, 4)
    # Save where the words should be assigned to (cluster)
    cluster_labels = torch.zeros(len(list_of_samples), max_len) # Where each word belongs to (ASPECT or NO ASPECT)
    cosine_aspects = torch.zeros(len(list_of_samples), max_len, 512) # cosine similarity wtr each cluster
    cluster_labels += 3
    for i, (emb,label,aspects,clusters) in enumerate(list_of_samples):
        output[i,:emb.shape[0],:] = emb
        labels[i,:label.shape[0],:] = label 
        cosine_aspects[i,:aspects.shape[0],:] = aspects
        cluster_labels[i,:clusters.shape[0]] = clusters

    output = output.to(device)
    labels = labels.to(device)
    cosine_aspects = cosine_aspects.to(device)
    cluster_labels = cluster_labels.to(device)

    return output, labels, cosine_aspects, cluster_labels, lenghts
    

'''
__________________________________________________________________________________________
EVALUATION FUNCTIONS

'''
def count_errors(preds, labels):
    '''
        Old function used to count the errors of the model 
        (not used anymore)
    '''
    if len(preds) != len(labels):
        raise Exception('Invalid shapes!')
    total = len(preds)
    errors = 0
    for i in range(total):
        if preds[i] != labels[i]:
            errors += 1
    correct = total - errors
    return errors, correct, total


SMALL_POSITIVE_CONST = 1e-4
def evaluate(model, dataset, batch_size=32):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=batchify)
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    f1_posnone = []
    f1_negnone = []
    f1_posneg = []
    
    for i, (emb, label, cosine, cluster_labels, lenghts) in enumerate(dataloader):
        output, aspect = model(emb,cosine, lenghts)
        output = output.reshape(-1,4)
        label = label.reshape(-1,4)
        cluster_labels = cluster_labels.reshape(-1)
        aspect = aspect.reshape(-1,2)
        tt,tf,ft,ff = compare_aspect(aspect, cluster_labels)
        true_pos += tt
        true_neg += ff
        false_pos += tf
        false_neg += ft
        r1,r2,r3,correct,total,neutral,other = compare_polarity(output, label)
        f1_posnone.append(r1[2])
        f1_negnone.append(r2[2])
        f1_posneg.append(r3[2])
    print(f'F1 POS NONE: {np.mean(f1_posnone)}')
    print(f'F1 NEG NONE: {np.mean(f1_negnone)}')
    print(f'F1 POS NEG: {np.mean(f1_posneg)}')
    p,r,f = score(true_pos,false_pos,false_neg,true_neg)
    print(f'Aspect Detection: Precision: {p}, Recall: {r}, F1: {f}')


def compare_aspect(predictions,labels):
    '''
        Count the errors and the type of errors of the model
        For the aspect detection task
        Args:
            predictions: the predictions of the model (if the word is an aspect or not) 
            labels: the labels (if the word is an aspect or not)
        Returns:
            true_true: the number of words that were predicted as aspect and were actually aspect
            true_false: the number of words that were predicted as aspect but were not aspect
            false_true: the number of words that were predicted as not aspect but were actually aspect
            false_false: the number of words that were predicted as not aspect and were not aspect
    '''
    predictions = torch.argmax(predictions, dim=1)
    assert len(predictions) == len(labels)
    total = len(predictions)
    true_true = 0
    true_false = 0
    false_true = 0
    false_false = 0
    for i in range(total):
        if labels[i] == 3: # Ignore when the label is 3 which means that it is padded
            continue
        if predictions[i] != labels[i]:
            if predictions[i] == 0.:
                false_true += 1
            else:
                true_false += 1
        else:
            if predictions[i] == 0.:
                false_false += 1
            else:
                true_true += 1
    return true_true, true_false, false_true, false_false

def compare_polarity(predictions,labels):
    '''
        Count the errors and the type of errors of the model
        For the polarity detection task
        Args:
            predictions: the predictions of the model (if the sentence is positive, negative, neutral or none) 
            labels: the labels (if the sentence is positive, negative, neutral or none)
        Returns:
            r1: the precision, recall and f1 score of the model for the positive and none classes
            r2: the precision, recall and f1 score of the model for the negative and none classes
            r3: the precision, recall and f1 score of the model for the positive and negative classes
            correct: the number of aspects that were predicted correctly
            total: the total number of words
            neutral_neutral: the number of aspects that were predicted as neutral and were actually neutral
            other_errors: the number of words that were predicted incorrectly
    '''
    predictions = torch.argmax(predictions, dim=1)
    assert len(predictions) == len(labels)
    total = len(predictions)
    positive_negative = 0
    positive_positive = 0
    negative_positive = 0
    negative_negative = 0
    none_positive = 0
    none_negative = 0
    none_none = 0
    neutral_neutral = 0
    positive_none = 0
    negative_none = 0
    other_errors = 0
    for i in range(total):
        if predictions[i]==0. and labels[i][0] == 1.:
            none_none += 1
        elif predictions[i]==0. and labels[i][1] == 1.:
            none_positive += 1
        elif predictions[i]==0. and labels[i][3] == 1.:
            none_negative += 1
        elif predictions[i]==1. and labels[i][0] == 1.:
            positive_none += 1
        elif predictions[i]==1. and labels[i][1] == 1.:
            positive_positive += 1
        elif predictions[i]==1. and labels[i][3] == 1.:
            positive_negative += 1
        elif predictions[i]==3. and labels[i][0] == 1.:
            negative_none += 1
        elif predictions[i]==3. and labels[i][1] == 1.:
            negative_positive += 1
        elif predictions[i]==3. and labels[i][3] == 1.:
            negative_negative += 1
        elif predictions[i]==2. and labels[i][2] == 1.: 
            neutral_neutral += 1
        else:
            other_errors += 1

    r1 = score(positive_positive,positive_none,none_positive,none_none)
    r2 = score(negative_negative,negative_none,none_negative,none_none)
    r3 = score(positive_positive,positive_negative,negative_positive,negative_negative)
    correct = positive_positive + negative_negative + neutral_neutral
    return r1,r2,r3,correct,total,neutral_neutral,other_errors

def score(tt,tf,ft,ff):
    '''
        Compute the precision, recall and f1 score
        Args: 
            tt: predicted true while label is true
            tf: predicted true while label is false
            ft: predicted false while label is true
            ff: predicted false while label is false
        Returns:
            precision: the precision of the model
            recall: the recall of the model
            f1: the f1 score of the model
    '''
    precision = tt/(tt+tf+SMALL_POSITIVE_CONST)
    recall = tt/(tt+ft+SMALL_POSITIVE_CONST)
    f1 = 2*precision*recall/(precision+recall+SMALL_POSITIVE_CONST)
    return precision, recall, f1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        '''
            Returns:
                glove_embedded_sent: the sentence embedded with GloVe
                label: the label of the sentence (One hot encoding of NONE, POS, NEG, NEU)
                cosine: the cosine similarity of the sentence with each cluster
                cluster_labels: the label of each word (ASPECT or NO ASPECT)
        '''
        glove_embedded_sent = torch.tensor(self.dataset[index]['emb']).to(dtype=torch.float32) #[seq_len, 300]
        label = self.dataset[index]['label'].to(dtype=torch.float32) #[seq_len, 4]
        cosine = self.dataset[index]['cosine'].to(dtype=torch.float32) #[seq_len, 2]
        cluster_labels = self.dataset[index]['cluster'].to(dtype=torch.float32) #[seq_len]
        return glove_embedded_sent, label, cosine, cluster_labels
    

def batchify(list_of_samples):
    '''
        Makes the length of each sentence the same by padding with zeros inside the batch
        Returns:
            output: the sentences embedded with GloVe
            labels: the labels of the sentences (One hot encoding of NONE, POS, NEG, NEU)
            cosine_aspects: the cosine similarity of the sentences with each cluster
            cluster_labels: the labels of each word (ASPECT or NO ASPECT)
            lenghts: the lenghts of each sentence (for packing later)
    '''
    #list_of_samples = samples x [len, 300]
    lenghts = []
    for i, (emb,label,aspects,cluster_labels) in enumerate(list_of_samples):
        lenghts.append(label.shape[0])

    max_len = max(lenghts)
    output = torch.zeros(len(list_of_samples), max_len, 300)
    labels = torch.zeros(len(list_of_samples), max_len, 4)
    # Save where the words should be assigned to (cluster)
    cluster_labels = torch.zeros(len(list_of_samples), max_len) # Where each word belongs to (ASPECT or NO ASPECT)
    cosine_aspects = torch.zeros(len(list_of_samples), max_len, 512) # cosine similarity wtr each cluster
    cluster_labels += 3
    for i, (emb,label,aspects,clusters) in enumerate(list_of_samples):
        output[i,:emb.shape[0],:] = emb
        labels[i,:label.shape[0],:] = label 
        #print(aspects.shape)
        cosine_aspects[i,:aspects.shape[0],:] = aspects
        cluster_labels[i,:clusters.shape[0]] = clusters

    output = output.to(device)
    labels = labels.to(device)
    cosine_aspects = cosine_aspects.to(device)
    cluster_labels = cluster_labels.to(device)

    return output, labels, cosine_aspects, cluster_labels, lenghts
    

'''
__________________________________________________________________________________________
EVALUATION FUNCTIONS

'''
def count_errors(preds, labels):
    '''
        Old function used to count the errors of the model 
        (not used anymore)
    '''
    if len(preds) != len(labels):
        raise Exception('Invalid shapes!')
    total = len(preds)
    errors = 0
    for i in range(total):
        if preds[i] != labels[i]:
            errors += 1
    correct = total - errors
    return errors, correct, total


SMALL_POSITIVE_CONST = 1e-4
def evaluate(model, dataset, batch_size=32):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=batchify)
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    f1_posnone = []
    f1_negnone = []
    f1_posneg = []
    
    for i, (emb, label, cosine, cluster_labels, lenghts) in enumerate(dataloader):
        output, aspect = model(emb,cosine, lenghts)
        output = output.reshape(-1,4)
        label = label.reshape(-1,4)
        cluster_labels = cluster_labels.reshape(-1)
        aspect = aspect.reshape(-1,2)
        tt,tf,ft,ff = compare_aspect(aspect, cluster_labels)
        true_pos += tt
        true_neg += ff
        false_pos += tf
        false_neg += ft
        #print(tt,tf,ft,ff)
        r1,r2,r3,correct,total,neutral,other = compare_polarity(output, label)
        f1_posnone.append(r1[2])
        f1_negnone.append(r2[2])
        f1_posneg.append(r3[2])
        #print(r1,r2,r3,neutral,other)
    print(f'F1 POS NONE: {np.mean(f1_posnone)}')
    print(f'F1 NEG NONE: {np.mean(f1_negnone)}')
    print(f'F1 POS NEG: {np.mean(f1_posneg)}')
    p,r,f = score(true_pos,false_pos,false_neg,true_neg)
    print(f'Aspect Detection: Precision: {p}, Recall: {r}, F1: {f}')


def compare_aspect(predictions,labels):
    '''
        Count the errors and the type of errors of the model
        For the aspect detection task
        Args:
            predictions: the predictions of the model (if the word is an aspect or not) 
            labels: the labels (if the word is an aspect or not)
        Returns:
            true_true: the number of words that were predicted as aspect and were actually aspect
            true_false: the number of words that were predicted as aspect but were not aspect
            false_true: the number of words that were predicted as not aspect but were actually aspect
            false_false: the number of words that were predicted as not aspect and were not aspect
    '''
    predictions = torch.argmax(predictions, dim=1)
    assert len(predictions) == len(labels)
    total = len(predictions)
    true_true = 0
    true_false = 0
    false_true = 0
    false_false = 0
    for i in range(total):
        if labels[i] == 3:
            continue
        if predictions[i] != labels[i]:
            #print(f'Predicted {predictions[i]} but was {labels[i]}')
            if predictions[i] == 0.:
                false_true += 1
            else:
                true_false += 1
        else:
            if predictions[i] == 0.:
                false_false += 1
            else:
                true_true += 1
    return true_true, true_false, false_true, false_false

def compare_polarity(predictions,labels):
    '''
        Count the errors and the type of errors of the model
        For the polarity detection task
        Args:
            predictions: the predictions of the model (if the sentence is positive, negative, neutral or none) 
            labels: the labels (if the sentence is positive, negative, neutral or none)
        Returns:
            r1: the precision, recall and f1 score of the model for the positive and none classes
            r2: the precision, recall and f1 score of the model for the negative and none classes
            r3: the precision, recall and f1 score of the model for the positive and negative classes
            correct: the number of aspects that were predicted correctly
            total: the total number of words
            neutral_neutral: the number of aspects that were predicted as neutral and were actually neutral
            other_errors: the number of words that were predicted incorrectly
    '''
    predictions = torch.argmax(predictions, dim=1)
    assert len(predictions) == len(labels)
    total = len(predictions)
    positive_negative = 0
    positive_positive = 0
    negative_positive = 0
    negative_negative = 0
    none_positive = 0
    none_negative = 0
    none_none = 0
    neutral_neutral = 0
    positive_none = 0
    negative_none = 0
    other_errors = 0
    #print(predictions.shape, labels.shape)
    for i in range(total):
        if predictions[i]==0. and labels[i][0] == 1.:
            none_none += 1
        elif predictions[i]==0. and labels[i][1] == 1.:
            none_positive += 1
        elif predictions[i]==0. and labels[i][3] == 1.:
            none_negative += 1
        elif predictions[i]==1. and labels[i][0] == 1.:
            positive_none += 1
        elif predictions[i]==1. and labels[i][1] == 1.:
            positive_positive += 1
        elif predictions[i]==1. and labels[i][3] == 1.:
            positive_negative += 1
        elif predictions[i]==3. and labels[i][0] == 1.:
            negative_none += 1
        elif predictions[i]==3. and labels[i][1] == 1.:
            negative_positive += 1
        elif predictions[i]==3. and labels[i][3] == 1.:
            negative_negative += 1
        elif predictions[i]==2. and labels[i][2] == 1.: 
            neutral_neutral += 1
        else:
            other_errors += 1

    r1 = score(positive_positive,positive_none,none_positive,none_none)
    r2 = score(negative_negative,negative_none,none_negative,none_none)
    r3 = score(positive_positive,positive_negative,negative_positive,negative_negative)
    correct = positive_positive + negative_negative + neutral_neutral
    return r1,r2,r3,correct,total,neutral_neutral,other_errors

def score(tt,tf,ft,ff):
    '''
        Compute the precision, recall and f1 score
        Args: 
            tt: predicted true while label is true
            tf: predicted true while label is false
            ft: predicted false while label is true
            ff: predicted false while label is false
        Returns:
            precision: the precision of the model
            recall: the recall of the model
            f1: the f1 score of the model
    '''
    precision = tt/(tt+tf+SMALL_POSITIVE_CONST)
    recall = tt/(tt+ft+SMALL_POSITIVE_CONST)
    f1 = 2*precision*recall/(precision+recall+SMALL_POSITIVE_CONST)
    return precision, recall, f1

