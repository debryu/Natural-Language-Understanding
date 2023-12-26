# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

'''
Create the dataset class
'''
class Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_clusters = 128
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        #print(self.dataset[index]['emb'].shape)
        glove_embedded_sent = torch.tensor(self.dataset[index]['emb']).to(dtype=torch.float32) #[seq_len, 300]
        label = self.dataset[index]['label'].to(dtype=torch.float32) #[seq_len, 4]
        cosine = self.dataset[index]['cosine'].to(dtype=torch.float32) #[seq_len, 2]
        cluster_labels = self.dataset[index]['cluster'].to(dtype=torch.float32) #[seq_len]
        return glove_embedded_sent, label, cosine, cluster_labels
    

def batchify(list_of_samples):
    #list_of_samples = samples x [len, 300]
    lenghts = []
    for i, (emb,label,aspects,cluster_labels) in enumerate(list_of_samples):
        lenghts.append(label.shape[0])

    max_len = max(lenghts)
    output = torch.zeros(len(list_of_samples), max_len, 300)
    labels = torch.zeros(len(list_of_samples), max_len, 4)
    # Save where the words should be assigned to (cluster)
    cluster_labels = torch.zeros(len(list_of_samples), max_len) # Where each word belongs to (ASPECT or NO ASPECT)
    cosine_aspects = torch.zeros(len(list_of_samples), max_len, 128) # cosine similarity wtr each cluster
    cluster_labels += 3
    for i, (emb,label,aspects,clusters) in enumerate(list_of_samples):
        output[i,:emb.shape[0],:] = emb
        labels[i,:label.shape[0],:] = label 
        #print(aspects.shape)
        cosine_aspects[i,:aspects.shape[0],:] = aspects
        cluster_labels[i,:clusters.shape[0]] = clusters

    return output, labels, cosine_aspects, cluster_labels, lenghts
    
def count_errors(preds, labels):
    #print(len(preds), len(labels))
    if len(preds) != len(labels):
        raise Exception('Invalid shapes!')
    total = len(preds)
    errors = 0
    for i in range(total):
        if preds[i] != labels[i]:
            errors += 1
    correct = total - errors
    return errors, correct, total