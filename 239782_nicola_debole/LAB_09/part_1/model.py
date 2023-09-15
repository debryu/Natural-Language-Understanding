import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

from functions import *

class basic_LM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(basic_LM_RNN, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False)    
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.rnn(emb)
        output = self.output(rnn_out).permute(0,2,1)
        return output
    
    def get_word_embedding(self, token):
        return self.embedding(token).squeeze(0).detach().cpu().numpy()
    
    def get_most_similar(self, vector, top_k=10):
        embs = self.embedding.weight.detach().cpu().numpy()
        #Our function that we used before
        scores = []
        for i, x in enumerate(embs):
            if i != self.pad_token:
                scores.append(cosine_similarity(x, vector))
        # Take ids of the most similar tokens 
        scores = np.asarray(scores)
        indexes = np.argsort(scores)[::-1][:top_k]  
        top_scores = scores[indexes]
        return (indexes, top_scores) 
    
class LSTM_LM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LSTM_LM_RNN, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)    
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.rnn(emb)
        output = self.output(rnn_out).permute(0,2,1)
        return output
    
    def get_word_embedding(self, token):
        return self.embedding(token).squeeze(0).detach().cpu().numpy()
    
    def get_most_similar(self, vector, top_k=10):
        embs = self.embedding.weight.detach().cpu().numpy()
        #Our function that we used before
        scores = []
        for i, x in enumerate(embs):
            if i != self.pad_token:
                scores.append(cosine_similarity(x, vector))
        # Take ids of the most similar tokens 
        scores = np.asarray(scores)
        indexes = np.argsort(scores)[::-1][:top_k]  
        top_scores = scores[indexes]
        return (indexes, top_scores) 
    
class LSTM_LM_DO(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.2,
                 emb_dropout=0.4, n_layers=1):
        super(LSTM_LM_DO, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.dropout_emb = nn.Dropout(emb_dropout)
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)    
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        self.dropout_out = nn.Dropout(out_dropout)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.dropout_emb(emb)
        rnn_out, _  = self.rnn(emb)
        output = self.output(rnn_out)
        output = self.dropout_out(output)
        output = output.permute(0,2,1)
        return output
    
    def get_word_embedding(self, token):
        return self.embedding(token).squeeze(0).detach().cpu().numpy()
    
    def get_most_similar(self, vector, top_k=10):
        embs = self.embedding.weight.detach().cpu().numpy()
        #Our function that we used before
        scores = []
        for i, x in enumerate(embs):
            if i != self.pad_token:
                scores.append(cosine_similarity(x, vector))
        # Take ids of the most similar tokens 
        scores = np.asarray(scores)
        indexes = np.argsort(scores)[::-1][:top_k]  
        top_scores = scores[indexes]
        return (indexes, top_scores) 
    