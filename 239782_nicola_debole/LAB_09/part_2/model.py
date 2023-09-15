import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

from functions import *
from utils import *

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from typing import *
from torch.nn import Parameter
from functools import wraps

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
    
class LSTM_WT(nn.Module):
    def __init__(self, emb_size, output_size, pad_index=0, n_layers=1):
        super(LSTM_WT, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.rnn = nn.LSTM(emb_size, emb_size, n_layers, bidirectional=False)    
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(emb_size, output_size)
        self.output.weight = self.embedding.weight
        
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
    
class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """
    def __init__(self, dropout: float, batch_first: Optional[bool]=False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x

class LSTM_WITH_VAR_DROP(nn.LSTM):
    def __init__(self, *args, dropouti: float=0.,
                 dropoutw: float=0., dropouto: float=0.,
                 batch_first=True, unit_forget_bias=True, **kwargs):
        super().__init__(*args, **kwargs, batch_first=batch_first)
        self.unit_forget_bias = unit_forget_bias
        self.dropoutw = dropoutw
        self.input_drop = VariationalDropout(dropouti,
                                             batch_first=batch_first)
        self.output_drop = VariationalDropout(dropouto,
                                              batch_first=batch_first)
        self._init_weights()

    def _init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        """
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name and self.unit_forget_bias:
                nn.init.zeros_(param.data)
                param.data[self.hidden_size:2 * self.hidden_size] = 1

    def _drop_weights(self):
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                getattr(self, name).data = \
                    torch.nn.functional.dropout(param.data, p=self.dropoutw,
                                                training=self.training).contiguous()

    def forward(self, input, hx=None):
        self._drop_weights()
        input = self.input_drop(input)
        seq, state = super().forward(input, hx=hx)
        return self.output_drop(seq), state
    

class LSTM_LM_VAR_DROP(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LSTM_LM_VAR_DROP, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.rnn = LSTM_WITH_VAR_DROP(emb_size, hidden_size, n_layers, dropouti=0.4, dropouto=0.2, dropoutw=0, bidirectional=False)    
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