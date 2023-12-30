import numpy as np
from tqdm import tqdm 
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM(torch.nn.Module):
    def __init__(self, input_size = 300, hidden_size = 300, nl = 3):
        super(LSTM, self).__init__()
        self.num_layers = nl
        self.final_hidden_size = hidden_size*2
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=self.num_layers, bias=True, batch_first=True, dropout=0.1, bidirectional=True, proj_size=0, device=device, dtype=torch.float32)
    def forward(self, input):
        return self.lstm(input)
    

class MReLU(torch.nn.Module):
    def __init__(self, size = 128):
        super(MReLU, self).__init__()
        self.size = size
        self.alpha = torch.nn.Parameter(torch.ones((self.size)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.activation = torch.nn.ReLU()
    def forward(self, input):
        # The shape of the input is [batch, size]
        # Multiply each similarity score by the corresponding alpha
        input = torch.einsum('ea,a->ea', input, self.alpha)
        # Add the corresponding beta
        #input = input + self.beta
        
        # now we have to apply the max function
        input = self.activation(input)
        return input


'''
TO RECAP:
    - The dataset contains a batch of sentences
    - Each sentence is a list of words
    - Each word is represented as a vector (from GloVe)
    - Each word has a label (0,1,2,3)
    - The label 0 means that the word is not an aspect, 1 that is positive, 2 that is neutral, 3 that is negative
    - The dataset also contains the cosine similarity between each word and each cluster
    - The dataset also contains the cluster to which each word belongs to and can be either 0 (no cluster) or 1,2,3,...,128


'''
class ASPECT_DETECTION(torch.nn.Module):
    def __init__(self, n_clusters = 512, hidden = 256):
        super(ASPECT_DETECTION, self).__init__()
        self.n_clusters = n_clusters
        self.n_classes = n_clusters+1
        self.first_layer = MReLU(size=n_clusters)
        
        self.hidden_layer = torch.nn.Linear(n_clusters, hidden)
        self.hidden1 = torch.nn.Linear(hidden, 1024)
        self.hidden2 = torch.nn.Linear(1024, 1024)
        self.hidden3 = torch.nn.Linear(1024, hidden)
        self.droput = torch.nn.Dropout(p=0.1)
        self.final_layer = torch.nn.Linear(hidden, 2)
        self.activation = torch.nn.ReLU()
        # One class is the "no aspect" class
        self.logits = torch.nn.Softmax(dim=2)

        self.test1=torch.nn.Linear(1, 10)
        self.test2=torch.nn.Linear(10, 10)
        self.test3=torch.nn.Linear(10, 2)

    def forward(self, input):
        # Input size is [batch, seq_len, 300]
        batch_size = input.shape[0]
        sent_len = input.shape[1]
        input = input.reshape(-1, self.n_clusters)
        input = self.first_layer(input)
        input = self.hidden_layer(input)
        input = self.activation(input)
        input = self.hidden1(input)
        input = self.activation(input)
        input = self.hidden2(input)
        input = self.activation(input)
        input = self.droput(input)
        input = self.hidden3(input)
        input = self.activation(input)
        output = self.final_layer(input)
      
        output = output.reshape(batch_size, sent_len, 2)
        output = self.logits(output)
        return output


class ABSA(torch.nn.Module):
    def __init__(self, classes = 4, n_clusters = 512):
        super(ABSA, self).__init__()
        self.lstm_encoder = LSTM(input_size=300, hidden_size=1024, nl=5)
        self.lstm_compressor = torch.nn.Sequential(
            torch.nn.Linear(1024*2, 2),
            torch.nn.ReLU(),
        )
        self.lstm_bottle = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
        )
        self.elaborate = torch.nn.Sequential(
            torch.nn.Linear(66, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
        )
        self.lstm_decoder = LSTM(input_size=1024, hidden_size=32, nl=5)
        self.aspect_detection_module = ASPECT_DETECTION(n_clusters=n_clusters)
        #self.aspect_detection = AspectDetection()
        self.classifier1 = torch.nn.Linear(32, classes)
        self.classifier2 = torch.nn.Linear(1024, classes)
        self.logits = torch.nn.Softmax(dim=1)
        self.dropouLayer = torch.nn.Dropout(p=0.1)

    def forward(self, input, cosine, len):
        aspect = self.aspect_detection_module(cosine) #[batch_size, seq_len, 2]
        #input = self.dropouLayer(input) #[batch_size, seq_len, 300]
        input = pack_padded_sequence(input, lengths=len, batch_first=True, enforce_sorted=False)
        input, (h_n, c_n) = self.lstm_encoder(input) #[batch_size, seq_len, 2]
        pad_input, _ = pad_packed_sequence(input, batch_first=True)
        compressed_input = self.lstm_compressor(pad_input)
        #print(compressed_input.shape)
        #print(aspect.shape)
        output1 = aspect*compressed_input
        input = self.lstm_bottle(pad_input)
        input = pack_padded_sequence(input, lengths=len, batch_first=True, enforce_sorted=False)
        output2, (h_n, c_n) = self.lstm_decoder(input)
        output2, _ = pad_packed_sequence(output2, batch_first=True)
        output = torch.cat((output1, output2), dim=2)
        output = self.elaborate(output)
        output = self.classifier2(output)
        output = self.logits(output)
        return output, aspect
