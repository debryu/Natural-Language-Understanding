import numpy as np
from tqdm import tqdm 
import torch


GloVe_embeddings = {}
print("Loading GloVe embeddings...")
with open("dataset/LAB11_part2/glove.6B.300d.txt", 'r') as f:
    total_lines = 400000
    for line in tqdm(f,total=total_lines):
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        GloVe_embeddings[word] = vector
print("Done.")

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
        self.beta = torch.nn.Parameter(torch.ones((self.size)), requires_grad=True)
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
    def __init__(self, n_clusters = 128, hidden = 256):
        super(ASPECT_DETECTION, self).__init__()
        self.n_clusters = n_clusters
        self.n_classes = n_clusters+1
        self.first_layer = MReLU(size=n_clusters)
        
        self.hidden_layer = torch.nn.Linear(n_clusters, hidden)
        self.hidden1 = torch.nn.Linear(hidden, 1024)
        self.hidden2 = torch.nn.Linear(1024, 1024)
        self.hidden3 = torch.nn.Linear(1024, hidden)
        #self.droput = torch.nn.Dropout(p=0.2)
        self.final_layer = torch.nn.Linear(hidden, 2)
        self.activation = torch.nn.ReLU()
        # One class is the "no aspect" class
        self.logits = torch.nn.Softmax(dim=2)

    def forward(self, input):
        # Input size is [batch, seq_len, 300]
        batch_size = input.shape[0]
        sent_len = input.shape[1]
        #print(batch_size, sent_len)
        #print(batch_size*sent_len)
        #print(input.shape)
        input = input.reshape(-1, self.n_clusters)
        input = self.first_layer(input)
        input = self.hidden_layer(input)
        input = self.activation(input)
        input = self.hidden1(input)
        input = self.activation(input)
        input = self.hidden2(input)
        input = self.activation(input)
        input = self.hidden3(input)
        input = self.activation(input)
        #input = self.droput(input)
        output = self.final_layer(input)
        #print(output.shape) 
        output = output.reshape(batch_size, sent_len, 2)
        #print(output.shape)
        output = self.logits(output)
        return output


class ABSA(torch.nn.Module):
    def __init__(self, classes = 4, embedding_dim = 300):
        super(ABSA, self).__init__()
        self.lstm_compression = LSTM(input_size=300, hidden_size=1, nl=10)
        self.lstm_decoder = LSTM(input_size=4, hidden_size=32, nl=5)
        #self.aspect_detection = AspectDetection()
        self.classifier = torch.nn.Linear(64, classes)
        self.logits = torch.nn.Softmax(dim=1)
        self.dropouLayer = torch.nn.Dropout(p=0.2)

    def forward(self, input):
        aspect = self.aspect_detection(input) #[batch_size, seq_len, 2]
        input = self.dropouLayer(input) #[batch_size, seq_len, 300]
        compressed_input, (h_n, c_n) = self.lstm_compression(input) #[batch_size, seq_len, 2]
        input = torch.cat((compressed_input, aspect), dim=2) #[batch_size, seq_len, 302]
        output, (h_n, c_n) = self.lstm_decoder(input)
        #print(output.shape)
        output = self.classifier(output)
        output = self.logits(output)
        return output, aspect
