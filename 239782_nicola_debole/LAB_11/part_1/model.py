from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
# The tokenizer will not be used in this case
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained("bert-base-uncased")
embedding_dim = 768


longformer_tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')
longformer_model = AutoModel.from_pretrained("allenai/longformer-base-4096")
embedding_dim = 768

import math
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer, VaderConstants

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Define the model for subjectivity
class Subjectivity(nn.Module):

    def __init__(self, vocab_len, pad_index=0):
        super(Subjectivity, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (ouput size for intent class)
        # emb_size = word embedding size
        self.bert = bert_model
        self.pad_index = pad_index
        self.lastLayer = nn.Linear(embedding_dim, 2)
        # Dropout layer How do we apply it?
        #self.dropout = nn.Dropout(0.2)
        
    def forward(self, input):
        # Create the attention mask
        attention_mask = (input != self.pad_index).float()
        #print(input.shape)
        #print(input[0])
        
        #print('max len',torch.max(seq_lengths))
        # Process the batch
        # And return the last hidden state
        output = self.bert(input_ids = input, attention_mask = attention_mask, output_attentions = False, output_hidden_states = True).last_hidden_state
        #print(output.shape)
        #print(output[0])
        
        # Get the CLS token
        CLS_token = output[:,0,:]
        # Compute the class logits
        result = self.lastLayer(CLS_token)
        
        return result
    

class DocumentPolarity():
    def __init__(self):
        self.analizer = SentimentIntensityAnalyzer()

    def predict_polarity(self, document):
        for sentence in document:
            print(sentence)
            print(self.analizer.polarity_scores(sentence))

    def predict_polarity_batch(self, batch):
        results = []
        for document in batch:
            results.append(self.analizer.polarity_scores(document))
        return results


class Polarity(nn.Module):
    def __init__(self, pad_index=0):
        super(Polarity, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (ouput size for intent class)
        # emb_size = word embedding size
        self.feature_encoder = nn.Linear(40, 64)
        self.feature_FFN = nn.Linear(64, 5)
        self.classifier = nn.Linear(10, 2)
        self.simple_classifier = nn.Linear(5, 2)
        self.activation = nn.Sigmoid()
        # Dropout layer How do we apply it?
        #self.dropout = nn.Dropout(0.2)
        
    def forward(self, input):
        
        # Innput should be of shape
        # [batch, n_max_sentences, 5]
        number_of_sentences = input.shape[1]

        result = input.type(torch.FloatTensor).to(device)
        
        # Compute the average document embedding
        average_document_emb = torch.mean(result, dim = 1)
        
        # The shape is now [batch, 64]

        # Get the rounded by defect number of sentences/3
        index = number_of_sentences/8

        # Compute the average embedding for the beginning, mid section and end of the document
        # Basically splitting the review in 8 parts and computing the average embedding for each part
        average_1 = torch.mean(result[:,0:math.floor(index),:], dim = 1)
        average_2 = torch.mean(result[:,math.floor(1*index):math.floor(2*index),:], dim = 1)
        average_3 = torch.mean(result[:,math.floor(2*index):math.floor(3*index),:], dim = 1)
        average_4 = torch.mean(result[:,math.floor(3*index):math.floor(4*index),:], dim = 1)
        average_5 = torch.mean(result[:,math.floor(4*index):math.floor(5*index),:], dim = 1)
        average_6 = torch.mean(result[:,math.floor(5*index):math.floor(6*index),:], dim = 1)
        average_7 = torch.mean(result[:,math.floor(6*index):math.floor(7*index),:], dim = 1)
        average_8 = torch.mean(result[:,math.floor(7*index):,:], dim = 1)
       
        # Every average has shape [batch, 64]

        # Concatenate the three embeddings and the average document embedding
        concatenated = torch.cat((average_1, average_2, average_3, average_4, average_5, average_6, average_7, average_8), dim = 1)
        
        encoded = self.feature_encoder(concatenated)
        
        encoded = self.activation(encoded)
        # decode the concatenated embeddings to a single embedding
        document_emb = self.feature_FFN(encoded)
        document_emb = self.activation(document_emb)
        

        final_embs = torch.cat((average_document_emb,document_emb), dim = 1)
        # Compute the class logits
        result = self.classifier(final_embs)
        result = self.simple_classifier(average_document_emb)
        # Final dimension is [batch_size, 2]
        
        return result
    
class PolarityLongformer(nn.Module):
    def __init__(self, pad_index=0):
        super(PolarityLongformer, self).__init__()
        self.pad_index = pad_index
        self.longformer = longformer_model.to(device)
        self.encoder = nn.Linear(2*768, 768)
        self.classifier = nn.Linear(768, 2)
        self.activation = nn.Sigmoid()
        # Dropout layer How do we apply it?
        #self.dropout = nn.Dropout(0.2)
        
    def forward(self, input):
        input = input.to(device)
        # Get the attention mask
        attention_mask = (input != self.pad_index).float().to(device)
        # Compute the embedding using the longformer
        embeddings = self.longformer(input_ids = input, attention_mask = attention_mask, output_attentions = False, output_hidden_states = True).last_hidden_state
        #print(embeddings.shape)
        #print(embeddings)

        # Compute the average embedding 
        mean_embedding = torch.mean(embeddings[:,1:,:], dim = 1)
        #print('ME',mean_embedding)
        # Get the CLS token
        CLS_token = embeddings[:,0,:]
        # Compute the class logits
        #result = self.classifier(CLS_token)
        result = self.classifier(mean_embedding)
        #result = self.activation(result)
        
        return result, CLS_token.squeeze(0)
