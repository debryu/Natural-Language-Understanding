from transformers import BertTokenizer, BertModel
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
# The tokenizer will not be used in this case
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained("bert-base-uncased")

embedding_dim = 768 # Embedding dimension for BERT


# Define the model
class Bert_Intent_Slot(nn.Module):

    def __init__(self, out_slot, out_int, vocab_len, pad_index=0):
        super(Bert_Intent_Slot, self).__init__()
        '''
            args:
                out_slot: number of slots (output size for slot filling)
                out_int: number of intents (ouput size for intent class)
                vocab_len: vocabulary size
                pad_index: index of the padding token
        '''
        self.bert = bert_model
        # Model head for slot filling
        self.slot_out = nn.Linear(embedding_dim, out_slot)
        # Model head for intent classification
        self.intent_out = nn.Linear(embedding_dim, out_int)
        self.dropout = nn.Dropout(0.2)
        
        
    def forward(self, input, seq_lengths):
        # Create the attention mask
        attention_mask = (input != 0).float()
        # Process the batch
        # And return the last hidden state
        output = self.bert(input_ids = input, attention_mask = attention_mask, output_attentions = False, output_hidden_states = True).last_hidden_state
        output = self.dropout(output)   
        # Compute slot logits from the last hidden state
        slots = self.slot_out(output[:,1:,:])

        # Get the CLS token
        CLS_token = output[:,0,:]
        # Compute intent logits from the CLS token
        intent = self.intent_out(CLS_token)
        
        # Slot shape: batch, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot shape: batch_size, classes, seq_len
        
        return slots, intent