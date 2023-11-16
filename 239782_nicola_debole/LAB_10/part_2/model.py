from transformers import BertTokenizer, BertModel
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
# The tokenizer will not be used in this case
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained("bert-base-uncased")
embedding_dim = 768


# Define the model
class Bert_Intent_Slot(nn.Module):

    def __init__(self, out_slot, out_int, vocab_len, pad_index=0):
        super(Bert_Intent_Slot, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (ouput size for intent class)
        # emb_size = word embedding size
        self.bert = bert_model
        self.slot_out = nn.Linear(embedding_dim, out_slot)
        self.intent_out = nn.Linear(embedding_dim, out_int)
        # Dropout layer How do we apply it?
        #self.dropout = nn.Dropout(0.2)
        
    def forward(self, input, seq_lengths):
        # Create the attention mask
        attention_mask = (input != 0).float()
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
        # Compute slot logits
        #print('encoded utt', utt_encoded.shape)
        slots = self.slot_out(output[:,1:,:])
        # Compute intent logits
        intent = self.intent_out(CLS_token)
        
        # Slot size: seq_len, batch size, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        #print(slots.shape)
        return slots, intent