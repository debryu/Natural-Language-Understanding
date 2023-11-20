# Add functions or classes used for data loading and preprocessing

# Download the dataset
import nltk
import numpy
#nltk.download("subjectivity")
#nltk.download("movie_reviews")
from nltk.corpus import movie_reviews
from nltk.corpus import subjectivity
from sklearn.model_selection import KFold
mr = movie_reviews
PAD_TOKEN = 0
import torch
import torch.utils.data as data
from model import bert_tokenizer
from model import longformer_tokenizer
from main import device
from nltk.sentiment.vader import SentimentIntensityAnalyzer, VaderConstants
from main import Subjectivity
from tqdm import tqdm
import networkx as nx


vocab_len = len(bert_tokenizer.vocab)

print("Creating the categories...")
# Get the categories fot both the datasets
sent_obj = subjectivity.sents(categories='obj')
sent_subj = subjectivity.sents(categories='subj')
rev_neg = mr.paras(categories='neg')
rev_pos = mr.paras(categories='pos')

# Create the labels
# 0 for objective and 1 for subjective
labels_subjectivity = numpy.array([0] * len(sent_obj) + [1] * len(sent_subj))
# 0 for negative and 1 for positive
labels_polarity = numpy.array([0] * len(rev_neg) + [1] * len(rev_pos))

# Merge the datasets in just two lists, one for polarity and one for subjectivity
subjectivity_ds = sent_obj + sent_subj
polarity_ds = rev_neg + rev_pos

kfold = KFold(n_splits=10, random_state=None, shuffle = True)

def association(i,j,T):
    if (i-j) <= T:
        return 0
    else:
        return 1/(i-j)**2
def generate_graph(ds,logits):
    #print(len(ds),len(logits))
    # Create a node for each sentence
    # And add it to the graph
    G = nx.Graph()
    G.add_node("0", label = 'objective')
    G.add_node("1", label = 'subjective')
    for i,sent in enumerate(ds):
        G.add_node(i+2, label=sent)
    # Now we add the edges
    for i,sent1 in enumerate(ds):
        G.add_edge(f"node{i}","0", capacity=logits[i][0])
        G.add_edge(f"node{i}","1", capacity=logits[i][1])
        for j,sent2 in enumerate(ds):
            G.add_edge(i,j, capacity=association(i,j,5))

    return G


# Subjectivity dataset class
class SubjectivityDataset(data.Dataset):
    def __init__(self, dataset,labels):
        self.sentences = dataset
        self.labels = labels
        self.tokenized_sentences = []
        self.lengths = []

        for sent in self.sentences:
            self.lengths.append(len(sent))
            tok_sent = bert_tokenizer.encode(sent, return_tensors="pt", add_special_tokens=True).squeeze(0)
            self.tokenized_sentences.append(tok_sent)


    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sample = {'sentence': self.sentences[idx], 'label': self.labels[idx], 'tok_sent': self.tokenized_sentences[idx], 'len': self.lengths[idx]}
        return sample
    

def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths
    data.sort(key=lambda x: x['len'], reverse=True) 
    
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    tok_sent_padded, _ = merge(new_item['tok_sent'])
    new_item["tok_sent_padded"] = tok_sent_padded.to(device)
    
    return new_item


# Polarity dataset class
class PolarityDataset(data.Dataset):
    def __init__(self, dataset,labels, remove_subj = True):
        self.all_movie_reviews = dataset
        self.labels = labels
        self.documents = []
        self.documents_lengths = []
        self.documents_scores = []
        self.subjectivity_scores = []  

        '''
            This two are the same, one in clear format and one in the tokenized version
            Also we have the list of lengths of the tokenized version
        '''
        self.stringed_documents = []
        self.tokenized_reviews = []
        self.tokenized_reviews_lengths = []
        '''------------------------'''

        self.analizer = SentimentIntensityAnalyzer()
        #print(len(self.all_movie_reviews))
        removed_sentences = 0
        progress_bar = tqdm(total=len(self.all_movie_reviews))
        for review in self.all_movie_reviews:        
            review_length = 0
            document = []
            stringed_sents = []

            #print(len(review))
            for sent in review:
                sentence = []
                # Skip all the sentences with less than 3 words
                # We found that in the datasets there are some "sentences"
                # with just one (usually punctuation)
                if(len(sent) < 2):
                    continue

                # Join the words in a sentence
                # And append it to the list of stringed sentences
                # This is used to tokenize the entire document to then use the longformer model
                stringed_sent = ' '.join(sent)   
                stringed_sents.append(stringed_sent)
                for word in sent:
                    sentence.append(word)
                document.append(sentence)
                review_length += 1
                #tok_sent = bert_tokenizer.encode(sent, return_tensors="pt", add_special_tokens=True).squeeze(0)
                #print('tok sent', tok_sent)
                #tokenized_review.append(tok_sent)
            '''
                # Join the sentences in a document
                # Then do some basic reformatting of the string
                Then add everything to two different lists, one with the stringed documents and one with the tokenized documents
                This is used for the longformer model
            '''
            stringed_document = '\n'.join(stringed_sents)
            # Clean the string removing all the useless spaces 
            stringed_document = clean_string(stringed_document)
            self.stringed_documents.append(stringed_document)
            tokenized_stringed_document = longformer_tokenizer.encode(stringed_document, return_tensors="pt", add_special_tokens=True).squeeze(0)
            self.tokenized_reviews.append(tokenized_stringed_document)
            self.tokenized_reviews_lengths.append(len(tokenized_stringed_document))
            '''----------------------------------------------------------------------------------------------------------------'''
            #print('tok rev', len(tokenized_review))
            #self.tokenized_reviews.append(tokenized_review)
            self.documents_lengths.append(review_length)
            self.documents.append(document)
            progress_bar.update(1)
        
        ''' 
                Compute the Vader polarity scores for each sentence in each document 
                And append them to the list of documents scores
        '''
        for document in self.documents:
            score = []
            for sent in document:
                joined_sent = ' '.join(sent)
                score.append(self.analizer.polarity_scores(joined_sent))
            self.documents_scores.append(score)
            self.documents_lengths.append(len(score))
        '''----------------------------------------------------------------------------------------------------------------'''
        
        #print(len(self.tokenized_reviews))


    def __len__(self):
        return len(self.all_movie_reviews)

    def __getitem__(self, idx):
        sample = { 'corpus': self.all_movie_reviews[idx],
                   'label': self.labels[idx],         
                   'stringed_document': self.stringed_documents[idx],
                   'document_scores': self.documents_scores[idx],
                   'document_lengths': self.documents_lengths[idx],
                   'stringed_reviews': self.stringed_documents[idx],
                   'tokenized_reviews': self.tokenized_reviews[idx],
                   'tokenized_reviews_lengths': self.tokenized_reviews_lengths[idx],
                   }
        return sample

def collate_vader_fn(data):
    def merge(batch_of_reviews, lenghts):
        # First identify the maximum number of a sents in a batch
        max_len = max(lenghts)
        min_len = min(lenghts)  
        # Now we can create a tensor with dimension 
        # [batch_size, max_len, 5]
        # Where 5 is the number of features of the VADER model for each sentence
        padded_feats = torch.LongTensor(len(batch_of_reviews),max_len,5).fill_(PAD_TOKEN)

        for i, list_of_scores in enumerate(batch_of_reviews):
            #print(review)
            for j,score in enumerate(list_of_scores):
                features = torch.Tensor([score['neg'], score['neu'], score['pos'], score['compound'], score['compound']])
                padded_feats[i, j, :] = features # We copy each sequence into the matrix

        padded_feats = padded_feats.detach()  # We remove these tensors from the computational graph
        return padded_feats, max_len
    # Sort data by seq lengths
    data.sort(key=lambda x: x['document_lengths'], reverse=True) 
    
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    features, _ = merge(new_item['document_scores'], new_item['document_lengths'])
    new_item["features"] = features.to(device)
    #print(len(new_item))
    return new_item  



def collate_longformer_fn(data):
    def merge(batch_of_reviews, lenghts):
        # First identify the maximum number of a sents in a batch
        max_len = max(lenghts)
        min_len = min(lenghts)  
        # Now we can create a tensor with dimension 
        # [batch_size, max_len, 5]
        # Where 5 is the number of features of the VADER model for each sentence
        padded_tokens = torch.LongTensor(len(batch_of_reviews),max_len).fill_(PAD_TOKEN)

        for i, review in enumerate(batch_of_reviews):
            end = lenghts[i]
            padded_tokens[i, :end] = review

        padded_tokens = padded_tokens.detach()  # We remove these tensors from the computational graph
        return padded_tokens, max_len
    # Sort data by seq lengths
    data.sort(key=lambda x: x['document_lengths'], reverse=True) 
    
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
    # We just need one length for packed pad seq, since len(utt) == len(slots)

    features, _ = merge(new_item['tokenized_reviews'], new_item['tokenized_reviews_lengths'])
    new_item['embeddings'] = features.to(device)
    #print(len(new_item))
    return new_item 
    

def collate_rev_fn(data):
    def merge(batch_of_reviews):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        #print(len(batch_of_reviews))
        
        review_lengths = [len(review) for review in batch_of_reviews]
        sents_lengths = []
        for review in batch_of_reviews:
            for sent in review:
                sents_lengths.append(len(sent))
        max_len = 1 if max(sents_lengths)==0 else max(sents_lengths)
        max_sents = 1 if max(review_lengths)==0 else max(review_lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(batch_of_reviews),max_sents,max_len).fill_(PAD_TOKEN)

        # Add the sentence counter just to keep track of the index for the list sent_lengths
        sent_counter = 0
        for i, review in enumerate(batch_of_reviews):
            #print(review)
            for j,sent in enumerate(review):
                #print(sent)
                #print(len(sent))
                
                #end_rev = review_lengths[i]
                end_sent = sents_lengths[sent_counter]
                #print('end sent', end_sent)
                padded_seqs[i, j, :end_sent] = sent # We copy each sequence into the matrix
                sent_counter += 1
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, sents_lengths
    # Sort data by seq lengths
    data.sort(key=lambda x: max(x['len']), reverse=True) 
    
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    tok_corpus_padded, _ = merge(new_item['tok_corpus'])
    new_item["tok_corpus_padded"] = tok_corpus_padded.to(device)
    
    return new_item

'''
 This is used to do 4 things: 
  1) Clean and return the original dataset by removing all the sentences with less than 2 words
  2) Create and return another version of the dataset in which objective sentences are removed
     this is used to evaluate the difference using the longformer model
  3) Return the logits of the subjectivity model for each sentence
'''
def subjectivity_extraction(ds,fold_number):
    # First load the correct subjectivity model
    sub_model = Subjectivity(vocab_len, pad_index=PAD_TOKEN).to(device)
    sub_model.load_state_dict(torch.load(f'LAB_11/part_1/models/model_multi_lr_FOLD{fold_number}.pt'))
    sub_model.eval()
    removed_sentences = 0
    progress_bar = tqdm(total=len(ds))
    new_ds = []
    original_ds = []
    list_of_logits = []
    for document in ds:
        new_document = []
        batch_of_sentences = []
        batch_of_tok_sentences = []
        tok_len = []
        for list_of_words in document:
            if len(list_of_words) < 2:
                continue
            sent = clean_string(' '.join(list_of_words))
            batch_of_sentences.append(sent)
            sent = bert_tokenizer.encode(sent, return_tensors="pt", add_special_tokens=True)
            tok_len.append(sent.shape[1])
            # Store the sentences (as a list) 
            batch_of_tok_sentences.append(sent)
        # Add back the list of sentences to the original dataset
        original_ds.append(batch_of_sentences)
        #print(batch_of_sentences)
        # Create the input tensor for the model initialized as PAD_TOKEN
        input = torch.LongTensor(len(batch_of_sentences),max(tok_len)).fill_(PAD_TOKEN)
        # Fill the input tensor with all the sentences
        for i,sent in enumerate(batch_of_tok_sentences):
            end = sent.shape[1]
            input[i, :end] = sent
        
        #print(input.shape)
        input = input.to(device)
        result = sub_model(input).to(device)
        # Compute the logits for each sentence
        subjectivity_scores = torch.nn.functional.softmax(result,dim=1).to('cpu').detach().numpy()
        # Save the logits of the document in a list with all the documents (logits)
        list_of_logits.append(subjectivity_scores)
        result = result.argmax(dim=1).to('cpu').numpy()
        
        # If the sentence is objective we skip it
        for i, prediction in enumerate(result):
            if(prediction == 0):
                removed_sentences += 1
                progress_bar.set_description(f"Objective sentences detected: {removed_sentences}")
                continue
            new_document.append(batch_of_sentences[i])
        new_ds.append(new_document)
        progress_bar.update(1)
        
    return new_ds, original_ds, list_of_logits


def augment_dataset(ds,logits):
    aug_ds = []
    aug_str_ds = []
    for i,doc in enumerate(tqdm(ds)):
        #print('lenghts',len(doc),len(logits[i]))
        G = generate_graph(doc, logits[i])
        cut_value, partition = nx.minimum_cut(G, "0", "1")
        reachable, non_reachable = partition
        indexes = []
        for node in non_reachable:
            if node == '1':
                continue
            index = int(node.split('node')[1])
            indexes.append(index)
        indexes = sorted(set(indexes))
        subset = [ds[i][ind] for ind in indexes]
        aug_ds.append(subset)
        aug_str_ds.append('\n'.join(subset))
    # Return the augmented dataset in two formats:
    # 1) A list of lists of sentences
    # 2) A list of strings (each string is a document)
    return aug_ds, aug_str_ds

def clean_string(s):
    s = s.replace(' :', ':')
    s = s.replace(" '", "'")
    s = s.replace("' ", "'")
    s = s.replace(' .', '.')
    s = s.replace(' ,', ',')
    s = s.replace(' ?', '?')
    s = s.replace(' !', '!')
    s = s.replace(' - ', '-')
    s = s.replace(' ( ', ' (')
    s = s.replace(' ) ', ') ')
    s = s.replace(' )', ')')
    return s
