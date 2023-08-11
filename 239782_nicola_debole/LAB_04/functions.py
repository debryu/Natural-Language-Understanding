# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import math
from typing import Any
import nltk
# download treebank
#nltk.download('treebank')
from nltk.corpus import treebank
# rule-based tagging
from nltk.tag import RegexpTagger
from nltk.metrics.scores import accuracy as accuracy_nltk

# un-comment the lines below, if you get 'ModuleNotFoundError'
import en_core_web_sm

aug_rules = [
    (r'(in|In|among|Among|of|Of|above|Above)$', 'ADP'),   # prepositions
    (r'(To|to|Well|well|Up|up|Now|now|Not|not)$', 'PRT'),   # particles
    (r'(I|i|you|You|he|He|She|she|it|It|they|They|we|We)$', 'PRON'),   # pronouns
    (r'(and|And|or|Or|but|But|while|While|when|When|since|Since)$', 'CONJ'),   # conjunctions
    (r'^-?[0-9]+(.[0-9]+)?$', 'NUM'),   # cardinal numbers
    (r'(The|the|A|a|An|an)$', 'DET'),   # articles
    (r'.*able$', 'ADJ'),                # adjectives
    (r'.*ness$', 'NOUN'),               # nouns formed from adjectives
    (r'.*ly$', 'ADV'),                  # adverbs
    (r'.*s$', 'NOUN'),                  # plural nouns
    (r'.*ing$', 'VERB'),                # gerunds
    (r'.*ed$', 'VERB'),                 # past tense verbs
    (r'.*ed$', 'VERB'),                 # past tense verbs
    (r'[\.,!\?:;\'"]', '.'),            # punctuation (extension) 
    (r'.*', 'NOUN'),                     # nouns (default)
    (r'$', 'ADP'),   # Add prepositions
    (r'$', 'PRT'),   # Add particles
    (r'$', 'PRON'),   # Add pronouns
    (r'$', 'CONJ')   # Add conjunctions
]
aug_re_tagger = RegexpTagger(aug_rules)

# Dictionary for mapping spacy tags to NLTK tags
mapping_spacy2NLTK = {
    "ADJ": "ADJ",
    "ADP": "ADP",
    "ADV": "ADV",
    "AUX": "VERB",
    "CCONJ": "CONJ",
    "DET": "DET",
    "INTJ": "X",
    "NOUN": "NOUN",
    "NUM": "NUM",
    "PART": "PRT",
    "PRON": "PRON",
    "PROPN": "NOUN",
    "PUNCT": ".",
    "SCONJ": "CONJ",
    "SYM": "X",
    "VERB": "VERB",
    "X": "X",
    "SPACE":"SPACE"
}


def get_dataset(train_test_ratio=0.9):
    total_size = len(treebank.tagged_sents())
    train_indx = math.ceil(total_size * train_test_ratio)
    trn_data = treebank.tagged_sents(tagset='universal')[:train_indx]
    tst_data = treebank.tagged_sents(tagset='universal')[train_indx:]
    return trn_data, tst_data, total_size

class NLTK_tagger:
    def __init__(self, data, order=3, sel_backoff=aug_re_tagger, sel_cutoff=0):
        self.model = nltk.NgramTagger(order, data, backoff=sel_backoff, cutoff=sel_cutoff)
    
    def accuracy(self, data):
        return self.model.accuracy(data)
    

class Spacy_tagger:
    def __init__(self, train_test_ratio=0.9):
        self.model = en_core_web_sm.load()
        self.dataset = self.initialize_tagged_dataset(train_test_ratio)
        self.process_dataset = self.preprocess_dataset(train_test_ratio)
    
    def initialize_tagged_dataset(self,ttr):
        total_size = len(treebank.tagged_sents())
        train_indx = math.ceil(total_size * ttr)
        test_data = treebank.tagged_sents(tagset='universal')[train_indx:]
        return test_data

    def preprocess_dataset(self,ttr):
        total_size = len(treebank.tagged_sents())
        train_indx = math.ceil(total_size * ttr)
        test_sentences = treebank.sents()[train_indx:]
        sentences_raw = [' '.join(sentence) for sentence in test_sentences]
        return sentences_raw
    
    def tag(self):
        return [self.model(sentence) for sentence in self.process_dataset]
        
    def flatten_and_convert_to_NLTK(self, spacy_tagged):
        corpus = []
        corpus_pos = []
        tokens_flat_list = []
        tags_flat_list = []
        tags_flat_list_NLTK = []
        for sent in spacy_tagged:
            tokens = []
            tokens_pos_tag = []
            for token in sent:
                # Append everything to a flat list
                tokens_flat_list.append(token)
                tags_flat_list.append(token.pos_)
                tags_flat_list_NLTK.append(mapping_spacy2NLTK[token.pos_])
                # Also create a nested list of tokens and their POS tags
                tokens.append(token)
                tokens_pos_tag.append(token.pos_)
            corpus.append(tokens)
            corpus_pos.append(tokens_pos_tag)
        return tokens_flat_list, tags_flat_list, tags_flat_list_NLTK, corpus, corpus_pos

    # Since the Spacy tagger does not always parse the document correctly
    # The size of the list of tags does not match the ground truth tags size.
    # Just cropping the list is not a viable solution since the tags are all
    # mismatched and the accuracy will be close to 0, when in reality it is not.
    # So I created this function to align the tags and tokens of the Spacy tagger
    def align_tokens(self,gt_tokens,gt_tags,tokens,tags):
        i = 0
        j = 0
        accumulated_shift = 0
        gt_aligned_tags = []
        aligned_tags = []
        while i < len(gt_tokens):
            # If they match, add them to the new lists
            if str(gt_tokens[i]) == str(tokens[i+j]):
                gt_aligned_tags.append(gt_tags[i])
                aligned_tags.append(tags[i+j])
                i = i + 1
                accumulated_shift = j
                #print(f'{str(gt_tokens[i])} == {str(tokens[i+j])}')
                #print(f'{str(gt_tokens[i+1])} == {str(tokens[i+j+1])}')
                #print(i,j,accumulated_shift)
            # If they don't match, look for the next token if it matches
            else:
                j = j + 1
            # If you can't find a match, just skip the token and start again with the next
            if(i+j >= len(tokens)):
                i = i + 1
                j = accumulated_shift
                
            
        return gt_aligned_tags, aligned_tags



    def compute_accuracy(self, NLTK_tags, tokens_flat_list):
        # Flatten the ground truth dataset
        ground_truth_tags = []
        ground_truth_tokens = []
        for sent in self.dataset:
            for token in sent:
                ground_truth_tags.append(token[1])
                ground_truth_tokens.append(token[0])
        
        # Compute the length before the alignment
        original_size = len(NLTK_tags)
        # Align the Spacy tags with the ground truth tags
        ground_truth_tags, NLTK_tags = self.align_tokens(ground_truth_tokens,ground_truth_tags,tokens_flat_list,NLTK_tags)
        after_alignment_size = len(NLTK_tags)
        # Compute the accuracy with the aligned tags
        accuracy_after_alignment = accuracy_nltk(ground_truth_tags, NLTK_tags)
        # We also need to take into consideration that the un-aligned tags were more so we need to
        # correct the accuracy
        #print(after_alignment_size,original_size)
        accuracy = accuracy_after_alignment * (after_alignment_size / original_size)
        
        return accuracy