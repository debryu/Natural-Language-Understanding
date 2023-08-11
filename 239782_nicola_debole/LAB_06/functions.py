# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
# Spacy version 
from nltk.corpus import dependency_treebank
from nltk.parse.dependencygraph import DependencyGraph
from spacy.tokenizer import Tokenizer
import spacy 
import spacy_conll
import stanza
import spacy_stanza

from nltk.parse import DependencyEvaluator



def evaluate_parses(parses):
    de = DependencyEvaluator(parses, dependency_treebank.parsed_sents()[-100:])
    las, uas = de.eval()
    return las, uas

def create_dataset():
    treebank_sentences = dependency_treebank.sents()
    last_100_sentences = treebank_sentences[-100:]
    return last_100_sentences

def preprocess_data(list_of_list_of_words):
    sentences = []
    # Since the words are stored in a list of lists, we need to join all the words in a sentence
    for list_of_words in list_of_list_of_words:
        sentence = ' '.join(list_of_words)
        sentences.append(sentence)
    return sentences


# Load the spacy model
spacy_model = spacy.load("en_core_web_sm")

# Set up the conll formatter 
spacy_config = {"ext_names": {"conll_pd": "pandas"},
        "conversion_maps": {"DEPREL": {"nsubj": "subj"}}}

# Add the formatter to the pipeline
spacy_model.add_pipe("conll_formatter", config=spacy_config, last=True)
# Split by white space
spacy_model.tokenizer = Tokenizer(spacy_model.vocab) 

def spacy_parsing(sentence):
    # Parse the sentence
    doc = spacy_model(sentence)

    # Convert doc to a pandas object
    df = doc._.pandas

    # Select the columns accoroding to Malt-Tab format
    tmp = df[["FORM", 'XPOS', 'HEAD', 'DEPREL']].to_string(header=False, index=False)
    tmp_as_df = df[["FORM", 'XPOS', 'HEAD', 'DEPREL']]
    
    # See the outcome
    #print(tmp)

    # Get finally our the DepencecyGraph
    dp = DependencyGraph(tmp)
    #print('Tree:')
    #dp.tree().pretty_print(unicodelines=True, nodedist=4)

    return tmp_as_df, dp


# Set up the conll formatter 
#tokenize_pretokenized used to tokenize by whitespace 
stanza_model = spacy_stanza.load_pipeline("en", verbose=False, tokenize_pretokenized=True)
stanza_config = {"ext_names": {"conll_pd": "pandas"},
          "conversion_maps": {"DEPREL": {"nsubj": "subj", "root":"ROOT"}}}

# Add the formatter to the pipeline
stanza_model.add_pipe("conll_formatter", config=stanza_config, last=True)

def stanza_parsing(sentence):
    # Parse the sentence
    doc = stanza_model(sentence)
    # Convert doc to a pandas object
    df = doc._.pandas
    #print(df)
    # Select the columns accoroding to Malt-Tab format
    tmp = df[["FORM", 'XPOS', 'HEAD', 'DEPREL']].to_string(header=False, index=False)
    tmp_as_df = df[["FORM", 'XPOS', 'HEAD', 'DEPREL']]
    # See the outcome
    #print(tmp)

    # Get finally our the DepencecyGraph
    dp = DependencyGraph(tmp)
    #print('Tree:')
    #dp.tree().pretty_print(unicodelines=True, nodedist=4)

    return tmp_as_df, dp