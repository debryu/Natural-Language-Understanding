# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import nltk
nltk.download('senseval')
nltk.download('wordnet_ic')
from nltk.corpus import senseval
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from collections import Counter
from tqdm import tqdm
from nltk.metrics.scores import precision, recall, f_measure, accuracy
from nltk.corpus import wordnet_ic
semcor_ic = wordnet_ic.ic('ic-semcor.dat')

# Same function as in the lab
def create_dataset():
    # Sentences containing the word "interest" 
    data = [" ".join([t[0] for t in inst.context]) for inst in senseval.instances('interest.pos')]
    lbls = [inst.senses[0] for inst in senseval.instances('interest.pos')]
    return data, lbls

# Function to create the dataset and the stratifiedKFold
# also initialize the classifier and prepare the vector of features and the labels
def initialize_model():
    # Create the dataset
    data,lbls = create_dataset()
    # Initialize the vectorizer, classifier and the encoder
    vectorizer = CountVectorizer()
    classifier = MultinomialNB()
    lblencoder = LabelEncoder()
    # Initialize the KFold for the cross validation
    stratified_split = StratifiedKFold(n_splits=5, shuffle=True)
    vectors = vectorizer.fit_transform(data)
    # encoding labels for multi-calss
    lblencoder.fit(lbls)
    labels = lblencoder.transform(lbls)
    return classifier, vectors, labels, stratified_split

# Same function as in the lab
def compute_score(classifier, feat_vectors, labels, stratified_split):
    # Compute the score of the classifier using the cross validation
    scores = cross_validate(classifier, feat_vectors, labels, cv=stratified_split, scoring=['f1_micro'])
    score = sum(scores['test_f1_micro'])/len(scores['test_f1_micro'])
    return score

# Same function as in the lab
def collocational_features(inst):
    p = inst.position
    return {
        "w-2_word": 'NULL' if p < 2 else inst.context[p-2][0],
        "w-1_word": 'NULL' if p < 1 else inst.context[p-1][0],
        "w+1_word": 'NULL' if len(inst.context) - 1 < p+1 else inst.context[p+1][0],
        "w+2_word": 'NULL' if len(inst.context) - 1 < p+2 else inst.context[p+2][0]
    }

# Same function as in the lab but extended with POS
def collocational_features_withPOS(inst):
    p = inst.position
    return {
        "w-2_word": 'NULL' if p < 2 else inst.context[p-2][0],
        "w-1_word": 'NULL' if p < 1 else inst.context[p-1][0],
        "w+1_word": 'NULL' if len(inst.context) - 1 < p+1 else inst.context[p+1][0],
        "w+2_word": 'NULL' if len(inst.context) - 1 < p+2 else inst.context[p+2][0],
        # Add POS
        "w-2_POS": 'NULL' if p < 2 else inst.context[p-2][1],
        "w-1_POS": 'NULL' if p < 1 else inst.context[p-1][1],
        "w+1_POS": 'NULL' if len(inst.context) - 1 < p+1 else inst.context[p+1][1],
        "w+2_POS": 'NULL' if len(inst.context) - 1 < p+2 else inst.context[p+2][1]
    }

# Same function as in the lab but extended with POS and NGRAMS
def collocational_features_withPOS_withNGRAM(inst, window_size=2):
    p = inst.position
    words = [t[0] for t in inst.context]
    ngram_right = [] 
    ngram_left = []   
    ngrams = []
    # Compute the ngrams
    # For each word in the context
    for i in range(window_size+1):
        if i==0:
            # Add the word in the middle (inst)
            ngrams.append(words[p])
            ngram_left.append(words[p])
            ngram_right.append(words[p])
        else:
            if p-i >= 0:
                # Add the word to the left and the word of interest (inst)
                # Then extend sequentially to the left adding more words based on the window size
                # Stop if we reach the beginning
                ngram_left.insert(0,words[p-i])
                ngrams.insert(0," ".join(ngram_left))
            
            if p+i < len(words):
                # Add the word to the right and the word of interest (inst)
                # Then extend sequentially to the right adding more words based on the window size
                # Stop if we reach the end
                ngram_right.append(words[p+i])
                ngrams.append(" ".join(ngram_right))
    return {
        "w-2_word": 'NULL' if p < 2 else inst.context[p-2][0],
        "w-1_word": 'NULL' if p < 1 else inst.context[p-1][0],
        "w+1_word": 'NULL' if len(inst.context) - 1 < p+1 else inst.context[p+1][0],
        "w+2_word": 'NULL' if len(inst.context) - 1 < p+2 else inst.context[p+2][0],
        # Add POS
        "w-2_POS": 'NULL' if p < 2 else inst.context[p-2][1],
        "w-1_POS": 'NULL' if p < 1 else inst.context[p-1][1],
        "w+1_POS": 'NULL' if len(inst.context) - 1 < p+1 else inst.context[p+1][1],
        "w+2_POS": 'NULL' if len(inst.context) - 1 < p+2 else inst.context[p+2][1],
        # Add NGRAMS
        "ngrams": ngrams
    }

# Same function as in the lab
def produce_collocational_features(feat_type = 'all'):
    if feat_type == 'all':
        data_col = [collocational_features_withPOS_withNGRAM(inst) for inst in senseval.instances('interest.pos')]
    elif feat_type == 'POS':
        data_col = [collocational_features_withPOS(inst) for inst in senseval.instances('interest.pos')]
    else:
        data_col = [collocational_features(inst) for inst in senseval.instances('interest.pos')]
    dvectorizer = DictVectorizer(sparse=False)
    dvectors = dvectorizer.fit_transform(data_col)
    return dvectors

# Same function as in the lab
def concatenate_features(BOW_vector, collocational_f_vector):
    # types of CountVectorizer and DictVectorizer outputs are different 
    # we need to convert them to the same format
    concatenated_vectors = np.concatenate((BOW_vector.toarray(), collocational_f_vector), axis=1)
    return concatenated_vectors


# A bit of preprocessing, same as in the lab 
def preprocess(text):
    mapping = {"NOUN": wordnet.NOUN, "VERB": wordnet.VERB, "ADJ": wordnet.ADJ, "ADV": wordnet.ADV}
    sw_list = stopwords.words('english')
    
    lem = WordNetLemmatizer()
    # tokenize, if input is text
    tokens = nltk.word_tokenize(text) if type(text) is str else text
    # compute pos-tag
    tagged = nltk.pos_tag(tokens, tagset="universal")
    # lowercase
    tagged = [(w.lower(), p) for w, p in tagged]
    # optional: remove all words that are not NOUN, VERB, ADJ, or ADV (i.e. no sense in WordNet)
    tagged = [(w, p) for w, p in tagged if p in mapping]
    # re-map tags to WordNet (return orignal if not in-mapping, if above is not used)
    tagged = [(w, mapping.get(p, p)) for w, p in tagged]
    #print('tagged',tagged)
    # remove stopwords
    tagged = [(w, p) for w, p in tagged if w not in sw_list]
    # lemmatize
    tagged = [(w, lem.lemmatize(w, pos=p), p) for w, p in tagged]
    #print('tagged2',tagged)
    # unique the list
    tagged = list(set(tagged))
    
    return tagged

# Same function as in the lab
def get_top_sense(words, sense_list):
    # get top sense from the list of sense-definition tuples
    # assumes that words and definitions are preprocessed identically
    val, sense = max((len(set(words).intersection(set(defn))), ss) for ss, defn in sense_list)
    return val, sense

# Same function as in the lab
def get_top_sense_sim(context_sense, sense_list, similarity):
    scores = []
    for sense in sense_list:
        ss = sense[0]
        if similarity == "path":
            try:
                scores.append((context_sense.path_similarity(ss), ss))
            except:
                scores.append((0, ss))
        elif similarity == "lch":
            try:
                scores.append((context_sense.lch_similarity(ss), ss))
            except:
                scores.append((0, ss))
        elif similarity == "wup":
            try:
                scores.append((context_sense.wup_similarity(ss), ss))
            except:
                scores.append((0, ss))
        elif similarity == "resnik":
            try:
                scores.append((context_sense.res_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        elif similarity == "lin":
            try:
                scores.append((context_sense.lin_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        elif similarity == "jiang":
            try:
                scores.append((context_sense.jcn_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        else:
            print("Similarity metric not found")
            return None
    
    val, sense = max(scores)
    
    return val, sense

# Same function as in the lab
def get_sense_definitions(context):
    # input is text or list of strings
    lemma_tags = preprocess(context)
    #print('lemma_tags:',lemma_tags)
    # let's get senses for each
    senses = [(w, wordnet.synsets(l, p)) for w, l, p in lemma_tags]
    #print('senses:',senses)
    # let's get their definitions
    definitions = []
    for raw_word, sense_list in senses:
        if len(sense_list) > 0:
            # let's tokenize, lowercase & remove stop words 
            def_list = []
            for s in sense_list:
                defn = s.definition()
                # let's use the same preprocessing
                tags = preprocess(defn)
                toks = [l for w, l, p in tags]
                def_list.append((s, toks))
            definitions.append((raw_word, def_list))
    #print('definitions:',definitions)
    return definitions

# Same function as in the lab with a slight modification
def original_lesk(context_sentence, ambiguous_word, pos=None, synsets=None, majority=False):
    #print(ambiguous_word)
    #print('context_sent:',context_sentence)
    #print('ambiguous_word:',ambiguous_word)
    #print('subset:',set(context_sentence)-set([ambiguous_word]))
    #print('synsets:',synsets)
    context_senses = get_sense_definitions(set(context_sentence)-set([ambiguous_word]))
    #print('Context senses:',context_senses)
    if synsets is None:
        #print(get_sense_definitions(ambiguous_word))
        synsets = get_sense_definitions(ambiguous_word)[0][1]
    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]
    if not synsets:
        return None
    scores = []
    #print('context_senses:',context_senses)
    for senses in context_senses:
        #print('senses:',senses)
        for sense in senses[1]:
            #print(sense)
            # Append the score with the highest score and relative value 
            # Compare sense[1] with synsets
            # get_top_sense might help here
            #print(sense[1])
            highest_score = get_top_sense(sense[1], synsets)
            #print(highest_score)
            scores.append(highest_score)

    #print( 'scores:',scores)
    if majority:
        # We remove 0 scores, senses without overlapping
        filtered_scores = [x[1] for x in scores if x[0] != 0]
        if len(filtered_scores) > 0:
            best_sense = Counter(filtered_scores).most_common(1)[0][0]#We need to select the most common syn. Counter function might help here
        else:
            # Almost random selection
            # If there are no scores, we pick a random synset
            # This is the modification from the original function
            # Mainly because if the sentence is short and does not contain any word from the mapping
            # (except from the word we are trying to find the sense) we will have no scores
            if len(scores) == 0:
                # Pick a random synset
                size = len(synsets)
                random_index = np.random.randint(0,size)
                return synsets[random_index][0]
            else:
                best_sense = Counter([x[1] for x in scores]).most_common(1)[0][0]    # The same as above but using scores instead of filtered_scores
    else:
        _, best_sense = max(scores)# Get the maximum of scores.

    return best_sense

# Same function as in the lab
def lesk_similarity(context_sentence, ambiguous_word, similarity="resnik", pos=None, synsets=None, majority=False):
    context_senses = get_sense_definitions(set(context_sentence) - set([ambiguous_word]))
    if synsets is None:
        synsets = get_sense_definitions(ambiguous_word)[0][1]
    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]
    if not synsets:
        return None
    scores = []
    for senses in context_senses:
        for sense in senses[1]:
            scores.append(get_top_sense_sim(sense[0], synsets, similarity))      
    if len(scores) == 0:
        return synsets[0][0]               
    # Majority voting as before    
    if majority:
        # We remove 0 scores, senses without overlapping
        filtered_scores = [x[1] for x in scores if x[0] != 0]
        if len(filtered_scores) > 0:
            best_sense = Counter(filtered_scores).most_common(1)[0][0]#We need to select the most common syn. Counter function might help here
        else:
            # Almost random selection
            best_sense = Counter([x[1] for x in scores]).most_common(1)[0][0] # The same as above but using scores instead of filtered_scores
    else:
        _, best_sense = max(scores)# Get the maximum of scores.

    return best_sense


# Let's create mapping from convenience
mapping = {
    'interest_1': 'interest.n.01',
    'interest_2': 'interest.n.03',
    'interest_3': 'pastime.n.01',
    'interest_4': 'sake.n.01',
    'interest_5': 'interest.n.05',
    'interest_6': 'interest.n.04',
}

# Same function as in the lab
def evaluate_metrics(metric,mapping,dataset):
    refs = {k: set() for k in mapping.values()}
    hyps = {k: set() for k in mapping.values()}
    refs_list = []
    hyps_list = []
    synsets = []
    for ss in wordnet.synsets('interest', pos='n'):
        if ss.name() in mapping.values():
            #print(ss.name())
            defn = ss.definition()# estract the defitions
            #print(defn)
            tags = preprocess(defn)# Preproccess the definition
            #print(tags)
            toks = [l for w, l, p in tags]# From tags extract the tokens
            synsets.append((ss,toks))

    for i,index in enumerate(tqdm(dataset)):
        inst = senseval.instances('interest.pos')[index]
        txt = [t[0] for t in inst.context]
        #print(txt)
        #print(i)
        # Skip the 880th instance because it gives out of bound error
        #if len(txt) == 880:
        #    continue
        #print(i)
        raw_ref = inst.senses[0] # let's get first sense
        if(metric=='original'):
            hyp = original_lesk(txt, txt[inst.position], synsets=synsets, majority=True).name()
        elif(metric=='lesk_similarity'):
            hyp = lesk_similarity(txt, txt[inst.position], synsets=synsets, majority=True).name()
        else:
            print("Metric not found")
            return None
        ref = mapping.get(raw_ref)
        # for precision, recall, f-measure        
        refs[ref].add(i)
        hyps[hyp].add(i)
        # for accuracy
        refs_list.append(ref)
        hyps_list.append(hyp)

    for cls in hyps.keys():
        p = precision(refs[cls], hyps[cls])
        r = recall(refs[cls], hyps[cls])
        f = f_measure(refs[cls], hyps[cls], alpha=1)
        # Make sure that p, r, f are not None
        # If they are None, set them to 0
        if(p is None):
            p= 0.0
        if(r is None):
            r = 0.0
        if(f is None):
            f = 0.0
        print("{:15s}: p={:.3f}; r={:.3f}; f={:.3f}; s={}".format(cls, p, r, f, len(refs[cls])))
    acc = round(accuracy(refs_list, hyps_list), 3)
    return acc