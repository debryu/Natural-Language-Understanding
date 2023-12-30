import string
import numpy as np
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm
import spacy
nlp = spacy.load("en_core_web_sm")
import en_core_web_sm
nlp = en_core_web_sm.load()

def read_data(path):
    """
        Read data from the specified path
            Args:
                path: path of dataset
            Returns:
                dataset: list of records, each record is a dictionary
    """
    dataset = []
    with open(path, encoding='UTF-8') as fp:
        for line in fp:
            #print(line)
            record = {}
            sent, tag_string = line.strip().split('####')
            record['sentence'] = sent
            word_tag_pairs = tag_string.split(' ')
            # tag sequence for targeted sentiment
            ts_tags = []
            # tag sequence for opinion target extraction
            ote_tags = []
            # word sequence
            words = []
            for item in word_tag_pairs:
                # valid label is: O, T-POS, T-NEG, T-NEU
                eles = item.split('=')
                if len(eles) == 2:
                    word, tag = eles
                elif len(eles) > 2:
                    tag = eles[-1]
                    word = (len(eles) - 2) * "="
                if word not in string.punctuation:
                    # lowercase the words
                    words.append(word.lower())
                else:
                    # replace punctuations with a special token
                    words.append('PUNCT')
                if tag == 'O':
                    ote_tags.append('O')
                    ts_tags.append('O')
                elif tag == 'T-POS':
                    ote_tags.append('T')
                    ts_tags.append('T-POS')
                elif tag == 'T-NEG':
                    ote_tags.append('T')
                    ts_tags.append('T-NEG')
                elif tag == 'T-NEU':
                    ote_tags.append('T')
                    ts_tags.append('T-NEU')
                else:
                    raise Exception('Invalid tag %s!!!' % tag)
            record['words'] = words.copy()
            record['ote_raw_tags'] = ote_tags.copy()
            record['ts_raw_tags'] = ts_tags.copy()
            dataset.append(record)
    #print("Obtain %s records from %s" % (len(dataset), path))
    return dataset

def dataset2glove(dataset, GloVe_embeddings):
    '''
        Add to the dataset the GloVe embedding for each wordk
        Args:
            dataset: the dataset to be converted
            GloVe_embeddings: the GloVe embeddings to be used
        Returns:
            new_ds: the new dataset
    '''
    new_ds = []
    for element in dataset:
        sentence = element['words']
        emb_sent = []
        ort = []
        trt = []
        w = []
        for i,word in enumerate(sentence):
            if word != 'PUNCT':
                # Handle OOV words
                if word.lower() in GloVe_embeddings:
                    emb_sent.append(GloVe_embeddings[word.lower()])
                else:
                    emb_sent.append(np.zeros(300))
                ort.append(element['ote_raw_tags'][i])
                trt.append(element['ts_raw_tags'][i])
                w.append(word)
        total_len = len(emb_sent) # The length of the sentence
        emb_sent = np.array(emb_sent)
        label = torch.zeros(total_len, 4)
        aspects = torch.zeros(total_len, 2)
        index = 0
        for i,word in enumerate(sentence):
            tag = element['ts_raw_tags'][i]
            if word != 'PUNCT':
                if tag == 'O':
                    label[index][0] = 1
                    aspects[index][0] = 1
                elif tag == 'T-POS':
                    label[index][1] = 1
                    aspects[index][1] = 1
                elif tag == 'T-NEU':
                    label[index][2] = 1
                    aspects[index][1] = 1
                elif tag == 'T-NEG':
                    label[index][3] = 1
                    aspects[index][1] = 1
                else:
                    raise Exception('Invalid tag %s!!!' % tag)
                index += 1

        item = {'emb': emb_sent, 'label':label, 'aspects':aspects, 'len':total_len, 'words': w, 'sentence': element['sentence'], 'ote_raw_tags': ort, 'ts_raw_tags': trt}
        new_ds.append(item)
    return new_ds
    

def extract_centroids(dataset):
    '''
        First create a list of points (in glove embedding space) from all the aspect words in the dataset.
        Aspects that are multi-word are considered as one point, as their embedding is summed up.
        Args:
            dataset: the dataset to be used
        Returns:
            points: the list of points
            words: the list of strings, each point is associated with an item in the list, which defines the aspects.
    
    '''
    points = []
    words = []
    for element in dataset:
        s_length = len(element['ote_raw_tags'])
        i = 0
        start = 0
        latent_point = np.zeros(300)
        curr_tag = None
        curr_word = ''
        for i in range(s_length):
            if element['ote_raw_tags'][i] == 'T':
                #print(element['words'][i])
                if curr_tag is None:
                    curr_tag = element['ts_raw_tags'][i]
                    latent_point += element['emb'][i]
                    curr_word = element['words'][i]
                elif element['ts_raw_tags'][i] == curr_tag:
                    latent_point += element['emb'][i]
                    curr_word += f' { element["words"][i]}'
                elif element['ts_raw_tags'][i] != curr_tag:
                    points.append(latent_point)
                    words.append(curr_word)
                    #print(curr_word)
                    latent_point = np.zeros(300)
                    if element['ote_raw_tags'][i] == 'T':
                        curr_tag = element['ts_raw_tags'][i]
                        curr_word = element['words'][i]
                        latent_point += element['emb'][i]
                        curr_word = element['words'][i]
                    else:
                        curr_tag = None
                        curr_word = ''
            else:
                if curr_tag is not None:
                    points.append(latent_point)
                    words.append(curr_word)
                    #print(curr_word)
                    latent_point = np.zeros(300)
                    curr_tag = None
                    curr_word = ''
                else:
                    continue
        if curr_word != '':
            points.append(latent_point)
            words.append(curr_word)
        latent_point = np.zeros(300)
        curr_tag = None
        curr_word = ''                
    return points,words

def integrate_dataset_with_centroids(dataset, centroids:torch.tensor, cluster:KMeans):
    '''
        Add to the dataset the cosine similarity of each word with each cluster
        Args:
            dataset: the dataset to be integrated
            centroids: the centroids of the clusters
            cluster: the KMeans object
        Returns:
            dataset: the new dataset
    '''

    for element in tqdm(dataset):
        similarity_fn = torch.nn.CosineSimilarity(dim=1)
        sentence_embedding = torch.tensor(element['emb'])
        similarity_score = torch.zeros((sentence_embedding.shape[0],centroids.shape[0]))
        assigned_aspect = cluster.predict(sentence_embedding) # Not used right now 
        # It basicallt assigns each word to a cluster, for now I just want to distinguish between aspect and no aspect
        # Without caring about which aspect it is
        assigned_aspect += 1 # To leave the 0 for the padded/no cluster slots
        for i,word_embedding in enumerate(sentence_embedding):
            similarity_score[:] = similarity_fn(word_embedding, centroids[:])
        element['cosine'] = similarity_score
        element['cluster'] = torch.ones((assigned_aspect.shape))
        # Find where there is no aspect
        no_cluster = torch.argwhere(element['label'][:,0] == torch.tensor(1.0))
        element['cluster'][no_cluster] = torch.tensor(0.0)
        
    return dataset
