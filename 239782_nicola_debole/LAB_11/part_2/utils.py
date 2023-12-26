# Add functions or classes used for data loading and preprocessing
import string
import numpy as np
import torch
from sklearn.cluster import KMeans

def read_data(path):
    """
    read data from the specified path
    :param path: path of dataset
    :return:
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
                    #print(GloVe_embeddings[word.lower()])
                    emb_sent.append(GloVe_embeddings[word.lower()])
                else:
                    emb_sent.append(np.zeros(300))
                ort.append(element['ote_raw_tags'][i])
                trt.append(element['ts_raw_tags'][i])
                w.append(word)
        total_len = len(emb_sent)
        #print('tl',total_len)
        emb_sent = np.array(emb_sent)
        label = torch.zeros(total_len, 4)
        aspects = torch.zeros(total_len, 2)
        index = 0
        for i,word in enumerate(sentence):
            tag = element['ts_raw_tags'][i]
            #print(tag)
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
        #print(curr_word)
        latent_point = np.zeros(300)
        curr_tag = None
        curr_word = ''
        #break
                
    return points,words

def integrate_dataset_with_centroids(dataset, centroids:torch.tensor, cluster:KMeans):
   

    for element in dataset:
        similarity_fn = torch.nn.CosineSimilarity(dim=1)
        sentence_embedding = torch.tensor(element['emb'])
        similarity_score = torch.zeros((sentence_embedding.shape[0],centroids.shape[0]))
        assigned_aspect = cluster.predict(sentence_embedding)
        assigned_aspect += 1 # To leave the 0 for the padded/no cluster slots
        #print(assigned_aspect)
        for i,word_embedding in enumerate(sentence_embedding):
            similarity_score[:] = similarity_fn(word_embedding, centroids[:])
        element['cosine'] = similarity_score
        element['cluster'] = torch.ones((assigned_aspect.shape))#torch.tensor(assigned_aspect)
        # Find where there is no aspect
        no_cluster = torch.argwhere(element['label'][:,0] == torch.tensor(1.0))
        #print(element['label'].shape)
        #print(element['cluster'].shape)
        #print(no_cluster)
        element['cluster'][no_cluster] = torch.tensor(0.0)
        
    return dataset
        #print(similarity_score.shape)
        #print(similarity_score[0])
        #break
