# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
from conll import evaluate
from sklearn.metrics import classification_report
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_ds(tmp_train_raw, test_raw):
    # Firt we get the 10% of dataset, then we compute the percentage of these examples 
    # on the training set which is around 11% 
    portion = round(((len(tmp_train_raw) + len(test_raw)) * 0.10)/(len(tmp_train_raw)),2)
    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)
    Y = []
    X = []
    mini_Train = []
    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occure once only, we put them in training
            X.append(tmp_train_raw[id_y])
            Y.append(y)
        else:
            mini_Train.append(tmp_train_raw[id_y])
    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(X, Y, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=Y)
    X_train.extend(mini_Train)
    train_raw = X_train
    dev_raw = X_dev
    y_test = [x['intent'] for x in test_raw]

    return train_raw, dev_raw



def produce_words_int_slots(train_raw, dev_raw, test_raw):
    '''
        args: train dataset raw, dev dataset raw, test dataset raw
        returns: 
                - words: list of all the words in the dataset
                - intents: set of all the intents in the dataset
                - slots: set of all the slots in the dataset
    '''
    words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute 
                                                                # the cutoff
    corpus = train_raw + dev_raw + test_raw # We do not want unk labels, 
                                            # however this depends on the research purpose
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])
    return words, intents, slots

def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
    model.train()
    loss_array = []
    loss_intents = []
    loss_slots = []
    for sample in tqdm(data):
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['utterances'].to(device), sample['slots_len'].to(device))
        #print(slots)
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss_intents.append(loss_intent.detach().cpu().item())
        loss_slots.append(loss_slot.detach().cpu().item())
        loss = loss_intent + loss_slot                            
        loss_array.append(loss.detach().cpu().item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    print('Train loss int avg:', sum(loss_intents)/len(loss_intents))
    print('Train loss slot avg:', sum(loss_slots)/len(loss_slots))
    return loss_array


def train_loop_multiple_lr(data, bert_optimizer, head_optimizer, criterion_slots, criterion_intents, model):
    model.train()
    loss_array = []
    loss_intents = []
    loss_slots = []
    for sample in tqdm(data):
        bert_optimizer.zero_grad() # Zeroing the gradient
        head_optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['utterances'], sample['slots_len'])
        #print(slots)
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss_intents.append(loss_intent.item())
        loss_slots.append(loss_slot.item())
        loss = loss_intent + loss_slot                            
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        bert_optimizer.step() # Update the weights
        head_optimizer.step() # Update the weights
    print('Train loss int avg:', sum(loss_intents)/len(loss_intents))
    print('Train loss slot avg:', sum(loss_slots)/len(loss_slots))
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            #print('sample:')
            #print(sample['utterances'].shape)
            #print(sample['slots_len'].shape)
            slots, intents = model(sample['utterances'], sample['slots_len'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x] 
                           for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            #print(slots.shape)
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                # Need to start from 1 since we do not consider the CLS token
                utt_ids = sample['utterance'][id_seq][1:length+1].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                #print(gt_ids)
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                #print('slot len', len(gt_slots))
                #print('utt len', len(utt_ids))
                utterance = [lang.id2word(elem) for elem in utt_ids]
                to_decode = seq[:length].tolist()
                
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
        #print('evaluation')
        #print(ref_slots[:10])
        #print(hyp_slots[:10])
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predics a class that is not in REF
        print("Error type:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array