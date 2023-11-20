# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
from tqdm import tqdm
import torch
from sklearn.metrics import classification_report
import numpy as np

def train_loop(data, optimizer, criterion, model, key):
    model.train()
    loss_array = []
    loss_intents = []
    loss_slots = []
    for sample in tqdm(data):
        optimizer.zero_grad() # Zeroing the gradient
        predicted_sub = model(sample[key])
        #print(slots)
        loss = criterion(predicted_sub.to('cpu'), torch.LongTensor(sample['label']))  
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
        #break
    #print('Train loss batch avg:', sum(loss_array)/len(loss_array))
    return loss_array

def train_loop_multiple_lr(data, optimizer_bert,optimizer_head, criterion, model, key):
    model.train()
    loss_array = []
    loss_intents = []
    loss_slots = []
    for sample in tqdm(data):
        optimizer_bert.zero_grad() # Zeroing the gradient
        optimizer_head.zero_grad()
        predicted_sub = model(sample[key])
        #print(slots)
        loss = criterion(predicted_sub.to('cpu'), torch.LongTensor(sample['label']))  
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer_bert.step() # Update the weights
        optimizer_head.step()
        #break
    #print('Train loss batch avg:', sum(loss_array)/len(loss_array))
    return loss_array




def eval_loop(data, criterion, model, key):
    model.eval()
    loss_array = []
    
    labels = []
    predictions = []
    
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            #print('sample:')
            #print(sample['utterances'].shape)
            #print(sample['slots_len'].shape)
           
            predicted_sub = model(sample[key]).squeeze(0)
            loss = criterion(predicted_sub.to('cpu'), torch.LongTensor(sample['label']))
            loss_array.append(loss.item())
            for item in list(predicted_sub.argmax(dim=1).to('cpu').numpy()):
                predictions.append(item)
            for item in sample['label']:
                labels.append(item)
           
    #print(predictions)
    #print(labels)
    results = classification_report(predictions, labels, zero_division=False, output_dict=True)
    return results, loss_array


def train_loop_longformer(data, optimizer, criterion, model, key, gradient_accumulation_steps = 64):
    model.train()
    loss_array = []
    loss_intents = []
    loss_slots = []
    for i,sample in enumerate(tqdm(data)):
        #print('sample:',sample[key])
        predicted_sub, cls = model(sample[key])
        predicted_sub = predicted_sub.squeeze(0).to('cpu')
        #print('cls',cls)
        #print(predicted_sub)
        #print(slots)
        loss = criterion(predicted_sub, sample['label'][0].to(torch.long))/gradient_accumulation_steps
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        if (i+1) % gradient_accumulation_steps == 0 or i == len(data)-1:
                    #print('Optimizing')
                    optimizer.step()
                    optimizer.zero_grad()
        
        if i == -1:
            break
        #break
    #print('Train loss batch avg:', sum(loss_array)/len(loss_array))
    return loss_array

def eval_loop_longformer(data, model, key):
    model.eval()
    labels = []
    predictions = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for i,sample in enumerate(data):
            #print(sample[key].squeeze(0))
            #print('sample:',sample[key])
            predicted_sub, cls = model(sample[key])
            predicted_sub = predicted_sub.squeeze(0).to('cpu')
            #print('cls',cls)
            #print(predicted_sub)
            value = predicted_sub.argmax(dim=0).to('cpu').numpy()
            #print(value)     
            predictions.append(value)
            labels.append(sample['label'][0])
            if i == -1:
                break
    #print(predictions)
    #print(labels)
    #print(predictions)
    #print(labels)
    results = classification_report(predictions, labels, zero_division=False, output_dict=True)
    return results


def eval_vader(data):
    labels = []
    predictions = []
    for i,batch in enumerate(data):
        # Get every document score in the batch
        for document_score in batch['document_scores']:
            # Extract the negative and positive values
            # Ignore all the other values since we are trying to make a simple classifier
            # We also tried to use the 5 features as the input of a FFNN but the results were almost the same
            # but it was way slower
            neg_values = [-x['neg'] for x in document_score]
            pos_values = [x['pos'] for x in document_score]
            final_values = [x+y for x,y in zip(neg_values,pos_values)]
            average = np.mean(final_values)
            if average > 0:
                predictions.append(1)
            else:
                predictions.append(0)
            labels.append(batch['label'][0])
    results = classification_report(predictions, labels, zero_division=False, output_dict=True)
    return results

