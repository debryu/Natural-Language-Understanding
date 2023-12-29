# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
from functions import *
from model import *
from utils import *
import os
import random
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
    test_raw = load_data(os.path.join('dataset','ATIS','test.json'))
    train_raw, dev_raw = create_ds(tmp_train_raw, test_raw)
    words,intents,slots = produce_words_int_slots(train_raw, dev_raw, test_raw)

    tok_id = Tokenizer_id(words, intents, slots, cutoff=0)

    # Create our datasets
    train_dataset = IntentsAndSlots(train_raw, tok_id)
    dev_dataset = IntentsAndSlots(dev_raw, tok_id)
    test_dataset = IntentsAndSlots(test_raw, tok_id)
    # Dataloader instantiation
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    '''
    ----------------------------------------
    START TRAINING
    
    '''
    use_different_lr = True
    hid_size = 200
    emb_size = 300

    lr = 0.001 # learning rate
    clip = 5 # Clip the gradient

    out_slot = len(tok_id.slot2id)
    out_int = len(tok_id.intent2id)
    vocab_len = len(bert_tokenizer.vocab)

    runs = 2
    slot_f1s, intent_acc = [], []
    for run in tqdm(range(0, runs)):
        model = Bert_Intent_Slot(out_slot, out_int, vocab_len, pad_index=PAD_TOKEN).to(device)
        #model.apply(init_weights)
        params = list(model.parameters())
        bert_params = params[:197]
        head_params = params[197:]
        '''
        Code for showing all the parameters
            for i,p in enumerate(params):
                print(i,p[0], str(tuple(p[1].size())))
            for i,p in enumerate(bert_params):
                print(i,p[0], str(tuple(p[1].size())))
            for i,p in enumerate(head_params):
                print(i,p[0], str(tuple(p[1].size())))
        '''
        
        
        if(use_different_lr):
            # Use two different optimizer
            bert_optimizer = optim.Adam(bert_params, lr=0.000001)
            head_optimizer = optim.Adam(head_params, lr=lr)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)


        model = model.to(device)  
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()
        
        n_epochs = 15
        patience = 3
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        for x in range(1,n_epochs):
            if(use_different_lr):
                loss = train_loop_multiple_lr(train_loader, bert_optimizer, head_optimizer, criterion_slots, 
                                criterion_intents, model)
            else:
                loss = train_loop(train_loader, optimizer, criterion_slots, 
                            criterion_intents, model)
            if x % 5 == 0:
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                            criterion_intents, model, tok_id)
                losses_dev.append(np.asarray(loss_dev).mean())
                f1 = results_dev['total']['f']

                if f1 > best_f1:
                    best_f1 = f1
                else:
                    patience -= 1
                if patience <= 0: # Early stoping with patient
                    break
                break
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                                criterion_intents, model, tok_id)
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])
        # Save the model
        torch.save(model.state_dict(), f'LAB_10/part_2/models/model__multilr_run{run}.pt') 


    

    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))

    plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')
    plt.title('Train and Dev Losses')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(sampled_epochs, losses_train, label='Train loss')
    plt.plot(sampled_epochs, losses_dev, label='Dev loss')
    plt.legend()
    plt.show()


