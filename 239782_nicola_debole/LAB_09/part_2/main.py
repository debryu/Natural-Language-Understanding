# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Import everything from functions.py file
from functions import *
from model import *
from utils import *
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    # Loading the raw dataset
    train_raw = read_file("dataset/ptb.train.txt")
    dev_raw = read_file("dataset/ptb.valid.txt")
    test_raw = read_file("dataset/ptb.test.txt")

    # Compute the vocabulary only on the train set
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])

    # Compute the vocabulary of the token ids
    tokens_id = ID_tokenizer(train_raw, ["<pad>", "<eos>"])

    # Create the processed dataset (with token ids)
    train_dataset = PennTreeBank(train_raw, tokens_id)
    dev_dataset = PennTreeBank(dev_raw, tokens_id)
    test_dataset = PennTreeBank(test_raw, tokens_id)

    # Dataloader instantiation
    train_loader = DataLoader(train_dataset, batch_size=256, collate_fn=partial(collate_fn, pad_token=tokens_id.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=tokens_id.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=tokens_id.word2id["<pad>"]))

    '''
    ------------------ PART 1 ------------------
                    Weight Tying
                    
    --------------------------------------------
    '''
    clip = 5 # Clip the gradient
    #learning_rate = 0.5 # Used for training from scratch the model checkpoint
    learning_rate = 0.45 # Used for fine tuning
    n_epochs = 1000
    patience_lvl = 30
    patience = patience_lvl
    # Set to True if you want to train the model
    train = False
    # Set to False if you want to train the model from scratch
    # Otherwise it will load the model checkpoint
    load_model_checkpoint = True 
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    vocab_len = len(tokens_id.word2id)
    #                   (embedding size, vocab size)
    model = LSTM_WT(380, vocab_len, pad_index=tokens_id.word2id["<pad>"]).to(device)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion_train = nn.CrossEntropyLoss(ignore_index=tokens_id.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=tokens_id.word2id["<pad>"], reduction='sum')
    if not train:
        parameters = torch.load("LAB_09/models/part2/best_model_LSTM_SGD_WT.pt")
        model.load_state_dict(parameters)
        model.to(device)
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, model) 
    else:
        if load_model_checkpoint:
            parameters = torch.load("LAB_09/models/part2/model_checkpoint_LSTM_SGD_WT.pt")
            model.load_state_dict(parameters)
    
        pbar = tqdm(range(1,n_epochs))
        #If the PPL is too high try to change the learning rate
        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
            #torch.save(model.state_dict(), "LAB_09/models/part1/model_temp.pt") 
            if epoch % 1 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                losses_dev.append(np.asarray(loss_dev).mean())
                pbar.set_description(f"PPL: {ppl_dev} | PATIENCE: {patience}|")
                if  ppl_dev < best_ppl: # the lower, the better
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to('cpu')
                    patience = patience_lvl
                else:
                    patience -= 1
                    
                if patience <= 0: # Early stopping with patience
                    break # Not nice but it keeps the code clean

    
        # Save the best model 
        best_model.to(device)
        torch.save(best_model.state_dict(), "LAB_09/models/part2/best_model_LSTM_SGD_WT.pt")  
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)      
    
    
    print('[LSTM + SGD + Weight Tying]   Test ppl: ', final_ppl)



    '''
    ------------------ PART 2 ------------------
                Variational Dropout
                    
    --------------------------------------------
    '''
    clip = 5 # Clip the gradient
    #learning_rate = 0.45 # Used for training from scratch the model checkpoint
    learning_rate = 0.45 # Used for fine tuning
    n_epochs = 300
    patience_lvl = 10
    patience = patience_lvl
    # Set to True if you want to train the model
    train = True
    # Set to False if you want to train the model from scratch
    # Otherwise it will load the model checkpoint
    load_model_checkpoint = False 
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    vocab_len = len(tokens_id.word2id)
    #                   (embedding size, hidden size, vocab size)
    model = LSTM_LM_VAR_DROP(256,380, vocab_len, tokens_id.word2id["<pad>"], 0.2, 0.4).to(device)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion_train = nn.CrossEntropyLoss(ignore_index=tokens_id.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=tokens_id.word2id["<pad>"], reduction='sum')
    if not train:
        parameters = torch.load("LAB_09/models/part2/best_model_LSTM_SGD_VAR_DROP.pt")
        model.load_state_dict(parameters)
        model.to(device)
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, model) 
    else:
        if load_model_checkpoint:
            parameters = torch.load("LAB_09/models/part2/model_checkpoint_LSTM_SGD_VAR_DROP.pt")
            model.load_state_dict(parameters)
    
        pbar = tqdm(range(1,n_epochs))
        #If the PPL is too high try to change the learning rate
        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
            #torch.save(model.state_dict(), "LAB_09/models/part1/model_temp.pt") 
            if epoch % 1 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                losses_dev.append(np.asarray(loss_dev).mean())
                pbar.set_description(f"PPL: {ppl_dev} | PATIENCE: {patience}|")
                if  ppl_dev < best_ppl: # the lower, the better
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to('cpu')
                    patience = patience_lvl
                else:
                    patience -= 1
                    
                if patience <= 0: # Early stopping with patience
                    break # Not nice but it keeps the code clean

    
        # Save the best model 
        best_model.to(device)
        torch.save(best_model.state_dict(), "LAB_09/models/part2/best_model_LSTM_SGD_VAR_DROP.pt")  
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)      
    
    
    print('[LSTM + SGD + Variational Dropout]   Test ppl: ', final_ppl)



    '''
    ------------------ PART 3 ------------------
          Non-monotonically Triggered AvSGD
                    
    --------------------------------------------
    '''
    clip = 5 # Clip the gradient
    #learning_rate = 0.45 # Used for training from scratch the model checkpoint
    learning_rate = 4.5 # Used for fine tuning
    n_epochs = 300
    patience_lvl = 10
    patience = patience_lvl
    # Set to True if you want to train the model
    train = True
    # Set to False if you want to train the model from scratch
    # Otherwise it will load the model checkpoint
    load_model_checkpoint = False 
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    vocab_len = len(tokens_id.word2id)
    #                   (embedding size, hidden size, vocab size)
    model = LSTM_LM_VAR_DROP(256,380, vocab_len, tokens_id.word2id["<pad>"], 0.2, 0.4).to(device)
    model.apply(init_weights)

    optimizer = optim.ASGD(model.parameters(), lr=learning_rate)
    criterion_train = nn.CrossEntropyLoss(ignore_index=tokens_id.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=tokens_id.word2id["<pad>"], reduction='sum')
    if not train:
        parameters = torch.load("LAB_09/models/part2/best_model_LSTM_NT_AvSGD.pt")
        model.load_state_dict(parameters)
        model.to(device)
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, model) 
    else:
        if load_model_checkpoint:
            parameters = torch.load("LAB_09/models/part2/model_checkpoint_LSTM_NT_AvSGD.pt")
            model.load_state_dict(parameters)
    
        pbar = tqdm(range(1,n_epochs))
        #If the PPL is too high try to change the learning rate
        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
            #torch.save(model.state_dict(), "LAB_09/models/part1/model_temp.pt") 
            if epoch % 1 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                losses_dev.append(np.asarray(loss_dev).mean())
                pbar.set_description(f"PPL: {ppl_dev} | PATIENCE: {patience}|")
                if  ppl_dev < best_ppl: # the lower, the better
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to('cpu')
                    patience = patience_lvl
                else:
                    patience -= 1
                    
                if patience <= 0: # Early stopping with patience
                    break # Not nice but it keeps the code clean

    
        # Save the best model 
        best_model.to(device)
        torch.save(best_model.state_dict(), "LAB_09/models/part2/best_model_LSTM_NT_AvSGD.pt")  
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)      
    
    
    print('[LSTM + NT AvSGD]   Test ppl: ', final_ppl)
    
    