# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from model import *
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

    clip = 5 # Clip the gradient
    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    vocab_len = len(tokens_id.word2id)
    #                   (embedding size, hidden size, vocab size)
    model = basic_LM_RNN(256, 300, vocab_len, pad_index=tokens_id.word2id["<pad>"]).to(device)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    criterion_train = nn.CrossEntropyLoss(ignore_index=tokens_id.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=tokens_id.word2id["<pad>"], reduction='sum')

    pbar = tqdm(range(1,n_epochs))
    #If the PPL is too high try to change the learning rate
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
        
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
                
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean
                            
    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
    print('Test ppl: ', final_ppl)