# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Import everything from functions.py file
from functions import *
from utils import *
from model import *
from sklearn.model_selection import train_test_split
from functions import *
from model import *
from utils import *
import os
import random
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import copy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    print("Initializing the kfold...")
    kfold_results_subjectivity = []
    kfold_results_vader = []
    kfold_SVM_results = []
    kfold_SVM_aug_results = []
    kfold_results_longformer = []
    for i, (train_indexes, test_indexes) in enumerate(kfold.split(subjectivity_ds)):
        print(f"-----------------------[ SUBJECTIVITY Fold: {i} ]-----------------------")

        '''
        ----------------------------------------
            DATASET CREATION FOR SUBJECTIVITY
        ----------------------------------------
        '''
        train_ds = [subjectivity_ds[i] for i in train_indexes]
        train_labels = labels_subjectivity[train_indexes]
        test_ds = [subjectivity_ds[i] for i in test_indexes]
        test_labels = labels_subjectivity[test_indexes]
       
        # Create dev dataset
        train_ds, dev_ds, train_labels, dev_labels = train_test_split(train_ds, train_labels, test_size=0.1, random_state=None, shuffle=True, stratify=train_labels)

        print("Creating the dataset...")
        # Create our datasets
        train_ds = SubjectivityDataset(train_ds, train_labels)
        dev_ds = SubjectivityDataset(dev_ds, dev_labels)
        test_ds = SubjectivityDataset(test_ds, test_labels)
        
        print("Creating the dataloader...")
        # Dataloader instantiation
        train_loader = DataLoader(train_ds, batch_size=64, collate_fn=collate_fn,  shuffle=True)
        dev_loader = DataLoader(dev_ds, batch_size=32, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=32, collate_fn=collate_fn)

        '''
        ----------------------------------------
        START TRAINING SUBJECTIVITY
        ----------------------------------------
        '''
        use_different_lr = False
        load_model = True
        train = False
        

        lr = 0.0001 # learning rate
        clip = 5 # Clip the gradient

        vocab_len = len(bert_tokenizer.vocab)

        sub_model = Subjectivity(vocab_len, pad_index=PAD_TOKEN).to(device)

        if(load_model):
            if use_different_lr:
                sub_model.load_state_dict(torch.load(f'LAB_11/part_1/models/model_multi_lr_FOLD{i}.pt')) 
            else:
                sub_model.lastLayer = torch.load(f'LAB_11/part_1/models/sub_model_FOLD{i}.pt').to(device)
        
            
        params = list(sub_model.parameters())
        bert_params = params[:199]
        head_params = params[199:]
        
        
        if(use_different_lr):
            # Use two different optimizer
            bert_optimizer = optim.Adam(bert_params, lr=0.00000001)
            head_optimizer = optim.Adam(head_params, lr=0.0001)
        else:
            optimizer = optim.Adam(sub_model.lastLayer.parameters(), lr=lr)

        criterion_subjectivity = nn.CrossEntropyLoss()
    
        
        n_epochs = 100
        patience = 3
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        if train:
            saved_model = copy.deepcopy(sub_model).to('cpu')
            for x in range(1,n_epochs):
                if(use_different_lr):
                    loss = train_loop_multiple_lr(train_loader, bert_optimizer, head_optimizer, criterion_subjectivity, sub_model, 'tok_sent_padded')
                else:
                    loss = train_loop(train_loader, optimizer, criterion_subjectivity, sub_model, 'tok_sent_padded')
                if x % 5 == 0:
                    sampled_epochs.append(x)
                    losses_train.append(np.asarray(loss).mean())
                    results_dev, loss_dev = eval_loop(dev_loader, criterion_subjectivity, sub_model, 'tok_sent_padded')                            
                    losses_dev.append(np.asarray(loss_dev).mean())
                    f1 = results_dev['macro avg']['f1-score']
                    if f1 > best_f1:
                        best_f1 = f1
                        saved_model = copy.deepcopy(sub_model.lastLayer).to('cpu')
                    else:
                        patience -= 1
                    if patience <= 0: # Early stoping with patient
                        break 
            # Save the best model
            if use_different_lr:
                torch.save(saved_model, f'LAB_11/part_1/models/sub_model_multi_lr_FOLD{i}.pt') 
            else:
                torch.save(saved_model, f'LAB_11/part_1/models/sub_model_FOLD{i}.pt')
        
        # Evaluate
        results_test, intent_test = eval_loop(test_loader, criterion_subjectivity, sub_model, 'tok_sent_padded')
        print(results_test)
        kfold_results_subjectivity.append(results_test)
        
        # COMMENT THE BREAK FOR THE FINAL TRAINING WITH 10 FOLDS
        #break

    for i, (train_indexes, test_indexes) in enumerate(kfold.split(polarity_ds)):
        print(f"-----------------------[ POLARITY Fold: {i} ]-----------------------")

        '''
        ----------------------------------------
            DATASET CREATION FOR POLARITY
        ----------------------------------------
        '''
        train_ds = [polarity_ds[i] for i in train_indexes]
        train_labels = labels_polarity[train_indexes]
        test_ds = [polarity_ds[i] for i in test_indexes]
        test_labels = labels_polarity[test_indexes]

        # Create dev dataset
        train_ds, dev_ds, train_labels, dev_labels = train_test_split(train_ds, train_labels, test_size=0.1, random_state=None, shuffle=True, stratify=train_labels)

        # Now create the test dataset (with labels) where the objective sentences are removed for the longformer model
        # using the corresponding subjectivity model
        # It is called augmented because it is the original dataset with the objective sentences removed
        print("Computing the subjectivity scores and creating the augmented test dataset by removing the objective sentences...")
        augmented_test_ds, original_test_ds, test_subjectivity_logits = subjectivity_extraction(test_ds, 0)
        augmented_train_ds, original_train_ds, train_subjectivity_logits = subjectivity_extraction(train_ds, 0)
        # Dev dataset is skipped since it is not used for the SVM model and also for the longformer model

        print("Creating the dataset...")
        # Create our datasets
        train_ds = PolarityDataset(train_ds, train_labels, remove_subj = False)
        dev_ds = PolarityDataset(dev_ds, dev_labels, remove_subj = False)
        test_ds = PolarityDataset(test_ds, test_labels, remove_subj = False)
        #augmented_test_ds = PolarityDataset(augmented_test_ds, test_labels, remove_subj = False)
        #augmented_train_ds = PolarityDataset(augmented_train_ds, train_labels, remove_subj = False)
        #augmented_dev_ds = PolarityDataset(augmented_dev_ds, dev_labels, remove_subj = False)
        #for sample in dev_ds:
        #    print(sample['tokenized_reviews'])

        print("Creating the dataloaders...")
        # Dataloader instantiation used for the Vader model
        train_loader = DataLoader(train_ds, batch_size=64, collate_fn=collate_vader_fn, shuffle=True)
        dev_loader = DataLoader(dev_ds, batch_size=32, collate_fn=collate_vader_fn)
        test_loader = DataLoader(test_ds, batch_size=32, collate_fn=collate_vader_fn)
        
        # Dataloader instantiation used for the Longformer model
        # Batch size is 1 because otherwise the CUDA out of memory error is thrown
        # That is why we use the gradient accumulation
        train_longformer_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
        dev_longformer_loader = DataLoader(dev_ds, batch_size=1)
        test_longformer_loader = DataLoader(test_ds, batch_size=1)
        #augmented_test_longformer_loader = DataLoader(augmented_test_ds, batch_size=1)
        #augmented_train_longformer_loader = DataLoader(augmented_train_ds, batch_size=1, shuffle=True)
        #augmented_dev_longformer_loader = DataLoader(augmented_dev_ds, batch_size=1)
        
        print("Computing the Vader scores...")
        vader_result = eval_vader(test_loader)
        kfold_results_vader.append(vader_result)
        
        '''
        ----------------------------------------
              TRAINING SVM POLARITY MODEL
        ----------------------------------------
        '''
        print("Training and evaluating the SVM model...")
        # Compute the features using tf-ifd for each document
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(train_ds[:]['stringed_document'])
        vectorizer.get_feature_names_out()
        X_test = vectorizer.transform(test_ds[:]['stringed_document'])
        
        # Define the SVM model (Support Vector Classification)
        svm = SVC(C = 10000, degree=6, tol = 0.00001, gamma='auto')
        svm.fit(X_train, train_ds[:]['label'])
        predictions = svm.predict(X_test)        
        SVM_result = classification_report(test_ds[:]['label'], predictions, zero_division=False, output_dict=True)
        kfold_SVM_results.append(SVM_result)
        

        '''
        -------------------------------------------------------
              TRAINING PIPELINE SMV + SUBJ POLARITY MODEL
        -------------------------------------------------------
        '''
        print("Training and evaluating the pipeline model...")
        # Now we test the pipeline in which we remove the objective sentences from the dataset 
        # The augmenting is done by using the minimum cut of the graph composed by nodes that represents
        # The sentences of the documents with one source (0) and one sink (1)
        aug_train_ds, stringed_aug_train_ds = augment_dataset(original_train_ds, train_subjectivity_logits)
        aug_test_ds, stringed_aug_test_ds = augment_dataset(original_test_ds, test_subjectivity_logits)
        
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(stringed_aug_train_ds)
        vectorizer.get_feature_names_out()
        X_test = vectorizer.transform(stringed_aug_test_ds)
        svm = SVC(C = 10000, degree=6, tol = 0.00001, gamma='auto')
        svm.fit(X_train, train_ds[:]['label'])
        predictions = svm.predict(X_test)
        SVM_aug_result = classification_report(test_ds[:]['label'], predictions, zero_division=False, output_dict=True)
        kfold_SVM_aug_results.append(SVM_aug_result)
        
        
        '''
        ----------------------------------------
        START TRAINING POLARITY MODEL
        We skip the following code because it is not used
        And also take too much time to train
        It is the longformer model for polarity classification
        ----------------------------------------
        '''
        continue
        load_model = True
        train = False

        lr = 0.001 # learning rate
        clip = 5 # Clip the gradient

        # Initialize the model
        gradient_accumulation_steps = 64
        pol_model = PolarityLongformer().to(device)
        if(load_model):
            pol_model.load_state_dict(torch.load(f'LAB_11/part_1/models/polarity_longformer_model_FOLD0{i}.pt'))
        saved_model = copy.deepcopy(pol_model).to('cpu')
        criterion_subjectivity = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(pol_model.classifier.parameters(), lr=lr)
    
        
        n_epochs = 100
        patience = 3
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        if train:
            for x in range(1,n_epochs):
                loss = train_loop_longformer(train_longformer_loader, optimizer, criterion_subjectivity, pol_model, 'tokenized_reviews')
                if x % 2 == 0:
                    sampled_epochs.append(x)
                    losses_train.append(np.asarray(loss).mean())
                    results_dev = eval_loop_longformer(dev_longformer_loader, pol_model, 'tokenized_reviews')                           
                    f1 = results_dev['macro avg']['f1-score']
                    #print(f1)
                    #print(f'Epoch: {x} | F1: {f1}')
                    if f1 > best_f1:
                        best_f1 = f1
                        saved_model = copy.deepcopy(pol_model).to('cpu')
                    else:
                        patience -= 1
                    if patience <= 0: # Early stoping with patient
                        break # Not nice but it keeps the code clean
            # Save the best model
            torch.save(saved_model.state_dict(), f'LAB_11/part_1/models/polarity_longformer_model_FOLD{i}.pt') 
        
        # Evaluate
        saved_model = saved_model.to(device)
        saved_model.eval()
        results_test = eval_loop_longformer(test_longformer_loader, saved_model, 'tokenized_reviews') 
        print('Longformer res')
        print(results_test)
        kfold_results_longformer.append(results_test)

        
        
        # REMOVE THE BREAK FOR THE FINAL TRAINING WITH 10 FOLDS
        break

    print("___________________________________________________________")
    print("---------> [ SUMMARY of the average for F1 scores ] <----------")
    f1_scores_subjectivity = [ x['macro avg']['f1-score'] for x in kfold_results_subjectivity ]
    f1_scores_SVM = [ x['macro avg']['f1-score'] for x in kfold_SVM_results ]
    f1_scores_SVM_aug = [ x['macro avg']['f1-score'] for x in kfold_SVM_aug_results ]
    f1_scores_vader = [ x['macro avg']['f1-score'] for x in kfold_results_vader ]
    print(f"- SUBJECTIVITY MODEL: {np.mean(f1_scores_subjectivity)} \n- SVM: {np.mean(f1_scores_SVM)} \n- PIPELINE (SVM + MINIMUM CUT): {np.mean(f1_scores_SVM_aug)} \n- VADER: {np.mean(f1_scores_vader)}")
