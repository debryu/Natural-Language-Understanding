'''
RUN:
pip install --upgrade threadpoolctl
if you get an error like this:
'NoneType' object has no attribute 'split'
'''
# Import everything from functions.py file
from sklearn.metrics import classification_report
from functions import *
from utils import *
from model import *
from collections import Counter
from sklearn.cluster import KMeans
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import os


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists('dataset/LAB11_part2/td.pkl'):
        td = pickle.load(open('dataset/LAB11_part2/td.pkl', 'rb'))
    else: # Create the dataset from scratch
        #Wrtite the code to load the datasets and to run your functions
        # Print the results
        train_ds = read_data('dataset/LAB11_part2/laptop14_train.txt')
        test_ds = read_data('dataset/LAB11_part2/laptop14_test.txt')
        #print(train_ds[0:3])
        #for i in range(0,5):
            #print(train_ds[i]['sentence']) 
        
        GloVe_embeddings = {}
        print("Loading GloVe embeddings...")
        with open("dataset/LAB11_part2/glove.6B.300d.txt", 'r', encoding='UTF-8') as f:
            total_lines = 400000
            for line in tqdm(f,total=total_lines):
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                GloVe_embeddings[word] = vector
        print("Done.")
        print("Creating dataset from GloVe embeddings...")
        embeds = dataset2glove(train_ds, GloVe_embeddings)
        print("Done.")

        print("Extracting centroids...")
        centroids, wc = extract_centroids(embeds)
        print("Done.")
        centroids = np.array(centroids) #[500, 300]

        #print(centroids.shape)
        # Compute K-Means for the centroids
        kmeans = KMeans(n_clusters=512, random_state=0, n_init=10)
        aspects = kmeans.fit(centroids).cluster_centers_
        #print(aspects)
        aspects = torch.tensor(aspects) #[128, 300]
        #print(aspects.shape)

        print("Integrating dataset with centroids...")
        td = integrate_dataset_with_centroids(embeds, aspects, kmeans)
        print("Done.")
        # Saving the dataset for later use
        pickle.dump(td, open('dataset/LAB11_part2/td.pkl', 'wb'))


    n = 12
    #print(td[n]['ts_raw_tags'])
    #print(td[n]['cluster'])
    
    CEweights = torch.ones(2)
    CEweights[0] = 1/10.
    CEweights[1] = 1
    CE2weights = torch.ones(4)
    CE2weights[0] = 1/12.5
    CE2weights[2] = 1/12.5
    #print(CEweights)
    #print(td[0]['ts_raw_tags'][9])
    #print(td[0]['cluster'][9])
    
    train_dataset = Dataset(td)
    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, collate_fn=batchify)
    model = ABSA()
    
    if os.path.exists('LAB_11/part_2/models/model.pt'):
        model.load_state_dict(torch.load('LAB_11/part_2/models/model.pt'))
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss(weight = CE2weights, reduction='mean')
    loss_fn_aspect = torch.nn.CrossEntropyLoss(weight = CEweights, ignore_index = 3, reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    max_epochs = 35

    for epoch in range(0, max_epochs):
        eloss = []
        
        for emb,lab,cosine,cluster_labels,sent_len in train_loader:
            optimizer.zero_grad()
            #print(emb.shape)
            #print(lab.shape)
            #print(asp[1,-1,:])
            
            #print(cluster_labels)
            # Input size is [batch, seq_len, 300]
            output, aspect = model(emb,cosine, sent_len)
            # Output size is [batch, seq_len, 2]
            aspect = aspect.reshape(-1, 2).cpu()
            cluster_labels = cluster_labels.reshape(-1)
            output = output.reshape(-1, 4).cpu()
            lab= lab.reshape(-1,4).cpu()
            '''
            preds = []
            ground_truths = []
            for batch in range(output.shape[0]):
                predictions = output[batch,0:sent_len[batch],:]
                predictions = predictions.reshape(-1, 2)
                gt = cluster_labels[batch,0:sent_len[batch]]
                preds.append(predictions)
                ground_truths.append(gt)
            output = torch.cat(preds, dim=0)
            cluster_labels = torch.cat(ground_truths, dim=0)
            '''
            #print(output.shape,cluster_labels.shape) #[batch, seq_len, 1]    
            cluster_labels = cluster_labels.to(dtype=torch.long).cpu()
            #print(output.shape, lab.shape, aspect.shape, cluster_labels.shape)
            loss1 = loss_fn(output, lab)
            loss2 = loss_fn_aspect(aspect, cluster_labels)
            loss = loss1 + loss2
            
            #print(loss)
            loss.backward()
            optimizer.step()
            step_lr_scheduler.step()
            eloss.append(loss.detach().cpu().item())
            #print(output.shape)
            #print(output.shape)
        if epoch % 10 == 0:
            print('[TRAIN] loss/epoch:',np.mean(eloss), step_lr_scheduler.get_last_lr())
        #print(wc)
    total_predictions = 0
    total_0 = 0
    
    evaluate(model, train_dataset)


    # Save the model
    torch.save(model.state_dict(), 'LAB_11/part_2/models/model.pt')
    
