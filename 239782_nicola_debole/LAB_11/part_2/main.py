# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

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

if __name__ == "__main__":

    

    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    train_ds = read_data('dataset/LAB11_part2/laptop14_train.txt')
    test_ds = read_data('dataset/LAB11_part2/laptop14_test.txt')
    print(train_ds[0:3])
    for i in range(0,5):
        print(train_ds[i]['sentence']) 

    model = ABSA()
    weights = torch.tensor([1/12.44776995, 1.0, 1.0, 1.0])
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    criterion_aspect = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    embeds = dataset2glove(train_ds, GloVe_embeddings)
    centroids, wc = extract_centroids(embeds)
    centroids = np.array(centroids) #[500, 300]
    print(centroids.shape)
    # Compute K-Means for the centroids
    kmeans = KMeans(n_clusters=128, random_state=0, n_init=10)
    aspects = kmeans.fit(centroids).cluster_centers_
    #print(aspects)
    aspects = torch.tensor(aspects) #[128, 300]
    print(aspects.shape)
    td = integrate_dataset_with_centroids(embeds, aspects, kmeans)
    n = 12
    print(td[n]['ts_raw_tags'])
    print(td[n]['cluster'])
    
    CEweights = torch.ones(2)
    CEweights[0] = 1/12.5
    print(CEweights)
    print(td[0]['ts_raw_tags'][9])
    print(td[0]['cluster'][9])
    
    train_dataset = Dataset(td)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=batchify)
    model = ASPECT_DETECTION(n_clusters=128)
    loss_fn = torch.nn.CrossEntropyLoss(weight = CEweights, ignore_index = 3, reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.01)
    max_epochs = 100

    for epoch in range(0, max_epochs):
        eloss = []
        for emb,lab,cosine,cluster_labels,sent_len in train_loader:
            optimizer.zero_grad()
            #print(emb.shape)
            #print(lab.shape)
            #print(asp[1,-1,:])
            
            #print(cluster_labels)
            # Input size is [batch, seq_len, 300]
            output = model(cosine)
            # Output size is [batch, seq_len, 2]
            output = output.reshape(-1, 2)
            cluster_labels = cluster_labels.reshape(-1)
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
            cluster_labels = cluster_labels.to(dtype=torch.long)
            loss = loss_fn(output, cluster_labels)
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
    
    for i,(emb,lab,cosine,cluster_labels,sent_len) in enumerate(train_loader):
            #print('---')
            output = model(cosine)
            predictions = torch.argmax(output, dim=2)
            #print(predictions.shape)
            #print(cluster_labels.shape)
            gt = torch.argmax(cluster_labels, dim=1)
            for p,g in zip(predictions[n,:sent_len[n]], cluster_labels[n,:sent_len[n]]):
                print(p,g)
                if g == 0:
                    total_0 += 1
                total_predictions += 1
            #print(predictions.shape)
            cluster_labels = cluster_labels
            if(i == 4):
                break
            #print(output.shape,cluster_labels.shape) #[batch, seq_len, 1]    
    print(total_0, total_predictions, total_0/total_predictions)
    asdasd
    '''
    arrays = []
    for point in centroids:
        array = point.numpy()
        arrays.append(array)
    data_array = np.array(arrays)
    tsne = TSNE(n_components=3, random_state=42)
    embedded_data = tsne.fit_transform(data_array)

    # Plot the results
    plt.scatter(embedded_data[:, 0], embedded_data[:, 1])
    plt.title('t-SNE Visualization of Tensors')
    plt.show()      
    '''
    for element in embeds:
        features = []
        for i,word in enumerate(element['words']):
            similarity_fn = torch.nn.CosineSimilarity(dim=1)
            score = similarity_fn(torch.tensor(element['emb'][i]), aspects)
            features.append(score)
        element['features'] = features
    print(embeds[0])
    asd

    labels = []
    for i in embeds:
        for tag in i['ts_raw_tags']:
            labels.append(tag)

    class_distribution = Counter(labels)
    no_tag = class_distribution['O']
    positive = class_distribution['T-POS']
    neutral = class_distribution['T-NEU']
    negative = class_distribution['T-NEG']
    print(positive, neutral, negative, no_tag)
    
    embeds_val = dataset2glove(test_ds, GloVe_embeddings)
    train_dataset = Dataset(embeds)
    test_dataset = Dataset(embeds_val)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=batchify)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, collate_fn=batchify)
    validate_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print('embed0',embeds[0])
    print(embeds[0]['emb'].shape, len(embeds[0]['ts_raw_tags']))

    if False:
        model.eval()
        model.load_state_dict(torch.load('/Users/debryu/Desktop/bestmodel.pt'))
        total = 0
        correct = 0
        all_preds = []
        all_gt = []
        with torch.no_grad():
            for i, (emb, label,true_aspect) in enumerate(validate_loader):
                
                output,aspect = model(emb)
                preds = torch.argmax(output, dim=2).squeeze(0).numpy()
                output = output.reshape(-1, 4)
                label = label.reshape(-1,4)
                aspect = aspect.reshape(-1,2)
                true_aspect = true_aspect.reshape(-1,2)
                loss = criterion(output, label)
                #print(output)
                #print(torch.argmax(output, dim=2))
                
                ind_preds = []
                for pred in preds:
                    if pred == 0:
                        ind_preds.append('O')
                        all_preds.append('O')
                    elif pred == 1:
                        ind_preds.append('T-POS')
                        all_preds.append('T-POS')
                    elif pred == 2:
                        ind_preds.append('T-NEU')
                        all_preds.append('T-NEU')
                    elif pred == 3:
                        ind_preds.append('T-NEG')
                        all_preds.append('T-NEG')
                
                #print(ind_preds)
                #print(embeds_val[i]['ts_raw_tags'])
                #print(embeds_val[i]['words'])
                for p,gt,w in zip(ind_preds, embeds_val[i]['ts_raw_tags'], embeds_val[i]['words']):
                    print(w,p,gt)

                for el in embeds_val[i]['ts_raw_tags']:
                    all_gt.append(el)
                
                
                
        print(classification_report(all_gt, all_preds))
        asd
    patience = 10
    min_loss = 100000
    for epochs in tqdm(range(0, 100000)):
        losses = []
        model.train()
        for i, (emb, label, true_aspects) in enumerate(train_loader):
            optimizer.zero_grad()
            output, aspect = model(emb)
            output = output.reshape(-1, 4)
            label = label.reshape(-1,4)
            aspect = aspect.reshape(-1,2)
            true_aspects = true_aspects.reshape(-1,2)
            #print(output.shape, label.shape)
            loss1 = criterion(output,label)
            loss2 = criterion_aspect(aspect, true_aspects)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print('Epoch loss:', np.mean(losses))
        model.eval()
        with torch.no_grad():
            for i, (emb, label, true_aspects) in enumerate(test_loader):
                output,aspect = model(emb)
                output = output.reshape(-1, 4)
                label = label.reshape(-1,4)
                aspect = aspect.reshape(-1,2)
                true_aspects = true_aspects.reshape(-1,2)
                loss1 = criterion(output,label)
                loss2 = criterion_aspect(aspect, true_aspects)
                loss = loss1 + loss2
                losses.append(loss.item())
            vloss = np.mean(losses)
            print('Eval loss:', vloss)
            if vloss < min_loss:
                min_loss = vloss
                patience = 10
                torch.save(model.state_dict(), '/Users/debryu/Desktop/bestmodel.pt')
            else:
                patience -= 1
        if patience == 0:
            break
    
    torch.save(model.state_dict(), '/Users/debryu/Desktop/model.pt')
    '''
    points = extract_latent_points(embeds)
    print(len(points))
    arrays = []
    for point in points:
        array = point.numpy()
        arrays.append(array)
    data_array = np.array(arrays)
    tsne = TSNE(n_components=3, random_state=42)
    embedded_data = tsne.fit_transform(data_array)

    # Plot the results
    #plt.scatter(embedded_data[:, 0], embedded_data[:, 1])
    #plt.title('t-SNE Visualization of Tensors')
    #plt.show()      
    '''
    