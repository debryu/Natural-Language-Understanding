# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SKLEARN_STOP_WORDS

NUMBER_OF_SPLITS = 5


def count_vectorize(data, number_of_splits=NUMBER_OF_SPLITS):
    split_num = 1
    #INITIALIZE THE STRATIFIED KFOLD
    stratified_split = StratifiedKFold(n_splits=number_of_splits, shuffle=True)
    #INITIALIZE THE CLASSIFIER
    classifier = LinearSVC(max_iter=20000)
    for train, test in stratified_split.split(data.data, data.target):
        #Initialize the train samples
        data_train = [data.data[i] for i in train]
        print(f'SPLIT {split_num}:')
        vectorizer = CountVectorizer(strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, stop_words=None, max_df=1.0, min_df=1)
        vocabulary = vectorizer.fit(data_train)
        print('Vocabulary len:', len(vocabulary.get_feature_names_out()))
        
        #Vectorize the sample
        embeddings = vectorizer.transform(data_train)
        
        #Initialize the labels
        y = [data.target[i] for i in train]
        
        #Train the model
        classifier.fit(embeddings, y)

        #Initialize the test samples
        data_test = [data.data[i] for i in test]
        #Vectorize every sample
        X_test = vectorizer.transform(data_test)

        #Predict labels from test samples
        hyps = classifier.predict(X_test)
        refs = [data.target[i] for i in test]
        
        #Evaluate the model
        report = classification_report(refs, hyps, target_names=data.target_names)
        print(report)
        split_num += 1

def tfidf(data, number_of_splits=NUMBER_OF_SPLITS, LOWERCASING=True, STOPWORDS=None, MIN_DF=0.00, MAX_DF=1):
    tfidf_vectorizer = TfidfVectorizer(strip_accents=None, lowercase=LOWERCASING, preprocessor=None, tokenizer=None, stop_words=STOPWORDS, max_df=MAX_DF, min_df=MIN_DF)
    split_num = 1
    #INITIALIZE THE STRATIFIED KFOLD
    stratified_split = StratifiedKFold(n_splits=number_of_splits, shuffle=True)
    #INITIALIZE THE CLASSIFIER
    classifier = LinearSVC(C=2, max_iter=20000)
    for train, test in stratified_split.split(data.data, data.target):
        #Initialize the train samples
        data_train = [data.data[i] for i in train]
        #Initialize the test samples
        data_test = [data.data[i] for i in test]

        #Here we can just use the function "fit_transform" to do the 2 steps in only one but i kept them separated for clarity
        #Initialize the vocabulary
        tfidf_vectorizer.fit(data_train)
        #Vectorize the samples
        embeddings = tfidf_vectorizer.transform(data_train)

        #Initialize the labels
        y = [data.target[i] for i in train]
        
        #Train the model
        classifier.fit(embeddings, y)

        #Vectorize every sample
        X_test = tfidf_vectorizer.transform(data_test)

        #Predict labels from test samples
        predictions = classifier.predict(X_test)
        refs = [data.target[i] for i in test]

        
        #Evaluate the model
        report = classification_report(refs, predictions, target_names=data.target_names)
        print(report)