# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
import spacy
import en_core_web_sm
from collections import Counter
import nltk

if __name__ == "__main__":
    # Download the corpus
    nltk.download('gutenberg')
    # Select the corpus
    selected_corpus = 'milton-paradise.txt'
    #Load the corpus
    characters = nltk.corpus.gutenberg.raw(selected_corpus)
    words = nltk.corpus.gutenberg.words(selected_corpus)
    sentences = nltk.corpus.gutenberg.sents(selected_corpus)
     
    #Compute the statistics
    print('Computing the statistics for the original dataset')
    word_per_sent, char_per_word, char_per_sent, longest_sent, longest_sent_chars, longest_word = statistics(characters, words, sentences)

    # Parse using spacy
    nlp = spacy.load("en_core_web_sm",  disable=["tagger", "ner","lemmatizer"])
    characters_spacy = nltk.corpus.gutenberg.raw(selected_corpus)
    # Process the document
    doc = nlp(characters)

    # Words using spacy
    words_spacy = extract_words_spacy(doc)
    # Sentences using spacy
    sentences_spacy = extract_sentences_spacy(doc,nlp)
    
    # Compute the statistics for spacy
    print('Computing the statistics for the spacy processed corpus')
    word_per_sent_spacy, char_per_word_spacy, char_per_sent_spacy, longest_sent_spacy, longest_sent_chars_spacy, longest_word_spacy = statistics(characters_spacy, words_spacy, sentences_spacy)
    
    
    # download NLTK tokenizer
    nltk.download('punkt')
    characters_nltk = nltk.corpus.gutenberg.raw(selected_corpus)
    # Words using nltk
    words_nltk = nltk.word_tokenize(characters)

    # Sentences using nltk
    sentences_nltk = [nltk.word_tokenize(i) for i in nltk.sent_tokenize(characters)]

    # Compute the statistics for nltk
    print('Computing the statistics for the nltk processed corpus')
    words_per_sent_nltk, char_per_word_nltk, char_per_sent_nltk, longest_sent_nltk, longest_sent_chars_nltk, longest_word_nltk = statistics(characters_nltk, words_nltk, sentences_nltk)


    # Print the results
    print(f'Characters in the corpus:\n- {len(characters)} (reference)\n- {len(characters_spacy)} (SpaCy)\n- {len(characters_nltk)} (nltk)')
    print(f'Words in the corpus:\n- {len(words)} (reference)\n- {len(words_spacy)} (spacy)\n- {len(words_nltk)} (nltk)')
    print(f'Sentences in the corpus:\n- {len(sentences)} (reference)\n- {len(sentences_spacy)} (spacy)\n- {len(sentences_nltk)} (nltk)')
    
    print(f'Word per sentence:\n- {word_per_sent} (reference)\n- {word_per_sent_spacy} (spacy)\n- {words_per_sent_nltk} (nltk)')
    print(f'Char per word:\n- {char_per_word} (reference)\n- {char_per_word_spacy} (spacy)\n- {char_per_word_nltk} (nltk)')
    print(f'Char per sentence:\n- {char_per_sent} (reference)\n- {char_per_sent_spacy} (spacy)\n- {char_per_sent_nltk} (nltk)')
    print(f'Longest sentence (in words):\n- {longest_sent} (reference)\n- {longest_sent_spacy} (spacy)\n- {longest_sent_nltk} (nltk)')
    print(f'Longest sentence (in chars):\n- {longest_sent_chars} (reference)\n- {longest_sent_chars_spacy} (spacy)\n- {longest_sent_chars_nltk} (nltk)')
    print(f'Longest word:\n- {longest_word} (reference)\n- {longest_word_spacy} (spacy)\n- {longest_word_nltk} (nltk)')

    # Lowercase the words
    lexicon_ref_lowercased = [w.lower() for w in words]
    lexicon_spacy_lowercased = [w.lower() for w in words_spacy]
    lexicon_nltk_lowercased = [w.lower() for w in words_nltk]

    print(f'Size of different lowercased lexicons:\n - {len(lexicon_ref_lowercased)} (reference)\n - {len(lexicon_spacy_lowercased)} (spacy)\n - {len(lexicon_nltk_lowercased)} (nltk)')

    # Set N
    N = 8
    # Compute the frequency list
    ref_freq_list = Counter(words) 
    spacy_freq_list = Counter(words_spacy) 
    nltk_freq_list = Counter(words_nltk) 
    print('\nFrequency distribution of words:')
    print('----- REFERENCE -----')
    print(nbest(ref_freq_list, N))
    print('\n------- SPACY -------')
    print(nbest(spacy_freq_list, N))
    print('\n------- NLTK --------')
    print(nbest(nltk_freq_list, N))