# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

# Define the function to compute the statistics (copied from above)
def statistics(chars, words, sents):
    word_lens = [len(word) for word in words] # Add word lens 
    sent_lens = [len(sent) for sent in sents] # Add sentence lens
    chars_in_sents = [len(''.join(sent)) for sent in sents] # Add char lens
    
    word_per_sent = round(sum(sent_lens) / len(sents))
    char_per_word = round(sum(word_lens) / len(words))
    char_per_sent = round(sum(chars_in_sents) / len(sents))
    
    longest_sentence = max(sent_lens) # max(...)
    longest_sentence_chars = max(chars_in_sents)
    longest_word = max(word_lens) # max(...)
    
    return word_per_sent, char_per_word, char_per_sent, longest_sentence, longest_sentence_chars, longest_word


def extract_words_spacy(doc):
    words_spacy = [word.text for word in doc]
    return words_spacy

def extract_sentences_spacy(doc,nlp):
    sentences_spacy = [sent.text for sent in doc.sents]
    parsed_words_in_sents_spacy = []
    for s in sentences_spacy:
        sent_tokenized = nlp(s)
        words_in_sent = [w.text for w in sent_tokenized]
        parsed_words_in_sents_spacy.append(words_in_sent)
        

    return parsed_words_in_sents_spacy

def nbest(d, n=1):
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:n])
