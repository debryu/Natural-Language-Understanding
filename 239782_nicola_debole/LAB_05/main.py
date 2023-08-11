# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
import nltk
from nltk.parse.generate import generate
from nltk import Nonterminal
from pcfg import PCFG

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    sentences = [
        "io conosco uno studente",
        #'la grammatica italiana è difficile',
        "tu rispetti la persona",
        "tu osserverai i gatti",
        "loro tolgono i canditi",
        "noi osserveremo un albero",
        "lei ama i gatti",
        "lui rispetta uno studente"
    ]

    italian_rules = [
        'S -> PronVerb Compl',
        'PronVerb -> 1PersSing | 2PersSing | 3PersSing | 1PersPlur | 2PersPlur | 3PersPlur',
        '1PersSing -> "io" Verb1PS',
        '2PersSing -> "tu" Verb2PS',
        '3PersSing -> "lui" Verb3PS | "lei" Verb3PS',
        '1PersPlur -> "noi" Verb1PP',
        '2PersPlur -> "voi" Verb2PP',
        '3PersPlur -> "loro" Verb3PP',

        'Compl -> N',
        'N -> "uno" "studente" | "un" "albero" | "un" "uomo" | "i" "gatti" | "i" "canditi" | "la" "persona"',
        'Verb1PS -> "ho" "mangiato" | "tolgo" | "osserverò" | "amo" | "conoscevo" | "rispetto" | "conosco"',
        'Verb2PS -> "hai" "mangiato" | "togli" | "osserverai" | "ami" | "conoscevi" | "rispetti" | "conosci"',
        'Verb3PS -> "ha" "mangiato" | "toglie" | "osserverà" | "ama" | "conosceva" | "rispetta" | "conosce"',
        'Verb1PP -> "abbiamo" "mangiato" | "togliamo" | "osserveremo" | "amiamo" | "conoscevamo" | "rispettiamo" | "conosciamo"',
        'Verb2PP -> "avete" "mangiato" | "togliete" | "osserverete" | "amate" | "conoscevate" | "rispettate" | "conoscete"',
        'Verb3PP -> "hanno" "mangiato" | "tolgono" | "osserveranno" | "amano" | "conoscevano" | "rispettano" | "conoscono"'
    ]

    italian_grammar = nltk.CFG.fromstring(italian_rules)
    italian_parser = nltk.ChartParser(italian_grammar)

    productions_ita = parse_sentences(italian_parser, sentences)
    print(productions_ita)

    S = Nonterminal('S')
    Verb = Nonterminal('Verb')  
    Compl = Nonterminal('Compl')

    it_grammar_weighted_rules = nltk.induce_pcfg(S, productions_ita)
    it_grammar_weighted_rules.productions()
    
    print('------------------------------------------------------')
    print('Generating 10 sentences:')
    for sent in generate(it_grammar_weighted_rules, start = S ,depth=5, n=10):
        print(sent)

    # Converting the rules to string and then splitting them in a list
    string_grammar_ita = str(it_grammar_weighted_rules).split('\n')
    # Removing the first element of the list (the description)
    string_grammar_ita = string_grammar_ita[1:]
    
    print('------------------------------------------------------')
    print('Generating 10 sentences using PCFG.generate():')
    grammar_ita = PCFG.fromstring(string_grammar_ita)
    for sent in grammar_ita.generate(10):
        print(sent)