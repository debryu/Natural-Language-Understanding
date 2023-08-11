# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

def parse_sentences(parser, sentences):
    productions = []
    for sent in sentences:
        test_parse = parser.parse(sent.split())
        for tree in test_parse:
            print(tree.pretty_print())
            productions += tree.productions()

    return productions