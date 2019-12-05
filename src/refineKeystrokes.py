import re
import sys
from nltk.util import ngrams
# from spell import edits2
# from spell import correction
from spellchecker import SpellChecker


def spellchecker(misspelled):
    spell = SpellChecker()
    cands = []

    for word in misspelled:
        corrected.append(spell.candidates(word))

    return cands


def generate_n_grams(word, n):
    word = word.lower()
    word = re.sub(r'[^a-zA-Z0-9\s]', ' ', word)
    tokens = [token for token in word.split(" ") if token != ""]
    return list(ngrams(tokens, n))

def correct_sentence(content):
    misspelled = content.split(" ")
    candidates = spellchecker(misspelled)
    

def main():
    print("Refine key prediction...")
    filename = sys.argv[1]
    if len(sys.argv) > 2:
        label_file = sys.argv[2]
    content = None
    labels = None
    with open(filename, 'r') as f:
        content = f.read().strip('\n')

    print("Original sentence:")
    print(content)

    newsentence = spellchecker(content)
    print("Refined sentence:")
    print(newsentence)


main()
