import re
from nltk.util import ngrams
from spell import edits2
from spell import correction
from spellchecker import SpellChecker


spell = SpellChecker()
misspelled = ["aplpe", "bananb", "orangue"]


for word in misspelled:
    print(word + ": " + correction(word))
    print("Spellchecker correction: " + spell.correction(word))


def generate_n_grams(word, n):
    word = word.lower()
    word = re.sub(r'[^a-zA-Z0-9\s]', ' ', word)
    tokens = [token for token in word.split(" ") if token != ""]
    return list(ngrams(tokens, n))

print(generate_n_grams("Apple banana orange", 2))
