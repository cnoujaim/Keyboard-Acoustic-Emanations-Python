from spell import edits2
from spell import correction
from spellchecker import SpellChecker

spell = SpellChecker()
misspelled = ["aplpe", "bananb", "orangue"]

for word in misspelled:
    print(word + ": " + correction(word))
    print("Spellchecker correction: " + spell.correction(word))

