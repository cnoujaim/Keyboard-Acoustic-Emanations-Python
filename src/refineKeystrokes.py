from spell import edits2
from spell import correction

misspelled = ["appel", "bananb", "orangue"]
for word in misspelled:
    print(word + ": " + correction(word))


