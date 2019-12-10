import re
import sys
from nltk.util import ngrams
# from spell import edits2
# from spell import correction
from spellchecker import SpellChecker
import json
from nltk.corpus import words


english_words = set(words.words())


def spellchecker(misspelled):
    spell = SpellChecker()
    return [spell.candidates(word) for word in misspelled]


def generate_n_grams(word, n):
    word = word.lower()
    word = re.sub(r'[^a-zA-Z0-9\s]', ' ', word)
    tokens = [token for token in word.split(" ") if token != ""]
    return list(ngrams(tokens, n))


def correct_sentence(content):
    misspelled = content.split(" ")
    candidates = spellchecker(misspelled)
    print(candidates)


def get_initial_pred(content_json):
    result = ""
    for word in content_json:
        for char_set in word:
            result += char_set[0]
        result += ' '
    return result[:-1]


def swap_char_at_index(word, index, char_to_try):
	return word[:index] + char_to_try + word[index + 1:]


def gen_word_combinations(word, i, word_arr, words_set):
    # print("current word: " + word)
    if (i < len(word)):
        for j in range(len(word)):
            for k in range(4): 
                next_word = swap_char_at_index(word, j, word_arr[j][k])
                words_set.add(next_word)
                gen_word_combinations(next_word, i + 1, word_arr, words_set)
    return words_set


def generate_all_possible_words(content, predicted_text):
    spell = SpellChecker()
    result = set()
    print("\n")
    """
    for char_arr, word in zip(content, predicted_text):
        if word in english_words:
            result.add(word)
        current_set = set()
        current_possibilities = gen_word_combinations(word, 0, char_arr, current_set)
        for possible_word in current_possibilities:
            if possible_word in english_words:
                result.add(possible_word)
    """
    for char_arr, word in zip(content, predicted_text):
        word_set = set()
        word_len = len(word) - 1
        if word in english_words:
            result.add(word)
        word_set.add(word)
        for j in range(0, 4):
            potential_word = swap_char_at_index(word, 0, char_arr[0][j])
            checked = spell.correction(potential_word)
            if checked in english_words:
                result.add(checked)
                word_set.add(checked)
            for k in range(0, 4):
                next_potential = swap_char_at_index(potential_word, word_len, char_arr[word_len][j])
                spell_checked = spell.correction(next_potential)
                if spell_checked in english_words:
                    word_set.add(spell_checked)
                    result.add(spell_checked)
        out_str = ""
        for to_print in word_set:
            out_str += to_print + ", "
        print(out_str[:-2])

    return result


def main():
    print("Refine key prediction...")
    filename = sys.argv[1]
    if len(sys.argv) > 2:
        label_file = sys.argv[2]
    labels = None
    content = json.load(open(filename, 'r'))

    predicted_text = get_initial_pred(content).split()
    spell = SpellChecker()
    # possible_words = generate_all_possible_words(content, predicted_text)
    # print(possible_words)
    # print("num possible words: " + str(len(possible_words)))
    predicted_checked = [spell.correction(word) for word in predicted_text]
    print("predicted text:")
    for word, checked_word in zip(predicted_text, predicted_checked):
        print(word + ", " + checked_word)
    
    #newsentence = spellchecker(content.split(" "))
    #print("Refined sentence:")
    #print(newsentence)


main()
