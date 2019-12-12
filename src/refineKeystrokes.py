import re
import sys
from nltk.util import ngrams
# from spell import edits2
# from spell import correction
from spellchecker import SpellChecker
import json
from nltk.corpus import words
from aspell import Speller


english_words = set(words.words())
aspeller = Speller('lang', 'en')
all_words_set = set()


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
    if (i < len(word)):
        for j in range(len(word)):
            for k in range(4): 
                next_word = swap_char_at_index(word, j, word_arr[j][k])
                all_words_set.add(next_word)
                gen_word_combinations(next_word, i + 1, word_arr, words_set)
    return words_set


def generate_all_words(content, predicted_text):
    for char_arr, word in zip(content, predicted_text):
        if english_words.check(word):
            all_words_set.add(word)
        current_set = set()
        current_possibilities = gen_word_combinations(word, 0, char_arr, current_set)
        for possible_word in current_possibilities:
            if english_words.check(possible_word):
                all_words_set.add(possible_word)
    return all_words_set



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
        if english_words.check(word):
            result.add(word)
        word_set.add(word)
        for j in range(0, 4):
            potential_word = swap_char_at_index(word, 0, char_arr[0][j])
            if english_words.check(potential_word):
                result.add(potential_word)
                word_set.add(potential_word)
            else:
                suggestions = english_words.suggest(potential_word)
                if len(suggestions) > 0:
                    result.add(suggestions[0])
                    word_set.add(suggestions[0])
                else:
                    result.add(potential_word)
                    word_set.add(potential_word)
            for k in range(0, 4):
                next_potential = swap_char_at_index(potential_word, word_len, char_arr[word_len][j])
                spell_checked = spell.correction(next_potential)
                if english_words.check(next_potential):
                    word_set.add(next_potential)
                    result.add(next_potential)
                else:
                    suggestions = english_words.suggest(potential_word)
                    if len(suggestions) > 0:
                        result.add(suggestions[0])
                        word_set.add(suggestions[0])
                    else:
                        result.add(potential_word)
                        word_set.add(potential_word)
    
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
    
    predicted_checked = []
    for word in predicted_text:
        if aspeller.check(word) or word in english_words:
            predicted_checked.append(word)
        elif len(word) < 5:
            pyspell = spell.correction(word)
            if aspeller.check(pyspell) or pyspell in english_words:
                predicted_checked.append(pyspell)
            else:
                predicted_checked.append(word)
        else:
            pyspell = spell.correction(word)
            if aspeller.check(pyspell) or pyspell in english_words:
                predicted_checked.append(pyspell)
            else:
                suggestions = aspeller.suggest(word)
                if len(suggestions) > 0:
                    predicted_checked.append(suggestions[0])
                else:
                    predicted_checked.append(word)
    correct = open('../recordings/typingpractice2', 'r').read().split()
    # predicted_checked = generate_all_words(content, predicted_text)
    # predicted_checked = [spell.correction(word) for word in predicted_text]
    print("predicted text:")
    for word, checked_word in zip(predicted_text, predicted_checked):
        print(word + ", " + checked_word)
    print("\n\n")
    
    for checked_w, correct_w in zip(predicted_checked, correct):
        print(checked_w + ", " + correct_w)
    num_correct = sum([w1 == w2 for w1, w2 in zip(predicted_checked, correct)])
    print("num_correct: " + str(num_correct) + " / " + str(len(correct)))


main()
