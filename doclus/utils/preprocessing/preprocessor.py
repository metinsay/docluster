from bs4 import BeautifulSoup
import nltk
import numpy as np
import string

def tokenize(text, includeNumbers=False):
    html_free_text = BeautifulSoup(text, "html5lib").get_text()
    results = np.array([words for words in [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(html_free_text)]])

    if results.shape == (0,):
        return []

    flattened_results = np.hstack(results)
    return list(filter(lambda token: token not in string.punctuation and (not token.isdigit() or includeNumbers), flattened_results))

def stem(tokens, stem_func):
    return list(map(lambda token: stem_func(token), tokens))

def filter_stop_words(tokens, stop_words):
    return list(filter(lambda token: not token in stop_words, tokens))

tokenizer = lambda text, stop_words, stem_func: stem(filter_stop_words(tokenize(text),stop_words),stem_func)
