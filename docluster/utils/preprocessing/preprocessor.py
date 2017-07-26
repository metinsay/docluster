from bs4 import BeautifulSoup
from nltk import sent_tokenize, word_tokenize
import numpy as np
import string
import stop_words as sw
from ..constants.language import Language

tokenizer = lambda text, stop_words, stem_func: stem(filter_stop_words(tokenize(text),stop_words),stem_func)

class Preprocessor(object):

    def __init__(self, language=Language.english, lower=True, additional_stop_words=[], my_stop_words=None, parse_html=True, filter_punctuation=True, filter_numbers=True, filter_stop_words=True, do_stem=True, do_lemmatize=True):

        self.language = language
        self.lower = lower
        self.stop_words = my_stop_words if my_stop_words else sw.get_stop_words(language.value)
        self.stop_words.extend(additional_stop_words)
        self.parse_html = parse_html
        self.filter_punctuation = filter_punctuation
        self.filter_numbers = filter_numbers
        self.filter_stop_words = filter_stop_words
        self.do_stem = do_stem
        self.do_lemmatize = do_lemmatize

        self.tokens = None
        self.vocab = None

    def fit(self, text):

        tokens = self.tokenize(text)
        tokens = self.stem(tokens) if self.do_stem else tokens
        tokens = self.lemmatize(tokens) if self.do_lemmatize else tokens

        self.tokens = tokens
        self.vocab = list(set(self.tokens))

        return self.tokens

    def tokenize(self, text):

        text = BeautifulSoup(text, "html5lib").get_text() if self.parse_html else text
        sentece_tokens = np.array([words for words in [word_tokenize(sent) for sent in sent_tokenize(text)]])

        if sentece_tokens.shape == (0,):
            return []

        tokens = np.hstack(sentece_tokens)
        does_token_stay = lambda token: (token not in string.punctuation or not self.filter_punctuation) and \
                                        (token not in self.stop_words or not self.filter_stop_words) and \
                                        (not any(char.isdigit() for char in token) or not self.filter_numbers)

        filtered_tokens = filter(does_token_stay, tokens)
        return list(map(lambda token: token.lower(), filtered_tokens)) if self.lower else list(filtered_tokens)

    def stem(self):
        pass

    def lemmatize(self):
        pass
