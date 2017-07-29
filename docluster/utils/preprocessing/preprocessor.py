from bs4 import BeautifulSoup
from nltk import sent_tokenize, word_tokenize
import numpy as np
from ..constants.language import Language
from .token_filter import TokenFilter

tokenizer = lambda text, stop_words, stem_func: stem(filter_stop_words(tokenize(text),stop_words),stem_func)

class Preprocessor(object):

    def __init__(self, language=Language.english, lower=True, parse_html=True, token_filter=TokenFilter(language=Language.english), do_stem=False, do_lemmatize=False):

        self.language = language
        self.lower = lower
        self.parse_html = parse_html
        self.token_filter =  token_filter if token_filter else None
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

        filtered_tokens = filter(lambda token: not self.token_filter.fit(token), tokens) if self.token_filter else tokens
        return list(map(lambda token: token.lower(), filtered_tokens)) if self.lower else list(filtered_tokens)

    def stem(self, tokens):
        pass

    def lemmatize(self, tokens):
        pass
