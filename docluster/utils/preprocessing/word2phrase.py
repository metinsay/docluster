from itertools import tee

import pandas as pd

import numpy as np

from .preprocessor import Preprocessor
from .token_filter import TokenFilter


class Word2Phrase(object):

    def __init__(self, min_count=20, threshold=0.000, max_phrase_len=2, delimiter='_', preprocessor=None):
        self.threshold = threshold
        self.max_phrase_len = max_phrase_len
        self.min_count = min_count
        self.delimiter = delimiter

        if preprocessor is None:
            additional_filters = [lambda token: len(token) == 1]
            self.preprocessor = Preprocessor(lower=False,
                                             token_filter=TokenFilter(filter_stop_words=False, additional_filters=additional_filters))

        self.phrases = set()
        self.phrase_score = {}

    def fit(self, documents):
        tokens = []
        freqs = {}
        n_tokens = len(tokens)

        for document in documents:
            tokens.extend(self.preprocessor.fit(document))

        for token in tokens:
            freqs[token] = 1 if token not in freqs else freqs[token] + 1

        for phrase_len in range(self.max_phrase_len, 1, -1):

            token_groups = self._groupwise(tokens, phrase_len)
            for phrase in token_groups:
                freqs[phrase] = 1 if phrase not in freqs else freqs[phrase] + 1

            for index, phrase in enumerate(token_groups):
                indi_freqs = np.array([freqs[phrase[index]]
                                       for index in range(phrase_len)])
                freq_pair_token = freqs[phrase]
                if freq_pair_token > self.min_count:
                    score = (freq_pair_token - self.min_count) / np.prod(indi_freqs)
                    custom_threshold = self.threshold ** (phrase_len - 1)

                    if score > custom_threshold:
                        if all([self.delimiter.join(phrase) not in prev_phrase for prev_phrase in list(self.phrases)]):
                            self.phrases.add(self.delimiter.join(phrase))

        return self.phrases

    def put_phrases_in_documents(self, documents):
        for phrase in self.phrases:
            no_deli_phrase = phrase.replace(self.delimiter, ' ')
            documents = [document.replace(no_deli_phrase, phrase)
                         for document in documents]
        return documents

    def _groupwise(self, iterable, n_groups):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        # a, b = tee(iterable)
        # next(b, None)
        # return zip(a, b)
        return [tuple(iterable[i:i + n_groups]) for i in range(len(iterable) - n_groups - 1)]
