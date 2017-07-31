from itertools import tee

import pandas as pd

import numpy as np

from .preprocessor import Preprocessor
from .token_filter import TokenFilter


class Word2Phrase(object):

    def __init__(self, min_count=10, threshold=0.0005, max_phrase_len=2, preprocessor=None):
        self.threshold = threshold
        self.max_phrase_len = max_phrase_len
        self.min_count = min_count

        if preprocessor is None:
            self.preprocessor = Preprocessor(
                token_filter=TokenFilter(filter_stop_words=False))

    def fit(self, documents):
        documents = [document.lower()for document in documents]
        for _ in range(self.max_phrase_len - 1):
            tokens = []
            for document in documents:
                tokens.extend(self.preprocessor.fit(document))

            freqs = {}
            n_tokens = len(tokens)
            for pair in self._pairwise(tokens):
                freqs[pair[0]] = 1 if pair[0] not in freqs else freqs[pair[0]] + 1
                freqs[pair] = 1 if pair not in freqs else freqs[pair] + 1

            pairs = []
            for pair in self._pairwise(tokens):
                freq_left_token = freqs[pair[0]]
                freq_right_token = freqs[pair[1]]
                freq_pair_token = freqs[pair]
                score = (freq_pair_token - self.min_count) / \
                    (freq_left_token * freq_right_token)
                bundle = (freq_left_token, freq_right_token, freq_pair_token)
                if score > self.threshold and all([freq > self.threshold for freq in bundle]):
                    pairs.append(pair)

            for pair in pairs:
                documents = [document.replace(' '.join(pair), '_'.join(pair))
                             for document in documents]
        return set(pairs)

    def _pairwise(self, iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
