import string

import stop_words as sw
from utils import Language


class TokenFilter(object):

    def __init__(self, language=Language.english, my_stop_words=None, additional_stop_words=[], filter_stop_words=True, filter_punctuation=True, filter_numbers=True, regex=None, filter_contains=[], additional_filters=[]):

        self.stop_words = my_stop_words if my_stop_words else sw.get_stop_words(
            language.value)
        self.stop_words.extend(additional_stop_words)
        self.filter_stop_words = filter_stop_words
        self.filter_punctuation = filter_punctuation
        self.filter_numbers = filter_numbers
        self.regex = regex
        self.filter_contains = filter_contains
        self.additional_filters = additional_filters

        self.filters = []

        if self.filter_stop_words:
            self.filters.append(lambda token: token in self.stop_words)

        if filter_punctuation:
            self.filters.append(lambda token: all(
                char in string.punctuation for char in token))

        if filter_numbers:
            self.filters.append(lambda token: any(char.isdigit() for char in token))

        self.filters.extend(
            list(map(lambda item: (lambda token: item in token), filter_contains)))

        self.filters.extend(additional_filters)

    def fit(self, token):
        token = token.lower()
        for fil in self.filters:
            if fil(token):
                return True
        return False
