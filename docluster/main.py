import string
from multiprocessing import cpu_count
from os import listdir
from os.path import isfile, join

import pandas as pd

import gensim
import nltk
import numpy as np
from bs4 import BeautifulSoup
from core import Preprocessor, Word2Phrase
from utils import (BST, DistanceMetric, FileFetcher, FileSaver, FileType,
                   Language, Trie, TweetFetcher, WikiFetcher)

bst = BST()
bst.insert(15, None)
bst.insert(10, None)
bst.insert(20, None)
bst.insert(13, None)
bst.insert(14, None)
bst.insert(12, None)
bst.insert(11, None)
bst.insert(9, None)
bst.insert(40, None)
bst.insert(17, None)

# Creating the model
# import pickle
#
# with open('/Users/metinsay/Downloads/polyglot-tr.pkl', 'rb') as f:
#     data = pickle.load(f)
#     print(data)

#
# wiki_directories = ['/Users/metinsay/Downloads/wikiextractor-master/text/AA']
# onlyfiles = []
# for directory in wiki_directories:
#     onlyfiles.extend([join(directory, f)
#                       for f in listdir(directory) if isfile(join(directory, f))])
#
#
# documents = []
# for file_ in onlyfiles[:3]:
#     documents.append(BeautifulSoup(open(file_, 'r').read(), "lxml").get_text())
#
#
# w2p = Word2Phrase(max_phrase_len=4)
# w2p.fit(documents)
# print(w2p.phrases)
# documents = w2p.put_phrases_in_documents(documents)
#
# tokens = Preprocessor().fit(documents[0])
# #
# # w2v = Word2Vec(n_workers=8)
# #
# # w2v.fit(documents)
#
#
# def tokenize(text):
#     html = BeautifulSoup(text, "html5lib").get_text()
#     results = [words for words in [nltk.word_tokenize(
#         sent) for sent in nltk.sent_tokenize(html)]]
#
#     results = [[word.lower().replace('-', '').replace('/', '')
#                 for word in sentence] for sentence in results]
#     return [list(filter(lambda token: token not in string.punctuation and not token.isdigit(), sentence)) for sentence in results]
#
#
# clustered_data = documents
# flattened_data = '\n'.join(clustered_data)
# clustered_data = tokenize(flattened_data)
# # print(clustered_data)
#
# model = gensim.models.Word2Vec(clustered_data, size=300,
#                                window=10, min_count=0, workers=8, sg=1, iter=5)
# trie = Trie(tokens)
