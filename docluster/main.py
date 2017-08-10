import string
from multiprocessing import cpu_count
from os import listdir
from os.path import isfile, join

import pandas as pd
from sklearn.decomposition import PCA

# import gensim
import nltk
import numpy as np
from bs4 import BeautifulSoup
from core import Flow, KMeans, Preprocessor, TfIdf, Word2Phrase
from utils import (BST, DistanceMetric, FileFetcher, FileSaver, FileType, Heap,
                   Language, Trie, TweetFetcher, WikiFetcher)

# Creating the model
# import pickle


wiki_directories = ['/Users/metinsay/Downloads/wikiextractor-master/text/AA']
onlyfiles = []
for directory in wiki_directories:
    onlyfiles.extend([join(directory, f)
                      for f in listdir(directory) if isfile(join(directory, f))])


documents = []
for file_ in onlyfiles[:3]:
    documents.append(BeautifulSoup(open(file_, 'r').read(), "lxml").get_text())

w2p = Word2Phrase(min_count=10, max_phrase_len=2)
tfidf = TfIdf(n_words=10000, preprocessor=Preprocessor())
data = np.random.rand(40, 100)
print(data.shape)
km = KMeans(2, do_plot=False)
pca = PCA(n_components=2)
pca.fit_transform(data)
print(data.shape)
# flow.fit(documents, w2p)
#
# w2p = Word2Phrase(max_phrase_len=4)
# w2p.fit(documents)
# print(w2p.phrases)
# documents = w2p.put_phrases_in_documents(documents)
#
# tokens = Preprocessor().fit(documents[0])
# model = gensim.models.Word2Vec(sentences, size=300, window=10, min_count=0, workers=4)
#
# w2v = Word2Vec(n_workers=8)
#
# w2v.fit(documents)
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
# # trie = Trie(tokens)
