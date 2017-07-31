from os import listdir
from os.path import isfile, join

import pandas as pd

import numpy as np
from bs4 import BeautifulSoup
from classifier import Perceptron
from cluster.bisecting_kmeans import BisectingKMeans
from cluster.kmeans import KMeans
from nltk.corpus import movie_reviews
from utils import (PCA, DistanceMetric, FileFetcher, FileSaver, FileType,
                   Language, Preprocessor, TfIdf, TokenFilter, TweetFetcher,
                   WikiFetcher, Word2Vec)

# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf_vectorizer = TfidfVectorizer(max_df=0.6, min_df=0.0, use_idf=True, max_features=500,stop_words='english', ngram_range=(1,1), lowercase=True)
# vec = tfidf_vectorizer.fit_transform(docs)
#
# km = BisectingKMeans(k=8, dist_metric=DistanceMetric.eucledian, do_plot=True)
# km.fit(np.random.rand(1000,2))
# print(km.get_distances_btw_centroids())
# dists = km.get_distances_btw_centroids(dist_metric=DistanceMetric.manhattan, do_plot=True)
# print(dists[0,1], dists[0][1])
#
# positive_tweets = TweetFetcher([':)',':D',':-)',':-D'], access_token='306764865-4t6Y4i849t3Ujd8RW059l2bF14vlr14FLgNCdN2E', access_secret='KLIijF3PM5KPLhGoFhRtQJ4OZB4cwmU8ZPezoECIfGENE', consumer_key='c3NheGxIZQ1lsCS3zzqR3Cz2p', consumer_secret='utNIoGk6srV1rKwIX0eNTVC4Me6cISM26YfPhQkHR7hsnYmlP0', language=Language.english).fetch()
# negative_tweets = TweetFetcher([':(',':/',';(',':-('], access_token='306764865-4t6Y4i849t3Ujd8RW059l2bF14vlr14FLgNCdN2E', access_secret='KLIijF3PM5KPLhGoFhRtQJ4OZB4cwmU8ZPezoECIfGENE', consumer_key='c3NheGxIZQ1lsCS3zzqR3Cz2p', consumer_secret='utNIoGk6srV1rKwIX0eNTVC4Me6cISM26YfPhQkHR7hsnYmlP0', language=Language.english).fetch()
# tweets = pd.Series(positive_tweets + negative_tweets, index=[1]*len(positive_tweets) + [-1]*len(negative_tweets))
# print(positive_tweets + negative_tweets)
# #
# additional_filters = [lambda token: len(token) == 1]
# token_filter = TokenFilter(additional_filters=additional_filters, filter_contains=["#","@","http","/"])
# preprocessor = Preprocessor(token_filter=token_filter)
# tf_idf = TfIdf(min_df=0.0, max_df=0.9, preprocessor=preprocessor, do_plot=False)

# tfidf_vector = tf_idf.fit(tweets)
# tf_idf.save_model('model_tfidf')
# tf_idf = TfIdf()
# tf_idf.load_model('model_tfidf')
# print(tf_idf.tfidf_vector)

# import gensim
#
# # Load Google's pre-trained Word2Vec model.
# model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
# print(model.wv.similar('fruit'))

#
# n = len(tfidf_vector)
# print(preprocessor.fit(negative_tweets[-1]))
# perceptron = Perceptron()
# perceptron.train(tfidf_vector[:n-1],labels[:n-1])
# print(perceptron.fit(np.array([tfidf_vector[n-1]])))

# wikis = WikiFetcher(['ios','android','windows', 'corgi', 'puppy', 'dog']).fetch()

#
# km = KMeans(k=2, dist_metric=DistanceMetric.cosine, do_plot=False)
# print(km.fit(tf_idf.fit(wikis))[1])
# print(tf_idf.get_token_httpvectors(do_plot=False))


# for wiki in wikis:
#     print(len(wiki))
#
# import gensim
# from multiprocessing import cpu_count
# from bs4 import BeautifulSoup
# import numpy as np
# import nltk
# import string
#
#
# def tokenize(text):
#     html = BeautifulSoup(text, "html5lib").get_text()
#     results = [words for words in [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(html)]]
#
#
#     results = [[word.lower().replace('-','').replace('/','') for word in sentence] for sentence in results]
#     return [list(filter(lambda token: token not in string.punctuation and not token.isdigit(), sentence)) for sentence in results]
#
#
#
# clustered_data = wikis
# flattened_data = '.\n'.join(clustered_data)
# clustered_data = tokenize(flattened_data)
# # print(clustered_data)
#
# model = gensim.models.Word2Vec(clustered_data, size=10000, window=10, min_count=5, workers=4)
#
# print(model.wv.similarity('ios', 'web'))


# data = FileFetcher('/Users/metinsay/Downloads/trwiki-20160305/').load('articles',FileType.csv)
# name_filter = data[data.title == 'Elma'].text
# apple = name_filter.tolist()[0]
# print(apple)


# print(len(data))
# name_filter = data.text.where(lambda name: name.str.contains('') ).dropna()
# filtered_data =  data[data.text.isin(name_filter)].text.tolist()
# print(len(filtered_data))
# apples = list(map(lambda text: text.replace('[','').replace(']',''), apples))


# additional_filters = [lambda token: len(token) == 1]
# token_filter = TokenFilter(additional_filters=additional_filters, filter_contains=["#","@","http","/"])
# preprocessor = Preprocessor(token_filter=token_filter)
# tf_idf = TfIdf(min_df=0.0, max_df=0.7, preprocessor=preprocessor, do_plot=False)
#
# tfidf_vector = tf_idf.fit(apples)
# print(tf_idf.vocab)
# tf_idf.save_model('model_tfidf_apples')

# import xml.etree.ElementTree as etree
# import codecs
# import csv
# import time
# import os
#
# # http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
#
# PATH_WIKI_XML = '/Users/metinsay/Downloads/trwiki-20160305/'
# FILENAME_WIKI = 'wiki.xml'
# FILENAME_ARTICLES = 'articles.csv'
# ENCODING = "utf-8"
#
#
# # Nicely formatted time string
# def hms_string(sec_elapsed):
#     h = int(sec_elapsed / (60 * 60))
#     m = int((sec_elapsed % (60 * 60)) / 60)
#     s = sec_elapsed % 60
#     return "{}:{:>02}:{:>05.2f}".format(h, m, s)
#
#
# def strip_tag_name(t):
#     t = elem.tag
#     idx = k = t.rfind("}")
#     if idx != -1:
#         t = t[idx + 1:]
#     return t
#
#
# pathWikiXML = os.path.join(PATH_WIKI_XML, FILENAME_WIKI)
# pathArticles = os.path.join(PATH_WIKI_XML, FILENAME_ARTICLES)
#
# totalCount = 0
# articleCount = 0
# title = None
# start_time = time.time()
#
# with codecs.open(pathArticles, "w", ENCODING) as articlesFH:
#     articlesWriter = csv.writer(articlesFH, quoting=csv.QUOTE_MINIMAL)
#
#     articlesWriter.writerow(['id', 'title', 'text'])
#
#     for event, elem in etree.iterparse(pathWikiXML, events=('start', 'end')):
#         tname = strip_tag_name(elem.tag)
#
#         if event == 'start':
#             if tname == 'page':
#                 title = ''
#                 id = -1
#                 redirect = ''
#                 inrevision = False
#                 ns = 0
#             elif tname == 'revision':
#                 # Do not pick up on revision id's
#                 inrevision = True
#         else:
#             if tname == 'title':
#                 title = elem.text
#             elif tname == 'id' and not inrevision:
#                 _id = int(elem.text)
#             elif tname == 'text':
#                 text = elem.text
#             elif tname == 'page':
#                 totalCount += 1
#
#                 articlesWriter.writerow([_id, title, text])
#
#                 if totalCount > 1 and (totalCount % 100000) == 0:
#                     print("{:,}".format(totalCount))
#
#             elem.clear()
#
# elapsed_time = time.time() - start_time
#
# print("Total pages: {:,}".format(totalCount))
# print("Template pages: {:,}".format(templateCount))
# print("Article pages: {:,}".format(articleCount))
# print("Redirect pages: {:,}".format(redirectCount))
# print("Elapsed time: {}".format(hms_string(elapsed_time)))


wiki_directories = ['/Users/metinsay/Downloads/wikiextractor-master/text/AA']
onlyfiles = []
for directory in wiki_directories:
    onlyfiles.extend([join(directory, f)
                      for f in listdir(directory) if isfile(join(directory, f))])


documents = []
for file_ in onlyfiles[:1]:
    documents.append(BeautifulSoup(open(file_, 'r').read(), "lxml").get_text())

w2v = Word2Vec(n_workers=8)

w2v.fit(documents)
