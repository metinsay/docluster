import numpy as np
import pandas as pd
from cluster.bisecting_kmeans import BisectingKMeans
from cluster.kmeans import KMeans
from utils import DistanceMetric
from utils import PCA
from utils import WikiFetcher, TweetFetcher
from utils import TfIdf, Preprocessor
from utils import Language
from utils import TokenFilter
from classifier import Perceptron

# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf_vectorizer = TfidfVectorizer(max_df=0.6, min_df=0.0, use_idf=True, max_features=500,stop_words='english', ngram_range=(1,1), lowercase=True)
# vec = tfidf_vectorizer.fit_transform(docs)
#
# km = BisectingKMeans(k=8, dist_metric=DistanceMetric.eucledian, do_plot=True)
# km.fit(np.random.rand(1000,2))
# print(km.get_distances_btw_centroids())
# dists = km.get_distances_btw_centroids(dist_metric=DistanceMetric.manhattan, do_plot=True)
# print(dists[0,1], dists[0][1])

positive_tweets = TweetFetcher([':)',':D',':-)',':-D'], access_token='306764865-4t6Y4i849t3Ujd8RW059l2bF14vlr14FLgNCdN2E', access_secret='KLIijF3PM5KPLhGoFhRtQJ4OZB4cwmU8ZPezoECIfGENE', consumer_key='c3NheGxIZQ1lsCS3zzqR3Cz2p', consumer_secret='utNIoGk6srV1rKwIX0eNTVC4Me6cISM26YfPhQkHR7hsnYmlP0', language=Language.english).fetch()
negative_tweets = TweetFetcher([':(',':/',';(',':-('], access_token='306764865-4t6Y4i849t3Ujd8RW059l2bF14vlr14FLgNCdN2E', access_secret='KLIijF3PM5KPLhGoFhRtQJ4OZB4cwmU8ZPezoECIfGENE', consumer_key='c3NheGxIZQ1lsCS3zzqR3Cz2p', consumer_secret='utNIoGk6srV1rKwIX0eNTVC4Me6cISM26YfPhQkHR7hsnYmlP0', language=Language.english).fetch()
tweets = pd.Series(positive_tweets + negative_tweets, index=[1]*len(positive_tweets) + [-1]*len(negative_tweets))
additional_filters = [lambda token: len(token) == 1]
token_filter = TokenFilter(additional_filters=additional_filters, filter_contains=["#","@","http","/"])
preprocessor = Preprocessor(token_filter=token_filter)
tf_idf = TfIdf(min_df=0.0, max_df=0.9, preprocessor=preprocessor, do_plot=False)

tfidf_vector, labels = tf_idf.fit(tweets)

n = len(tfidf_vector)
print(preprocessor.fit(negative_tweets[-1]))
perceptron = Perceptron()
perceptron.train(tfidf_vector[:n-1],labels[:n-1])
print(perceptron.fit(np.array([tfidf_vector[n-1]])))

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
