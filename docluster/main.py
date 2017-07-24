import numpy as np
import pandas as pd
from cluster.bisecting_kmeans import BisectingKMeans
from utils import DistanceMetric
from utils import PCA
from utils import WikiFetcher

# docs = ["One two threee","Fours ficda asgag","I go to bed","Do you go to bed","I will maybe go to bed", "Meh maybe"]
# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf_vectorizer = TfidfVectorizer(max_df=0.6, min_df=0.0, use_idf=True, max_features=500,stop_words='english', ngram_range=(1,1), lowercase=True)
# vec = tfidf_vectorizer.fit_transform(docs)

# km = BisectingKMeans(k=8, dist_metric=DistanceMetric.eucledian, do_plot=True)
# km.fit(np.random.rand(1000,2))
# print(km.get_distances_btw_centroids())
# print(km.get_distances_btw_centroids(dist_metric=DistanceMetric.manhattan, do_plot=True))
wikis = WikiFetcher(['asp.net','html','ios','objective-c','asp','mvc','css','javascript','web developement'], suffix='').fetch()
for wiki in wikis:
    print(len(wiki))

import gensim
from multiprocessing import cpu_count
from bs4 import BeautifulSoup
import numpy as np
import nltk
import string


def tokenize(text):
    html = BeautifulSoup(text, "html5lib").get_text()
    results = [words for words in [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(html)]]


    results = [[word.lower().replace('-','').replace('/','') for word in sentence] for sentence in results]
    return [list(filter(lambda token: token not in string.punctuation and not token.isdigit(), sentence)) for sentence in results]



clustered_data = wikis
flattened_data = '.\n'.join(clustered_data)
clustered_data = tokenize(flattened_data)
# print(clustered_data)

model = gensim.models.Word2Vec(clustered_data, size=10000, window=10, min_count=5, workers=4)

print(model.wv.similarity('ios', 'web'))
