import numpy as np
import pandas as pd
from cluster.k_means import KMeans
from utils import DistanceMetric
from utils import PCA

# docs = ["One two threee","Fours ficda asgag","I go to bed","Do you go to bed","I will maybe go to bed", "Meh maybe"]
# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf_vectorizer = TfidfVectorizer(max_df=0.6, min_df=0.0, use_idf=True, max_features=500,stop_words='english', ngram_range=(1,1), lowercase=True)
# vec = tfidf_vectorizer.fit_transform(docs)

km = KMeans(k=5, do_plot=True, dist_metric=DistanceMetric.chebyshev)
print(km.fit(np.random.rand(1000,4)))
