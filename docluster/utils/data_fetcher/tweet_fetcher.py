import tweepy

from ..constants.language import Language
from .data_fetcher import DataFetcher


class TweetFetcher(DataFetcher):
    def __init__(self, queries, access_token, access_secret, consumer_key, consumer_secret, n_limit_per_query=100, n_limit_total=1000, suffix='', filterNone=False, language=Language.english, do_browse=False):
        self.queries = queries
        self.n_limit_per_query = n_limit_per_query
        self.n_limit_total = n_limit_total
        self.suffix = suffix
        self.filterNone = filterNone
        self.language = language
        self.do_browse = do_browse

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_secret)
        self.twitter = tweepy.API(auth)

    def fetch(self):
        tweets = []
        for query in self.queries:
            results = self.twitter.search(q=query, lang=self.language.value, count=5000)
            tweets.extend([tweet.text for tweet in results])
        return tweets
