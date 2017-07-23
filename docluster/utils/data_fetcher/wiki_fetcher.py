from .data_fetcher import DataFetcher
from ..constants.language import Language
import requests

class WikiFetcher(DataFetcher):
    def __init__(self, pages, suffix='', language=Language.english, do_open=False):

        self.pages = pages
        self.suffix = suffix
        self.language = language
        self.do_open = do_open

    def fetch(self):

        url = 'http://' + self.language.lower() + '.wikipedia.org/w/api.php'
        searched_pages = []
        for page in pages:
            search_params = {
                'list': 'search',
                'srprop': '',
                'srlimit': 1,
                'limit': 1,
                'srsearch': page
                }
            results = self.make_request(search_params)

            if 'error' in results:
                raise ValueError('The Wikipedia request coulnd\'t be compeleted.')
            else:
                searched_pages.extend(list(results))

        titles = '|'.join(searched_pages)
        query_params = {
            'prop': 'info|pageprops',
            'inprop': 'url',
            'ppprop': 'disambiguation',
            'redirects': '',
            'titles': titles
            }
        print(self.make_request(query_params))



    def make_request(self, params):
        params['format'] = 'json'
        params['action'] = 'query'
        user_agent = 'Docluster/1.0 (metin@mit.edu)'
        header = {'User-Agent':user_agent}
        response = requests.get(API_URL, params=params, headers=headers)
        return response.json()
