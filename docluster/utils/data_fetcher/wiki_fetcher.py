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
        suffixed_pages = list(map(lambda page: page + ' ' + self.suffix, self.pages))
        searched_pages = []
        for page in suffixed_pages:
            search_params = {
                'list': 'search',
                'srprop': '',
                'srlimit': 1,
                'srsearch': page
                }
            results = self.make_request(search_params)

            if 'error' in results:
                raise ValueError('The Wikipedia request coulnd\'t be compeleted.')
            else:
                searched_pages.extend([results['query']['search'][0]['title']])

        contents = []
        for title in searched_pages:
            query_params = {
            'prop': 'extracts|revisions',
            'explaintext': '',
            'rvprop': 'ids',
            'titles': title
            }

            results = self.make_request(query_params)

            if 'error' in results:
                raise ValueError('The Wikipedia request coulnd\'t be compeleted.')
            else:
                page_ids = list(results['query']['pages'].keys())
                text = ' '.join(map(lambda page_id: results['query']['pages'][page_id]['extract'].strip(), page_ids))
                contents.append(text)
        return contents



    def make_request(self, params):
        url = 'http://' + self.language.value.lower() + '.wikipedia.org/w/api.php'
        params['format'] = 'json'
        params['action'] = 'query'
        user_agent = 'Docluster/1.0 (metin@mit.edu)'
        headers = {}
        response = requests.get(url, params=params, headers=headers)
        return response.json()
