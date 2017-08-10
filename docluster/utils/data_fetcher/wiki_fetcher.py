import requests

from docluster.utils.constants import Language


class WikiFetcher(object):
    def __init__(self, titles, suffix='', filterNone=False, language=Language.english, do_browse=False):
        """ Initialize WikiFetcher
        titles - a list of page titles that is going to be fetched, where
                these titles don't need to be perfect since a search
                is performed prior to the fetch of the contents
        suffix - string that is going to be appended with a space at
                 the end of each page title in order to increase the chances
                 of choicing the correct wikipedia page (usually suffix is
                 the common theme of the pages)
        filterNone - filter all the failing queries
        language - language of the content that is intended
        do_browse - show the results in an html page
        """

        self.titles = titles
        self.suffix = suffix
        self.language = language
        self.filterNone = filterNone
        self.do_browse = do_browse

    def fetch(self):
        suffixed_titles = list(map(lambda title: title + ' ' + self.suffix, self.titles))
        searched_titles = []
        for title in suffixed_titles:
            search_params = {
                'list': 'search',
                'srprop': '',
                'srlimit': 1,
                'srsearch': title
            }
            results = self.make_request(search_params)

            if 'error' in results:
                raise ValueError('The Wikipedia request coulnd\'t be compeleted.')
            else:
                searched_titles.extend([results['query']['search'][0]['title']])

        contents = []
        for title in searched_titles:
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
                text = ' '.join(
                    map(lambda page_id: results['query']['pages'][page_id]['extract'].strip(), page_ids))
                contents.append(text)
        return contents

    def make_request(self, params):
        url = 'http://' + self.language.value.lower() + '.wikipedia.org/w/api.php'
        params['format'] = 'json'
        params['action'] = 'query'
        response = requests.get(url, params=params)
        return response.json()
