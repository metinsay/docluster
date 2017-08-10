import os.path

import pandas as pd

from docluster.utils.constants import FileType


class FileFetcher(object):

    def __init__(self,  directory_path='~/Desktop/Docluster-data', encoding='utf-8'):
        self.directory_path = directory_path
        self.encoding = encoding

    def load(self, file_name, file_type=FileType.csv):
        full_path = os.path.join(self.directory_path, file_name + '.' + file_type.value)
        if file_type == FileType.csv:
            return pd.read_csv(full_path, encoding=self.encoding, index_col=0)
