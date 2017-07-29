import pandas as pd
from ..constants.file_type import *
import os.path

class FileSaver(object):

    def __init__(self, directory_path='~/Desktop/Docluster-data', encoding='utf-8'):
        self.directory_path = directory_path
        self.encoding = encoding

    def save(self, data, file_name, file_type=FileType.csv, safe=True):

        full_path = os.path.join(self.directory_path, file_name + '.' + file_type.value)
        if not safe or os.path.isfile(full_path):
            return False

        if file_type == FileType.csv:
            data.to_csv(full_path, encoding=self.encoding)

        return True
