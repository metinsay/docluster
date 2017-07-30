

class Document(object):

    def __init__(self, text, label=None):
        self.text = text
        self.label = label

    def __str__(self):
        return self.label

    def __len__(self):
        return len(self.text)
