class ItemSet(object):
    def __init__(self, items, ids):
        self.items = frozenset(items)
        self.ids = frozenset(ids)

    @staticmethod
    def unite(itemset_1, itemset_2):
        items = itemset_1.items | itemset_2.items
        ids = itemset_1.ids & itemset_2.ids
        return ItemSet(items, ids)

    def __lt__(self, other):
        return len(other.items) > len(self.items)

    def __gt__(self, other):
        return len(other.items) < len(self.items)

    def __eq__(self, other):
        return type(other) == type(self) and other.items == self.items

    def __str__(self):
        return '{' + ','.join(list(self.items)) + '}'

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.items)
