import itertools
import math

import data_extractor as data_extractor
import text_miner as tm


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


def generate_candidates(prev_freq_itemsets, word_to_doc_map):
    candidates = set()
    for itemset_1, itemset_2 in itertools.combinations(list(prev_freq_itemsets), 2):
        union_itemset = ItemSet.unite(itemset_1, itemset_2)
        if len(union_itemset.items) == len(itemset_1.items) + 1:
            candidates.add(union_itemset)
    return candidates


def run_apriori_tid(documents, min_sup_perc, max_itemset_size, tokenizer=tm.tokenizer):

    word_to_doc_map = {}
    for doc_id, text in enumerate(documents):
        for token in tokenizer(text):
            token = token.lower()
            if token not in word_to_doc_map:
                word_to_doc_map[token] = set()
            word_to_doc_map[token].add(doc_id)

    min_sup_num = math.ceil(len(documents) * min_sup_perc)
    freq_itemsets = set()

    for word, doc_id_set in word_to_doc_map.items():
        if len(doc_id_set) >= min_sup_num:
            itemset = ItemSet(set([word]), set(doc_id_set))
            freq_itemsets.add(itemset)

    k = 2
    prev_freq_itemsets = set(freq_itemsets)
    while k <= max_itemset_size and len(prev_freq_itemsets) != 0:
        candiate_itemsets = generate_candidates(prev_freq_itemsets, word_to_doc_map)
        prev_freq_itemsets = set(filter(lambda itemset: len(
            itemset.ids) >= min_sup_num, candiate_itemsets))
        freq_itemsets |= prev_freq_itemsets
        k += 1

    return freq_itemsets
