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
