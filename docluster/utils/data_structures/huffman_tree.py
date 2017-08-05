from heapq import heapify, heappop, heappush

# Inspired by https://rosettacode.org/wiki/Huffman_coding#Python


class HuffmanTree(object):

    def __init__(self, frequencies):
        self.frequencies = frequencies
        self.encodings = self._create_tree()

    def __init__(self, iterable):
        self.frequencies = self._calculate_frequencies(iterable)
        self.encodings = self._create_tree()

    def _calculate_frequencies(self, iterable):
        freqs = {}
        for item in iterable:
            freqs[item] = freqs.get(item, 0) + 1
        return freqs

    def _create_tree(self):
        heap = [[freq, [item, '']] for item, freq in list(self.frequencies.items())]
        heapify(heap)
        while len(heap) > 1:
            min1, min2 = heappop(heap), heappop(heap)
            min1[1:] = [[item[0], '0' + item[1]] for item in min1[1:]]
            min2[1:] = [[item[0], '1' + item[1]] for item in min2[1:]]
            merged_item = [min1[0] + min2[0]] + min1[1:] + min2[1:]
            heappush(heap, merged_item)
        encodings_list = heap[0][1:]
        return {pair[0]: pair[1] for pair in encodings_list}

    def encode(self, iterable):
        return ''.join(map(lambda item: self.encodings[item], iterable))

    def decode(self, code):
        reverse_encodings = dict(zip(self.encodings.values(), self.encodings.keys()))
        cur = ''
        decoded_items = []
        for binary in code:
            cur += binary
            if cur in reverse_encodings:
                decoded_items.append(reverse_encodings[cur])
                cur = ''
        return decoded_items
