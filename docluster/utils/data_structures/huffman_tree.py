from utils import Heap

# Inspired by https://rosettacode.org/wiki/Huffman_coding#Python


class HuffmanTree(object):

    def __init__(self, frequencies):
        self.frequencies = frequencies

    def __init__(self, iterable):
        self.frequencies = self._calculate_frequencies(iterable)

    def _calculate_frequencies(self, iterable):
        freqs = {}
        for element in iterable:
            freqs[element] = freqs.get(element, 0) + 1
        return freqs

    def _create_tree(self):
        items = [(freq, (item, '')) for item, freq in self.frequencies.items()]
        heap = Heap(items)
        while len(heap) > 1:
            min1, min2 = heap.pop(), heap.pop()
            min1[1:] = [[item[0], item[1] + '0'] for item in min1[1:]]
            min2[1:] = [[item[0], item[1] + '1'] for item in min2[1:]]
            merged_item = [min1[0] + min2[0]] + min1[1:] + min2[1:]
            heap.insert(merged_item)
        return heap
