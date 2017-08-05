from heapq import heapify, heappop, heappush

# Inspired by https://rosettacode.org/wiki/Huffman_coding#Python


class HuffmanTree(object):

    def __init__(self, frequencies=None, iterable=None):
        """
            A implementation of Huffman tree that provides encoding without loss of data.

            Paramaters:
            -----------
            frequencies : list((any, int))
                Frequencies of items given in tuples of (item, frequency).
            iterable : iterable
                Iterable which its items going to be used to create encodings.
                If frequencies is None, then this parameter is ignored.

            Attributes:
            -----------
            encodings : dict(any : str)
                The encoding of each item inside the iterable.
        """
        self.frequencies = frequencies if frequencies else self._calculate_frequencies(
            iterable)
        self.encodings = self._create_tree()

    def _calculate_frequencies(self, iterable):
        """Calculates the freqeuncies of each item inside tthe iterable."""
        freqs = {}
        for item in iterable:
            freqs[item] = freqs.get(item, 0) + 1
        return freqs

    def _create_tree(self):
        """Creates the tree and returns the encodings."""
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
        """
            Encode the data from the learnt encodings.

            Paramaters:
            -----------
            iterable : iterable
                Iterable that is going to be encoded.

            Return:
            -------
            encodings : str
                The encoded version of each item inside the iterable.
        """

        return list(map(lambda item: self.encodings[item], iterable))

    def decode(self, code):
        """
            Decode the encoded data from the learnt encodings.

            Paramaters:
            -----------
            code : str
                Encoded data that is going to be decoded.

            Return:
            -------
            decoded_items : list(any)
                Decoded items of the data.
        """
        reverse_encodings = dict(zip(self.encodings.values(), self.encodings.keys()))
        cur = ''
        decoded_items = []
        for binary in code:
            cur += binary
            if cur in reverse_encodings:
                decoded_items.append(reverse_encodings[cur])
                cur = ''
        return decoded_items
