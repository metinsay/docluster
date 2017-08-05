

class Trie(object):

    def __init__(self, words=None):
        """
            A recursive data type implementation of Trie tree.

            Paramaters:
            -----------
            words : list(str)
                Words that is going to be inserted into the trie.

            Attributes:
            -----------
            children : dict(char : Trie)
                The children of the current Trie.
        """

        self.frequency = 0
        self.children = {}

        if words:
            for word in words:
                self.insert(word)

    def insert(self, word, frequency=None):
        """
            Insert word into the trie.

            Paramaters:
            -----------
            word : str
                Word that is going to be inserted into the trie.
        """

        children = self.children
        for i, c in enumerate(word):
            if c in children:
                child = children[c]
            else:
                child = Trie()
                children[c] = child
            if i == len(word) - 1:
                child.frequency = frequency if frequency else child.frequency + 1
            else:
                children = child.children

    def find(self, prefix):
        """
            Insert word into the trie.

            Paramaters:
            -----------
            prefix : str
                Prefix that is going to be searched inside the trie.

            Return:
            -------
            found_trie : Trie
                The remaining trie after the prefix.
        """

        if prefix == '':
            return self
        children = self.children
        for i, c in enumerate(prefix):
            if c in children:
                child = children[c]
                if i == len(prefix) - 1:
                    return child
                children = child.children
            else:
                return None

    def __contains__(self, word):
        """
            If the trie contains the word.

            Paramaters:
            -----------
            word : str
                Word that is being searched inside the trie.

            Return:
            -------
            found : bool
                If the word exists inside the trie.
        """

        trie = self.find(word)
        return trie.frequency > 0 if trie else False

    def __iter__(self):
        """
            Iterator that yields (word, freq) pairs.

            Yield:
            ------
            pair : (str, int)
                Pair of word and frequency that belongs in trie.
        """

        words = []

        def iterate(word, trie):
        """Recursive helper function to iterate."""
            if trie:
                if trie.frequency > 0:
                    words.append([word, trie.frequency])
                for char, child in trie.children.items():
                    iterate(word + char, child)

        iterate('', self)
        for word, freq in words:
            yield [word, freq]

    def autocomplete(self, prefix, n_suggestions=5):
        """
            Autocomplete the prefix.

            Paramaters:
            -----------
            prefix : str
                Prefix that the autocompletion will be based on.
            n_suggestions : int
                Number of autocomplete suggestions to be returned.

            Return:
            -------
            suggestions : list(str)
                Autocomplete suggestions.
        """

        if not self.find(prefix):
            return []
        words = []
        for rest, freq in self.find(prefix):
            words.append([prefix + rest, freq])
        words = sorted(words, key=lambda x: -1 * x[1])
        return [word for word, freq in words[:n_suggestions]]

    def autocorrect(self, prefix, n_suggestions=5):
        chars = list('abcdefghijklmnopqrstuvwxyz')
        words = set([])
        autocomplete = self.autocomplete(prefix, n_suggestions)
        word = prefix
        for i, char in enumerate(word):
            for add in chars:
                # INSERT
                # Inserts a character from a-z
                inserted = word[:i] + add + word[i:]
                if inserted in self and not inserted in autocomplete:
                    words.add((inserted, self.find(inserted).frequency))

                # REPLACE
                # Replaces one character from word with one character from a-z
                replaced = list(word)
                replaced[i] = add
                replaced = ''.join(replaced)
                if replaced in self and not replaced in autocomplete:
                    words.add((replaced, self.find(replaced).frequency))

            # REMOVE
            # Removes one character from word
            removed = list(word)
            removed[i] = ''
            removed = ''.join(removed)
            if removed in self and not removed in autocomplete:
                words.add((removed, self.find(removed).frequency))

            # TRANSPOSE
            for j, other_char in enumerate(word):  # Switches two characters in word
                transposed = list(word)
                transposed[i], transposed[j] = transposed[j], transposed[i]
                transposed = ''.join(transposed)
                if transposed in self and not transposed in autocomplete:
                    words.add((transposed, self.find(transposed).frequency))

        words = sorted(list(words), key=lambda x: -1 * x[1])
        words = [word for word, freq in words[:n_suggestions]]
        return autocomplete + words[:n_suggestions - len(autocomplete)]
