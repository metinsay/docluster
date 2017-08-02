

class Trie(object):

    def __init__(self, words=None):
        self.frequency = 0
        self.children = {}

        if words:
            for word in words:
                self.insert(word)

    def insert(self, word, frequency=None):
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
        trie = self.find(word)
        return trie.frequency > 0 if trie else False

    def __iter__(self):
        words = []

        def iterate(word, trie):
            if trie:
                if trie.frequency > 0:
                    words.append([word, trie.frequency])
                for char, child in trie.children.items():
                    iterate(word + char, child)
        iterate('', self)
        for word, freq in words:
            yield [word, freq]

    def autocomplete(self, prefix, n_suggestions=5):
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
