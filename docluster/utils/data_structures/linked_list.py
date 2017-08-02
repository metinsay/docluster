class LinkedList:
    def __init__(self, elt=None):
        self.next = None
        self.elt = elt

    def insert(self, item):
        """
        Adds the item into the LinkedList.
        Arguments:
            item (any): The item that is going to added to the LinkedList.
        """
        if not self.elt:
            self.elt = item
            return
        else:
            node = self
            while node.next:
                node = node.next
            node.next = LinkedList(item)
            return

    def length(self):
        """
        Returns the length of the LinkedList.
        """
        node = self
        if not node.elt:
            return 0
        count = 1
        while node.next:
            count += 1
            node = node.next
        return count

    def car(self):
        """
        Returns the value stored in the current LinkedList.
        """
        if self.isEmpty():
            raise EvaluationError
        return self.elt

    def cdr(self):
        """
        Returns next LinkedList.
        """
        return self.next

    def elementAt(self, i):
        """
        Returns the element at i.
        Arguments:
            i (int): the index of the element
        """
        if i >= self.length():
            raise EvaluationError
        node = self
        count = 0
        while node.next:
            count += 1
            if count - 1 == i:
                return node.elt
            node = node.next
        return node.elt

    def concat(self, other):
        result = LinkedList()
        for i in range(self.length()):
            result.insert(self.elementAt(i))
        for i in range(other.length()):
            result.insert(other.elementAt(i))
        return result

    @staticmethod
    def concatList(list_of_linked):
        """
        Concatinates the linked lists together and returns it.
        Arguments:
            list_of_linked (lsit): the list of LinkedLists that will be concatinated
        """
        result = LinkedList()
        for l in list_of_linked:
            result = result.concat(l)
        return result

    def isEmpty(self):
        """
        Returns true if the LinkedList is empty.
        """
        return not self.elt
