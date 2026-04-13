import sys
from dataclasses import dataclass


@dataclass
class Measurement:
    index: int
    l: float


class MaxHeap:

    def __init__(self, distances):
        self.maxsize = len(distances)
        self.size = self.maxsize
        self.Heap = [Measurement(index=-1, l=sys.maxsize)] \
                    + [Measurement(index=int(i) + 1, l=float(d)) for i, d in zip(range(self.maxsize), distances)]
        self.ids = [i for i in range(self.maxsize + 1)]  # the ith element's value represent the index of the original ith element in the current heap, 0 means deleted
        self.FRONT = 1
        self._build_max_heap()

    # Function to return the position of the parent for the node currently at pos
    def parent(self, pos):
        return pos // 2

    # Function to return the position of the left child for the node currently at pos
    def leftChild(self, pos):
        return 2 * pos

    # Function to return the position of the right child for the node currently at pos
    def rightChild(self, pos):
        return (2 * pos) + 1

    # Function that returns True if the passed node is a leaf node
    def isLeaf(self, pos):
        return pos > (self.size // 2) and pos <= self.size

    # Function to swap two nodes of the heap
    def swap(self, fpos, spos):
        i, j = self.Heap[fpos].index, self.Heap[spos].index
        self.Heap[fpos], self.Heap[spos], self.ids[i], self.ids[j] = self.Heap[spos], self.Heap[fpos], spos, fpos

    # Function to heapify the node at pos
    def _heapify_down(self, pos):
        # If the node is a non-leaf node and smaller than any of its children
        if not self.isLeaf(pos):
            left = self.leftChild(pos)
            right = self.rightChild(pos)
            largest = pos

            if left <= self.size and self.Heap[left].l > self.Heap[largest].l:
                largest = left
            if right <= self.size and self.Heap[right].l > self.Heap[largest].l:
                largest = right

            # Swap and continue heapifying if the current node is not the largest
            if largest != pos:
                self.swap(pos, largest)
                self._heapify_down(largest)

    # Function to print the contents of the heap
    def Print(self):
        for i in range(1, (self.size // 2) + 1):
            left = "LEFT CHILD : " + str(self.Heap[2 * i].l) if 2 * i <= self.size else "No LEFT CHILD"
            right = "RIGHT CHILD : " + str(self.Heap[2 * i + 1].l) if (2 * i + 1) <= self.size else "No RIGHT CHILD"
            print(f"PARENT : {self.Heap[i].l} {left} {right}")

    # Function to remove and return the maximum element from the heap
    def pop(self):
        if self.size == 0:
            raise None
        popped = self.Heap[self.FRONT]
        self.Heap[self.FRONT] = self.Heap[self.size]
        self.ids[popped.index] = -1  # Remove this term
        self.size -= 1
        self._heapify_down(self.FRONT)
        return popped

    def decrease_key(self, k, new_distance):
        if k == -1:
            return 0 # the node specified in index is removed

        if self.Heap[k].l > new_distance:
            self.Heap[k].l = new_distance
            self._heapify_down(k)
        return 1

    def _build_max_heap(self):
        for i in range(self.size // 2, 0, -1):
            self._heapify_down(i)



# Driver Code
if __name__ == "__main__":
    print('The maxHeap is ')

    dists = [5.0, 3.0, 17.0, 10.0, 84.0, 19.0, 6.0, 22.0, 9.0, 20, 84]
    maxHeap = MaxHeap(dists)
    # Print the heap structure
    maxHeap.Print()
    # Extract and print the max element
    max_element = maxHeap.pop()
    print(f"\nThe Max val is (index: {max_element.index}, l: {max_element.l})")
    print("\n")
    maxHeap.Print()
