import random
import unittest
from queue import SimpleQueue

from kNN_digit_classifier.kd_tree import KDTree


class TestKDTreeInit(unittest.TestCase):
    def test_basic(self):
        random.seed(3165)
        point_list = [(7, 2), (5, 4), (9, 6), (4, 7), (8, 1), (2, 3)]
        expected = [(7, 2), (4, 7), (9, 6), (5, 4), (8, 1), (2, 3)]
        kd_tree = KDTree(point_list)
        origin = kd_tree.origin
        queue = SimpleQueue()
        queue.put(origin)
        i = 0
        while not queue.empty():
            node = queue.get()
            self.assertEqual(expected[i], node.value)
            if node.left is not None:
                queue.put(node.left)
            if node.right is not None:
                queue.put(node.right)
            i += 1


class TestKDTreeKNNSearch(unittest.TestCase):
    def test_basic(self):
        random.seed(3165)
        point_list = [(7, 2), (5, 4), (9, 6), (4, 7), (8, 1), (2, 3), (1, 2), (6, 3), (2, 8), (4, 2)]
        expected = []
        kd_tree = KDTree(point_list)
        test_list = kd_tree.kNN_search((3, 1))
        print(test_list)



if __name__ == '__main__':
    unittest.main()
