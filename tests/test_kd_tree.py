import random
import unittest
from queue import SimpleQueue

from kNN_digit_classifier.kd_tree import KDTree


class TestKDTreeInit(unittest.TestCase):
    def test_basic(self):
        point_list = [(7, 2), (5, 4), (9, 6), (4, 7), (8, 1), (2, 3)]
        expected = [(7, 2), (5, 4), (9, 6), (2, 3), (4, 7), (8, 1)]
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
        test_data = []
        for c_x in range(1, 9, 7):
            for c_y in range(1, 9, 7):
                for x in range(3):
                    for y in range(3):
                        test_data.append((x+c_x, y+c_y))
        expected = []
        kd_tree = KDTree(test_data)
        knn = kd_tree.kNN_search((2.5, 2.5))
        print(knn)


if __name__ == '__main__':
    unittest.main()
