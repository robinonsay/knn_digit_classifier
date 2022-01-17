import unittest
from queue import SimpleQueue

from kNN_digit_classifier import kd_tree


class KDTreeBasic(unittest.TestCase):
    def test_matches_basic(self):
        point_list = [(7, 2), (5, 4), (9, 6), (4, 7), (8, 1), (2, 3)]
        origin = kd_tree.kd_tree(point_list)
        expected = [(7, 2), (5, 4), (9, 6), (2, 3), (4, 7), (8, 1)]
        queue = SimpleQueue()
        queue.put(origin)
        i = 0
        while not queue.empty():
            node = queue.get()
            self.assertEqual(expected[i], node.data)
            if node.left is not None:
                queue.put(node.left)
            if node.right is not None:
                queue.put(node.right)
            i += 1


if __name__ == '__main__':
    unittest.main()
