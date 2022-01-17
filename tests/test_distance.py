import unittest

import numpy

from kNN_digit_classifier import distance


class TestEuclideanDistance(unittest.TestCase):
    def test_345(self):
        a = [0, 3]
        b = [4, 0]
        expected = 5
        self.assertEqual(expected, distance.euclidean_distance(a, b))


if __name__ == '__main__':
    unittest.main()
