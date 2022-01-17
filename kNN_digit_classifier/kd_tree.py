import random
from queue import SimpleQueue

from math import ceil
from operator import itemgetter
from typing import TypeVar, List
from pprint import pformat

T = TypeVar("T")            # represents generic type
KDNode = TypeVar("KDNode")      # represents a Node object (forward-declare to use in Node __init__)


class KDNode:
    """
    Implementation of Node element of KD-Tree
    """
    __slots__ = ["value", "parent", "left", "right"]

    def __init__(self, value: T, parent: KDNode = None,
                 left: KDNode = None, right: KDNode = None) -> None:
        """
        Constructs KD-Tree Node
        :param value: Data to store in KD-Tree Node
        :param parent: Parent node in KD-Tree
        :param left: Left-child
        :param right: Right-child
        """
        self.value = value
        self.parent = parent
        self.left = left
        self.right = right

    def __repr__(self):
        return f"{self.value}"

    def __str__(self):
        return repr(self)


class KDTree:
    __slots__ = ["origin"]

    def __init__(self, data: List = None, origin: KDNode = None):
        self.origin = origin

        def kd_tree(data_list: List, parent: KDNode = origin, depth: int = 0):
            if data_list is None or len(data_list) == 0:
                return None
            dimension = len(data_list[0])
            axis = depth % dimension
            sample_length = 3 if len(data_list) > 3 else len(data_list)
            samples = random.choices(data_list, k=sample_length)
            samples.sort(key=itemgetter(axis))
            median = samples[len(samples) // 2]
            left_subtree = []
            right_subtree = []
            for point in data_list:
                if point == median:
                    continue
                if point[axis] <= median[axis]:
                    left_subtree.append(point)
                else:
                    right_subtree.append(point)
            node = KDNode(median, parent)
            node.left = kd_tree(left_subtree, node, depth + 1)
            node.right = kd_tree(right_subtree, node, depth + 1)
            return node

        if data is not None:
            self.origin = kd_tree(data)
