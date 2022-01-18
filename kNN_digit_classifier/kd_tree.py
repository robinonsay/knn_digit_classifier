import heapq
import random
from collections import deque
from operator import itemgetter
from queue import LifoQueue
from typing import TypeVar, List
from numpy.typing import ArrayLike
from kNN_digit_classifier import distance

T = TypeVar("T")            # represents generic type
KDNode = TypeVar("KDNode")      # represents a Node object (forward-declare to use in Node __init__)


class KDNode:
    """
    Implementation of Node element of KD-Tree
    """
    __slots__ = ["value", "parent", "left", "right"]

    def __init__(self, value: ArrayLike, parent: KDNode = None,
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


class KDNodeHeapWrapper:
    __slots__ = ["node", "ref_val"]

    def __init__(self, node: KDNode, ref_val: int):
        self.node = node
        self.ref_val = ref_val

    def __lt__(self, other: KDNode):
        node_dist = distance.euclidean_distance(self.node.value, self.ref_val)
        other_dist = distance.euclidean_distance(other.node.value, self.ref_val)
        return node_dist < other_dist


class KDTree:
    __slots__ = ["origin"]

    def __init__(self, data: List[ArrayLike]):
        """

        :param data:
        """
        def kd_tree(data_list: List[ArrayLike], parent: KDNode = None, depth: int = 0):
            """

            :param data_list:
            :param parent:
            :param depth:
            :return:
            """
            if data_list is None or len(data_list) == 0:
                return None
            dimension = len(data_list[0])
            axis = depth % dimension
            data_list.sort(key=itemgetter(axis))
            median = len(data_list) // 2
            node = KDNode(data_list[median], parent)
            node.left = kd_tree(data_list[:median], node, depth + 1)
            node.right = kd_tree(data_list[median + 1:], node, depth + 1)
            return node
        self.origin = kd_tree(data)

    def kNN_search(self, search_point: ArrayLike, k: int = 5) -> List:
        pass
