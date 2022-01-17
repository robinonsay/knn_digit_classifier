from operator import itemgetter
from typing import TypeVar, List
from pprint import pformat

T = TypeVar("T")            # represents generic type
KDNode = TypeVar("KDNode")      # represents a Node object (forward-declare to use in Node __init__)


class KDNode:
    """
    Implementation of Node element of KD-Tree
    """
    __slots__ = ["data", "location", "parent", "left", "right"]

    def __init__(self, data: T,
                 left: KDNode = None, right: KDNode = None) -> None:
        """
        Constructs KD-Tree Node
        :param data: Data to store in KD-Tree Node
        :param left: Left-child
        :param right: Right-child
        """
        self.data = data
        self.left = left
        self.right = right

    def __repr__(self):
        return f"{self.data}"

    def __str__(self):
        return repr(self)


def kd_tree(points_list: List, depth: int = 0):
    """
    Thanks https://en.wikipedia.org/wiki/K-d_tree#Example_implementation
    :param points_list:
    :param depth:
    :return:
    """
    if points_list is None or len(points_list) == 0:
        return None
    dimension = len(points_list[0])
    axis = depth % dimension
    points_list.sort(key=itemgetter(axis))
    median = len(points_list) // 2
    return KDNode(data=points_list[median],
                  left=kd_tree(points_list[:median], depth + 1),
                  right=kd_tree(points_list[median + 1:], depth + 1))
