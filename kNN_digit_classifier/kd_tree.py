import heapq
import random
from collections import deque
from operator import itemgetter
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

    def kNN_search(self, value: ArrayLike, k: int = 5) -> List:
        dimension = len(value)
        best_nodes = deque(maxlen=k)

        def inner_kNN_search(current_node: KDNode = self.origin, depth: int = 0):
            axis = depth % dimension
            if current_node is None:
                return
            current_node_dist = distance.euclidean_distance(current_node.value, value)
            if current_node.left is None and current_node.right is None:
                if len(best_nodes) == 0:
                    best_nodes.append(current_node)
                else:
                    if current_node_dist < distance.euclidean_distance(best_nodes[0].value, value):
                        best_nodes.appendleft(current_node)
                    else:
                        best_nodes.append(current_node)
            else:
                is_le_node = value[axis] <= current_node.value[axis]
                if is_le_node:
                    inner_kNN_search(current_node.left, depth + 1)
                else:
                    inner_kNN_search(current_node.right, depth + 1)
                if len(best_nodes) == 0:
                    best_nodes.append(current_node)
                else:
                    if current_node_dist < distance.euclidean_distance(best_nodes[0].value, value):
                        best_nodes.appendleft(current_node)
                    else:
                        best_nodes.append(current_node)
                radius = distance.euclidean_distance(best_nodes[0].value, value)
                dist_to_plane = abs(current_node.value[axis] - best_nodes[0].value[axis])
                if dist_to_plane <= radius and current_node is not best_nodes[0]:
                    if is_le_node:
                        inner_kNN_search(current_node.right, depth + 1)
                    else:
                        inner_kNN_search(current_node.left, depth + 1)
        inner_kNN_search()
        return list(best_nodes)
