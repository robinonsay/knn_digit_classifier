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
    __slots__ = ["value", "parent", "left", "right", "axis"]

    def __init__(self, value: ArrayLike, axis: int, parent: KDNode = None,
                 left: KDNode = None, right: KDNode = None) -> None:
        """
        Constructs KD-Tree Node
        :param value: Data to store in KD-Tree Node
        :param parent: Parent node in KD-Tree
        :param left: Left-child
        :param right: Right-child
        """
        self.value = value
        self.axis = axis
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
            node = KDNode(data_list[median], axis, parent)
            node.left = kd_tree(data_list[:median], node, depth + 1)
            node.right = kd_tree(data_list[median + 1:], node, depth + 1)
            return node
        self.origin = kd_tree(data)

    def kNN_search(self, search_point: ArrayLike, k: int = 5) -> List:
        best_nodes: deque[KDNode] = deque(maxlen=k)
        distances = {}

        def kNN_search(current_node: KDNode):
            if current_node is None:
                return
            axis = current_node.axis
            if current_node not in distances:
                distances[current_node] = distance.squared_euclidean(current_node.value, search_point)
            if current_node.left is None and current_node.right is None:
                if len(best_nodes) == 0:
                    best_nodes.appendleft(current_node)
                elif distances[current_node] <= distance.squared_euclidean(best_nodes[0].value, search_point):
                    best_nodes.appendleft(current_node)
            else:
                if search_point[axis] <= current_node.value[axis]:
                    kNN_search(current_node.left)
                else:
                    kNN_search(current_node.right)
                if best_nodes[0] not in distances:
                    distances[best_nodes[0]] = distance.squared_euclidean(best_nodes[0].value, search_point)
                if distances[current_node] <= distances[best_nodes[0]]:
                    best_nodes.appendleft(current_node)
                    distances[best_nodes[0]] = distance.squared_euclidean(best_nodes[0].value, search_point)
                radius = distances[best_nodes[0]]
                dist_plane_to_sp = abs(search_point[axis] - current_node.value[axis])
                if dist_plane_to_sp < radius or len(best_nodes) < best_nodes.maxlen:
                    if search_point[axis] <= current_node.value[axis]:
                        kNN_search(current_node.right)
                    else:
                        kNN_search(current_node.left)

        kNN_search(self.origin)
        best_values = []
        for node in best_nodes:
            best_values.append(node.value)
        return best_values
