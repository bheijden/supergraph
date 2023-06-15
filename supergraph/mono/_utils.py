import networkx as nx
from collections import deque
import queue
from enum import Enum


try:
    SimpleQueue = queue.SimpleQueue
except:
    SimpleQueue = queue.Queue


class QueuePolicy(Enum):
    """
    An Enum for queue pop policy.

    DEPTHFIRST policy is identified with pushing and popping on the same side
    of the queue. BREADTHFIRST is identified with pushing and popping from the
    opposite side of the queue.

    """

    DEPTHFIRST = 1
    BREADTHFIRST = 2


class Deque:
    """
    A double-ended queue implementation.

    """

    def __init__(self, policy: QueuePolicy = QueuePolicy.DEPTHFIRST):
        """
        Create a new double-ended queue.

        Arguments:
            policy (QueuePolicy): The policy to use when adding and removing
                elements from this queue. Defaults to depth-first.

        Returns:
            None

        """
        self._dq = deque()
        if policy == QueuePolicy.DEPTHFIRST:
            self._put = self._dq.append
            self._get = self._dq.popleft
        elif policy == QueuePolicy.BREADTHFIRST:
            self._get = self._dq.pop
            self._put = self._dq.append

    def put(self, *args, **kwargs):
        """
        Put a new element into the queue.

        Arguments:
            element: The element to add to the queue

        Returns:
            None

        """
        return self._put(args[0])

    def get(self, *args, **kwargs):
        """
        Get a new item from the queue.

        Arguments:
            None

        Returns:
            Any: The element popped from the queue

        """
        return self._get()

    def empty(self):
        """
        Returns True if the queue is empty.

        Arguments:
            None

        Returns:
            bool: True if the queue has nothing more to pop

        """
        return False if self._dq else True


def uniform_node_interestingness(motif: nx.DiGraph) -> dict:
    """
	Sort the nodes in a motif by their interestingness.

	Most interesting nodes are defined to be those that most rapidly filter the
	list of nodes down to a smaller set.

	"""
    return {n: 1 for n in motif.nodes}