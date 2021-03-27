import time
from sortedcontainers import SortedKeyList
from .rush_hour_puzzle import RushHourPuzzle
from typing import Sequence, Union, List, Callable, Any, Deque, Set
from collections import deque
import copy


##########################################
class SearchVersion:
    TREE = 'tree'
    GRAPH = 'graph'


class Algorithm:
    DFS = 'DFS'
    BFS = 'BFS'
    IDS = 'IDS'
    A_STAR = 'A*'
    IDA_STAR = 'IDA*'


class SearchStats:
    def __init__(self,
                 algorithm: Algorithm,
                 version: SearchVersion,
                 path: Sequence[RushHourPuzzle],
                 explored_list: Sequence[RushHourPuzzle],
                 node_num_record: Sequence[int],
                 n_expand: int,
                 time_cost: float):
        # Length of explored_set and node_num_record must be the same.
        # Each element in node_num_record is the number of nodes in the container
        #   before the index-corresponding state to be put in the explored_set.
        if len(explored_list) != len(node_num_record):
            raise ValueError('Length of path and node_num_record must be the same.')
        self.algorithm: Algorithm = algorithm
        self.version: SearchVersion = version
        self.path: List[RushHourPuzzle] = list(path)  # If there's no element, it means it doesn't find a solution.
        self.explored: List[RushHourPuzzle] = list(explored_list)
        self.node_num_record: List[int] = list(node_num_record)
        self.n_expand: int = n_expand
        self.time_cost: float = time_cost


class Node:
    def __init__(self,
                 rush_hour_puzzle: RushHourPuzzle,
                 node_depth: int,
                 so_far_n_conn: int,
                 g_cost: float,
                 h_cost: float):
        self.depth: int = node_depth
        self.rush_hour_puzzle = rush_hour_puzzle
        self.parent: Union['Node', Any] = None
        self.so_far_n_conn: int = so_far_n_conn
        self.g_cost: float = g_cost
        self.h_cost: float = h_cost

    @staticmethod
    def compare_key(node: 'Node'):
        return node.g_cost + node.h_cost


def search(algorithm: Algorithm,
           version: SearchVersion,
           source_board: RushHourPuzzle,
           heuristic_fun: Union[Callable[[RushHourPuzzle], float], Any] = None,
           max_depth: int = 99999,
           max_n_explored: int = 1000) -> 'SearchStats':
    #
    if algorithm not in [Algorithm.DFS, Algorithm.BFS, Algorithm.IDS, Algorithm.A_STAR, Algorithm.IDA_STAR]:
        raise ValueError('Algorithm "{}" is invalid.'.format(algorithm))

    #
    start_time = time.time()

    # Need to be a renewed
    cur_max_depth: int = 0
    last_node = None
    explored_encoded_set: Set = set()
    explored_list: List = []  # These two about explored nodes should be maintained simultaneously
    sol_found: bool = False
    node_num_record: List = []
    n_expand: int = 1

    # Each SortedKeyList contains the same group of nodes.
    # While choosing next node, we pick from the group with the smallest idx in container list.
    container: Deque[SortedKeyList] = deque()

    # Put into the source node
    container.append(SortedKeyList([Node(
        rush_hour_puzzle=source_board,
        node_depth=1,
        so_far_n_conn=0,
        g_cost=0,
        h_cost=heuristic_fun(source_board) if heuristic_fun else 0)], key=Node.compare_key))

    #
    while len(container) != 0:
        # Record current number of nodes in the container
        node_num_record.append(sum(len(x) for x in container))

        # Pick the next node.
        node: Node = container[0].pop(0)
        last_node = node
        if len(container[0]) == 0:
            container.pop(0)
        cur_max_depth = max(cur_max_depth, node.depth)

        #
        # TODO: Write down why not hash?
        explored_encoded_set.add(node.rush_hour_puzzle.encode())
        explored_list.append(node.rush_hour_puzzle)

        # Check if terminal

        # Check if find the solution
        if node.rush_hour_puzzle.is_solved():
            sol_found = True
            break

        # Check current total number of visited nodes and current max depth
        if len(explored_list) >= max_n_explored or cur_max_depth >= max_depth:
            break

        # Continue to dig out a deeper layer
        sorted_key_list = SortedKeyList([], key=Node.compare_key)
        for action in node.rush_hour_puzzle.actions:
            new_puzzle = copy.deepcopy(node.rush_hour_puzzle)
            new_puzzle.apply_action(action=action)

            # If it's graph search, we don't visit a repeated state.
            if version == SearchVersion.GRAPH:
                if RushHourPuzzle.board_encoded_exists(explored_encoded_set, new_puzzle.encode()):
                    continue

            # Make a new node
            new_node = Node(rush_hour_puzzle=new_puzzle,
                            node_depth=node.depth + 1,
                            so_far_n_conn=node.so_far_n_conn + 1,
                            g_cost=node.g_cost + 1,
                            h_cost=heuristic_fun(new_puzzle) if heuristic_fun else 0)
            new_node.parent = node
            sorted_key_list.add(new_node)

            # Renew number of nodes expanded
            n_expand += 1

        # Put int the container, depending on the algorithm chose
        if algorithm == Algorithm.DFS:
            container.append(sorted_key_list)
        else:
            container += sorted_key_list

    #
    time_cost = time.time() - start_time  # Unit: sec

    # Backtrace the path if there's a solution found
    path: List[RushHourPuzzle] = []
    if sol_found:
        # Trace back the path
        node: Node = last_node
        path.append(node.rush_hour_puzzle)
        while node.parent:
            path.append(node.parent.rush_hour_puzzle)
            node = node.parent
        path.reverse()

    # Gather data
    return SearchStats(algorithm=algorithm,
                       version=version,
                       path=path,
                       explored_list=explored_list,
                       node_num_record=node_num_record,
                       n_expand=n_expand,
                       time_cost=time_cost)
