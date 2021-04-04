import time
from sortedcontainers import SortedKeyList, SortedSet
from .rush_hour_puzzle import RushHourPuzzle
from typing import Sequence, Union, List, Callable, Any, Deque, Set, Tuple, Optional, NewType, Dict, TypedDict
from collections import deque
import copy

##########################################
SearchVersionT = NewType('SearchVersionT', str)
AlgorithmT = NewType('AlgorithmT', str)
SearchWhatFirstT = NewType('SearchWhatFirstT', str)


class SearchVersion:
    TREE: SearchVersionT = 'tree'
    GRAPH: SearchVersionT = 'graph'


class Algorithm:
    DFS: AlgorithmT = 'DFS'
    BFS: AlgorithmT = 'BFS'
    IDS: AlgorithmT = 'IDS'
    A_STAR: AlgorithmT = 'A_STAR'
    IDA_STAR: AlgorithmT = 'IDA_STAR'


class SearchWhatFirst:
    DEPTH_FIRST: SearchWhatFirstT = 'depth-first'
    BREADTH_FIRST: SearchWhatFirstT = 'breadth-first'


#################################

class SearchStats:
    def __init__(self,
                 algorithm: AlgorithmT,
                 version: SearchVersionT,
                 sol_found: bool,
                 path: Sequence[RushHourPuzzle],
                 explored_list: Sequence[RushHourPuzzle],
                 node_num_record: Sequence[int],
                 n_expand: int,
                 time_cost: float):
        # Length of explored_set and node_num_record must be the same.
        # Each element in node_num_record is the number of nodes in the container
        #   before the index-corresponding state to be put in the explored_set.
        if len(explored_list) != len(node_num_record):
            raise ValueError('Length of path and node_num_record must be the same when not using iterative deepening.')
        self.algorithm: AlgorithmT = algorithm
        self.version: SearchVersionT = version
        self.sol_found: bool = sol_found
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


class SearchContainer:
    def init_container(self) -> Union[Deque[Node], SortedKeyList]:
        if self._algorithm in [Algorithm.DFS, Algorithm.BFS, Algorithm.IDS]:
            return deque()
        elif self._algorithm in [Algorithm.A_STAR, Algorithm.IDA_STAR]:
            return SortedKeyList(key=Node.compare_key)
        else:
            raise ValueError('Not accepted algorithm:', self._algorithm)

    def __init__(self, algorithm: AlgorithmT):
        self._algorithm: AlgorithmT = algorithm
        self._container: Union[Deque[Node], SortedKeyList[Node]] = self.init_container()

    def add(self, node: Node):
        if isinstance(self._container, SortedKeyList):
            self._container.add(node)
        elif isinstance(self._container, Deque):
            self._container.append(node)
        else:
            raise TypeError('Invalid type of "self._container".')

    def pop_front(self) -> Node:
        if isinstance(self._container, SortedKeyList):
            return self._container.pop(0)
        elif isinstance(self._container, Deque):
            return self._container.popleft()
        else:
            raise TypeError('Invalid type of "self._container".')

    def pop_back(self) -> Node:
        if isinstance(self._container, (SortedKeyList, Deque)):
            return self._container.pop()
        else:
            raise TypeError('Invalid type of "self._container".')

    def __len__(self) -> int:
        return len(self._container)


class SearchEngine:
    def make_init_node(self) -> Node:
        return Node(
            rush_hour_puzzle=self._source_puzzle,
            node_depth=0,
            so_far_n_conn=0,
            g_cost=0,
            h_cost=self._heuristic_fun(self._source_puzzle) if self._heuristic_fun else 0)

    def renew_explored(self, puzzle: RushHourPuzzle):
        self._explored_encoded_set.add(puzzle.encode())
        self._explored_list.append(puzzle)

    def renew_part_time_record(self, part_id: int):
        if self._show_part_info:
            time_consumed = time.time() - self._time_point_start
            self._part_time[part_id] = self._part_time.get(part_id, 0.) + time_consumed
            self._part_count[part_id] = self._part_count.get(part_id, 0) + 1
        else:
            return

    def show_part_time_stat(self):
        if self._show_part_info:
            print('Part avg time (loop={}):'.format(self._loop_cnt))
            for part_id in self._part_time:
                part_time = self._part_time[part_id]
                part_count = self._part_count[part_id]
                if part_id == 0:
                    print('Error: part_count of part_id "{}" is 0. (Invalid)'.format(part_id))
                    break
                else:
                    avg_time = part_time / part_count
                    print('{}: {:.8f}'.format(part_id, avg_time))
        else:
            return

    def record_time_point_start(self):
        if self._show_part_info:
            self._time_point_start = time.time()
        else:
            return

    def __init__(self,
                 config: dict,
                 algorithm: AlgorithmT,
                 version: SearchVersionT,
                 source_puzzle: RushHourPuzzle,
                 heuristic_fun: Optional[Callable[[RushHourPuzzle], float]] = None,
                 max_depth: int = 99999,
                 max_n_explored: int = 100000):
        # Check params
        if algorithm not in [Algorithm.DFS, Algorithm.BFS, Algorithm.IDS, Algorithm.A_STAR, Algorithm.IDA_STAR]:
            raise ValueError('Invalid algorithm:', algorithm)
        if version not in [SearchVersion.TREE, SearchVersion.GRAPH]:
            raise ValueError('Invalid search version:', version)
        if not isinstance(source_puzzle, RushHourPuzzle):
            raise TypeError('Type of source_puzzle must be "RushHourPuzzle", but got {}'.format(type(source_puzzle)))
        if heuristic_fun and not callable(heuristic_fun):
            raise ValueError('Valid heuristic_fun is "Optional[Callable[[RushHourPuzzle], float]]".')
        if heuristic_fun and algorithm in [Algorithm.DFS, Algorithm.BFS, Algorithm.IDS]:
            raise ValueError('If the algorithm is DFS, BFS, or IDS, heuristic_fun should be None.')
        if not heuristic_fun and algorithm in [Algorithm.A_STAR, Algorithm.IDA_STAR]:
            raise ValueError('If the algorithm is A* or IDA*, you should provide heuristic function.')
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError('Max_depth must be int and >= 1.')
        if not isinstance(max_n_explored, int) or max_n_explored < 1:
            raise ValueError('Max_n_explored must be int and >= 1.')

        # Store params
        self._config: dict = config
        self._algorithm: AlgorithmT = algorithm
        self._version: SearchVersionT = version
        self._source_puzzle: RushHourPuzzle = source_puzzle
        self._heuristic_fun: Optional[Callable[[RushHourPuzzle], float]] = heuristic_fun
        self._max_depth: int = max_depth
        self._max_n_explored: int = max_n_explored

        # Class vars based on the input params
        self._what_first: SearchWhatFirstT = SearchWhatFirst.DEPTH_FIRST \
            if algorithm in [Algorithm.DFS, Algorithm.IDS] else SearchWhatFirst.BREADTH_FIRST
        self._use_iter_deepening: bool = True if algorithm in [Algorithm.IDS, Algorithm.IDA_STAR] else False

        # Vars need renewing when searching
        self._cur_occur_max_depth: int = 0  # TODO: if depth reaches max, it continues and just cannot go deeper
        self._last_node: Optional[Node] = None
        self._explored_encoded_set: SortedSet = SortedSet()
        self._explored_list: List = []  # These two about explored nodes should be maintained simultaneously # TODO: make a func to renew these two toghether
        self._n_visited: int = 0
        self._sol_found: bool = False
        self._node_num_record: List[int] = []
        self._n_expand: int = 1
        self._cur_depth_limit: int = 1 if algorithm in [Algorithm.IDS, Algorithm.IDA_STAR] else max_depth
        self._loop_cnt: int = 0
        self._part_time: Dict[int, float] = {}  # How much time consumed in what part
        self._part_count: Dict[int, int] = {}  # TODO: use a function to renew these two at a time
        self._container: SearchContainer = SearchContainer(algorithm=algorithm)

        # For tmp used
        self._time_point_start: float = 0.

        # For setting
        self._show_part_info: bool = config['search_debug']['show_part_info']
        self._n_loop_to_show_part_info: int = config['search_debug']['n_loop_to_show_part_info']
        self._n_visited_to_show_info: int = config['search_debug']['n_visited_to_show_info']
        self._show_new_max_depth: bool = config['search_debug']['show_new_max_depth']
        ################ end of __init__ ####################

    def run(self):

        #
        end: bool = False
        start_time: float = time.time()

        # Iterative deepening loop
        while self._cur_depth_limit <= self._max_depth and not end:  # IDS and IDA* utilize this

            # Init container
            self._container = SearchContainer(algorithm=self._algorithm)
            self._container.add(node=self.make_init_node())

            # Init other vars
            self._cur_occur_max_depth = 0
            self._explored_encoded_set: SortedSet = SortedSet()  # Explored list doesn't need to init evey new depth limit.
            # Main loop
            while len(self._container) != 0:
                #
                self._loop_cnt += 1

                # For recording consuming time
                self.record_time_point_start()

                # Record current number of nodes in the container
                self._node_num_record.append(len(self._container))

                # Pick the next node.
                if self._algorithm in [Algorithm.BFS, Algorithm.A_STAR, Algorithm.IDA_STAR]:
                    node: Node = self._container.pop_front()
                elif self._algorithm in [Algorithm.DFS, Algorithm.IDS]:
                    node: Node = self._container.pop_back()
                else:
                    raise ValueError('Invalid algorithm:', self._algorithm)
                self._last_node = node

                # Record the max depth that has occurred so far
                if self._cur_occur_max_depth < node.depth:
                    self._cur_occur_max_depth = node.depth
                    if self._show_new_max_depth:
                        print("New max depth occurs:", self._cur_occur_max_depth)

                # Renew number of visited nodes
                self._n_visited += 1

                # Renew explored set
                if self._version == SearchVersion.GRAPH:
                    self.renew_explored(puzzle=node.rush_hour_puzzle)

                # Show info when visited number reaches specific amount
                if self._n_visited % self._n_visited_to_show_info == 0:
                    print("# n_visited, cur_node_num_in_container, "
                          "explored_list_len: {}, {}, {}".format(self._n_visited, self._node_num_record[-1],
                                                                 len(self._explored_encoded_set)))

                #
                self.renew_part_time_record(part_id=1)

                # Check if terminal

                # Check if find the solution
                if node.rush_hour_puzzle.is_solved():
                    self._sol_found = True
                    end = True
                    break

                # Check current total number of visited nodes and current max depth
                if len(self._explored_list) >= self._max_n_explored:
                    end = True
                    break

                #
                if node.depth < self._cur_depth_limit:
                    # Continue to dig out a deeper layer
                    # Expand children
                    for action in node.rush_hour_puzzle.actions:
                        self.record_time_point_start()

                        # Copy the puzzle
                        new_puzzle = RushHourPuzzle(config=self._config)
                        new_puzzle.set_board(copy.deepcopy(node.rush_hour_puzzle.board), renew_actions=False)

                        self.renew_part_time_record(part_id=2)

                        self.record_time_point_start()

                        # Apply action
                        new_puzzle.apply_action(action=action)

                        self.renew_part_time_record(part_id=3)

                        # If it's graph search, we don't visit a repeated state.
                        if self._version == SearchVersion.GRAPH:
                            if RushHourPuzzle.board_encoded_exists(self._explored_encoded_set, new_puzzle.encode()):
                                continue

                        self.record_time_point_start()

                        # Make a new node
                        new_node = Node(rush_hour_puzzle=new_puzzle,
                                        node_depth=node.depth + 1,
                                        so_far_n_conn=node.so_far_n_conn + 1,
                                        g_cost=node.g_cost + 1,
                                        h_cost=self._heuristic_fun(new_puzzle) if self._heuristic_fun else 0)
                        new_node.parent = node

                        # Append to the container
                        self._container.add(new_node)

                        # Renew number of nodes expanded
                        self._n_expand += 1

                        self.renew_part_time_record(part_id=4)

                if self._loop_cnt % self._n_loop_to_show_part_info == 0:
                    self.show_part_time_stat()

            #
            self._cur_depth_limit += 1

        #
        time_cost = time.time() - start_time  # Unit: sec

        # Backtrace the path if there's a solution found
        path: List[RushHourPuzzle] = []
        if self._sol_found:
            # Trace back the path
            node: Node = self._last_node
            path.append(node.rush_hour_puzzle)
            while node.parent:
                path.append(node.parent.rush_hour_puzzle)
                node = node.parent
            path.reverse()

        # Gather data
        return SearchStats(algorithm=self._algorithm,
                           version=self._version,
                           sol_found=self._sol_found,
                           path=path,
                           explored_list=self._explored_list,
                           node_num_record=self._node_num_record,
                           n_expand=self._n_expand,
                           time_cost=time_cost)

#
# def search(config: dict,
#            algorithm: Algorithm,
#            version: SearchVersion,
#            source_puzzle: RushHourPuzzle,
#            heuristic_fun: Union[Callable[[RushHourPuzzle], float], Any] = None,
#            max_depth: int = 99999,
#            max_n_explored: int = 100000) -> 'SearchStats':
#     # Check if algorithm valid
#     if algorithm not in [Algorithm.DFS, Algorithm.BFS, Algorithm.IDS, Algorithm.A_STAR, Algorithm.IDA_STAR]:
#         raise ValueError('Algorithm "{}" is invalid.'.format(algorithm))
#
#     # Whether heuristic func is valid here based on the algorithm
#     if algorithm not in [Algorithm.A_STAR, Algorithm.IDA_STAR]:
#         print('Warning: You are using the algorithm which does not need heuristic function '
#               'but you give it a heuristic function.')
#         heuristic_fun = None
#
#     #
#     start_time = time.time()
#
#     # Need to be renewed
#     cur_max_depth: int = 0
#     last_node = None
#     explored_encoded_set: SortedSet = SortedSet()
#     explored_list: List = []  # These two about explored nodes should be maintained simultaneously
#     n_visited = 0
#     sol_found: bool = False
#     node_num_record: List = []
#     n_expand: int = 1
#
#     # For IDS and IDA*
#     cur_depth_limit = 1
#
#     # Time
#     loop_count = 0
#     part_1_time = 0.0
#     part_1_count = 1
#     part_2_time = 0.0
#     part_2_count = 0
#     part_2_2_time = 0.0
#     part_2_2_count = 0
#     part_3_time = 0.0
#     part_3_count = 0
#     part_4_time = 0.0
#     part_4_count = 0
#
#     # Each SortedKeyList contains the same group of nodes.
#     # While choosing next node, we pick from the group with the smallest idx in container list.
#     container: Deque[SortedKeyList] = deque()
#
#     # Put into the source node
#     container.append(SortedKeyList([Node(
#         rush_hour_puzzle=source_puzzle,
#         node_depth=1,
#         so_far_n_conn=0,
#         g_cost=0,
#         h_cost=heuristic_fun(source_puzzle) if heuristic_fun else 0)], key=Node.compare_key))
#
#     #
#     while len(container) != 0:
#         # All "st" here is for recording consuming time
#         st1 = time.time()
#
#         # Record current number of nodes in the container
#         cur_node_num = sum(len(x) for x in container)
#         node_num_record.append(cur_node_num)
#
#         # Pick the next node.
#         node: Node = container[0].pop(0)
#         last_node = node
#         if len(container[0]) == 0:
#             container.popleft()
#         if cur_max_depth < node.depth:
#             cur_max_depth = node.depth
#             print("Cur_max_dept:", cur_max_depth)
#         n_visited += 1
#
#         #
#         # TODO: Write down why not hash?
#         if version == SearchVersion.GRAPH:
#             explored_encoded_set.add(node.rush_hour_puzzle.encode())
#             explored_list.append(node.rush_hour_puzzle)
#             # TODO: Make tree version have no access to explored record.
#
#         if n_visited % 3000 == 0:
#             print("n_visited, cur_node_num_in_container, explored_list_len: {}, {}, {}". \
#             format(n_visited, cur_node_num,
#                                                                                                len(explored_list)))
#         st2 = time.time()
#         part_1_time += st2 - st1
#         part_1_count += 1
#         # Check if terminal
#
#         # Check if find the solution
#         if node.rush_hour_puzzle.is_solved():
#             sol_found = True
#             break
#
#         # Check current total number of visited nodes and current max depth
#         if len(explored_list) >= max_n_explored or cur_max_depth >= max_depth:
#             break
#
#         # In IDS and IDA*, a child whose depth is more than the limit won't be added to the container
#         if algorithm in [Algorithm.IDS, Algorithm.IDA_STAR] and node.depth == cur_depth_limit:
#             if len(container) == 0:
#                 cur_depth_limit += 1
#                 explored_encoded_set = SortedSet()
#                 cur_max_depth = 0
#                 container  # TODO: not yet
#
#         # Continue to dig out a deeper layer
#         sorted_key_list = SortedKeyList([], key=Node.compare_key)
#         for action in node.rush_hour_puzzle.actions:
#             st3 = time.time()
#             # new_puzzle = copy.deepcopy(node.rush_hour_puzzle)
#             new_puzzle = RushHourPuzzle(config=config)
#             new_puzzle.set_board(copy.deepcopy(node.rush_hour_puzzle.board), renew_actions=False)
#
#             st3_2 = time.time()
#
#             part_2_count += 1
#             part_2_time += st3_2 - st3
#
#             st3_3 = time.time()
#             new_puzzle.apply_action(action=action)
#             st4 = time.time()
#             part_2_2_count += 1
#             part_2_2_time += st4 - st3_3
#
#             # If it's graph search, we don't visit a repeated state.
#             if version == SearchVersion.GRAPH:
#                 if RushHourPuzzle.board_encoded_exists(explored_encoded_set, new_puzzle.encode()):
#                     continue
#
#             st5 = time.time()
#             # Make a new node
#             new_node = Node(rush_hour_puzzle=new_puzzle,
#                             node_depth=node.depth + 1,
#                             so_far_n_conn=node.so_far_n_conn + 1,
#                             g_cost=node.g_cost + 1,
#                             h_cost=heuristic_fun(new_puzzle) if heuristic_fun else 0)
#             new_node.parent = node
#             sorted_key_list.add(new_node)
#
#             # Renew number of nodes expanded
#             n_expand += 1
#
#             st6 = time.time()
#             part_3_count += 1
#             part_3_time += st6 - st5
#
#         # Put int the container, depending on the algorithm chose
#         st7 = time.time()
#         if len(sorted_key_list) > 0:
#             if algorithm in [Algorithm.DFS, Algorithm.IDS]:
#                 container.append(sorted_key_list)
#             else:
#                 if len(container) == 0:
#                     container.append(sorted_key_list)
#                 elif len(container) == 1:
#                     container[0] += sorted_key_list
#                 else:
#                     raise ValueError('Why it come into this "else"?')
#         st8 = time.time()
#         part_4_count += 1
#         part_4_time += st8 - st7
#
#         if part_1_count % 1000 == 0:
#             loop_count += 1
#             try:
#                 print('Part avg time (loop={}):'.format(loop_count * 1000))
#                 t1 = part_1_time / part_1_count
#                 t2 = part_2_time / part_2_count
#                 t2_2 = part_2_2_time / part_2_2_count
#                 t3 = part_3_time / part_3_count
#                 t4 = part_4_time / part_4_count
#                 print('1: {:.8f}'.format(t1))
#                 print('2: {:.8f}'.format(t2))
#                 print('2-2: {:.8f}'.format(t2_2))
#                 print('3: {:.8f}'.format(t3))
#                 print('4: {:.8f}'.format(t4))
#                 print('ep: {:.8f}'.format(t2 + t2_2 + t3 + t4))
#             except ZeroDivisionError:
#                 pass
#             part_1_time = 0.0
#             part_1_count = 1
#             part_2_time = 0.0
#             part_2_count = 0
#             part_2_2_time = 0.0
#             part_2_2_count = 0
#             part_3_time = 0.0
#             part_3_count = 0
#             part_4_time = 0.0
#             part_4_count = 0
#
#     #
#     time_cost = time.time() - start_time  # Unit: sec
#
#     # Backtrace the path if there's a solution found
#     path: List[RushHourPuzzle] = []
#     if sol_found:
#         # Trace back the path
#         node: Node = last_node
#         path.append(node.rush_hour_puzzle)
#         while node.parent:
#             path.append(node.parent.rush_hour_puzzle)
#             node = node.parent
#         path.reverse()
#
#     # Gather data
#     return SearchStats(algorithm=algorithm,
#                        version=version,
#                        sol_found=sol_found,
#                        path=path,
#                        explored_list=explored_list,
#                        node_num_record=node_num_record,
#                        n_expand=n_expand,
#                        time_cost=time_cost)
