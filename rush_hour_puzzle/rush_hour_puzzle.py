# Reference:
# https://stackoverflow.com/questions/10194482/custom-matplotlib-plot-chess-board-like-table-with-colored-cells
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from matplotlib import colors
from numpy import ndarray
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from typing import Sequence, Union, Callable, Any, List, Deque, Set, NewType
import copy
import cv2
import io
from sortedcontainers import SortedSet
import time

DirectionT = NewType('DirectionT', int)


class Direction:
    UP: DirectionT = 0
    RIGHT: DirectionT = 1
    DOWN: DirectionT = 2
    LEFT: DirectionT = 3


class Action:
    def __init__(self, car_idx: int,
                 top_left_cell_row: int,
                 top_left_cell_col: int,
                 car_len: int,
                 move_direction: Union[DirectionT, int],
                 move_len: int):
        # What does "top left cell" means here?
        #  _ _ _ _ _ _
        # |_|_|_|_|_|_|
        # |_|_|_|_|_|_|
        # |_|C|a|r|_|_|
        # |_|_|_|_|_|_|
        # |_|_|_|C|_|_|
        # |_|_|_|a|_|_|
        # |_|_|_|r|_|_|
        # The two "C" of the two "Car" are their top_left_cell respectively.
        self.car_idx: int = car_idx
        self.top_left_cell_row: int = top_left_cell_row
        self.top_left_cell_col: int = top_left_cell_col
        self.car_len: int = car_len
        self.move_direction: int = move_direction
        self.move_len: int = move_len


class RushHourPuzzle:
    # We use "idx" to indicate the situation of a grid.
    # Idx "BACKGROUND_IDX" indicate that grid is empty.
    # Idx 0 is the car that we have to accomplish solving this puzzle by
    # move it to the specific position.
    BACKGROUND_IDX = -1
    BACKGROUND_COLOR: Union[str, Any] = None
    CAR_COLORS: Union[List[str], Any] = None

    def __init__(self, config: dict):
        # Initialize static vars
        if not RushHourPuzzle.BACKGROUND_COLOR:
            RushHourPuzzle.BACKGROUND_COLOR = config['board']['background_color']
            RushHourPuzzle.CAR_COLORS = config['board']['car_colors']

        # Initialization
        self._nrows: int = config['board']['n_rows']
        self._ncols: int = config['board']['n_cols']
        self._board: ndarray = self.empty_board()
        self._actions: List[Action] = self.get_legal_actions()
        self._last_action: Union[Action, None] = None  # Indicate what is the last action that lead to this board state

    @property
    def nrows(self) -> int:
        return self._nrows

    @property
    def ncols(self) -> int:
        return self._ncols

    @property
    def board(self) -> ndarray:
        return self._board

    @board.setter
    def board(self, board: ArrayLike):
        self.set_board(board=board)

    @property
    def last_action(self) -> Union["Action", None]:
        return self._last_action

    @property
    def actions(self) -> List[Action]:
        return self._actions

    def set_board(self, board: ArrayLike, renew_actions: bool = True):
        # Ensure format
        board = np.array(board)
        # Check size
        _nrows, _ncols = board.shape
        if _nrows != self._nrows or _ncols != self._ncols:
            raise ValueError(
                'Board size must be the same as ({},{}) but got ({},{}).'.format(self._nrows, self._ncols, _nrows,
                                                                                 _ncols))
        # Self
        self._board = board
        # Renew legal actions
        if renew_actions:
            # start_time = time.time()
            self._actions = self.get_legal_actions()
            # print('{:.8f}'.format(time.time() - start_time))

    def empty_board(self) -> ndarray:
        return np.full((self._nrows, self._ncols), self.BACKGROUND_IDX)

    def show_board(self,
                   show_img: bool = True,
                   show_actions=True,
                   return_cv2mat=False,
                   side_length: int = 10,
                   dpi: int = 80):
        # Reference:
        #     https://stackoverflow.com/questions/34975972/how-can-i-make-a-video-from-array-of-images-in-matplotlib
        # side_length's unit is "inch"

        # Set colors
        cmap = colors.ListedColormap([self.BACKGROUND_COLOR] + self.CAR_COLORS)
        bounds = [-1.5 + x for x in range(len(self.CAR_COLORS) + 2)]  # [-1.5, -0.5, 0.5, ......, car_max_idx + 0.5]
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)

        fig = plt.figure(figsize=(side_length, side_length), dpi=dpi)

        plt.imshow(self._board, cmap=cmap, norm=norm)

        row_labels = range(self._nrows)
        col_labels = range(self._ncols)
        plt.xticks(range(self._ncols), col_labels, fontsize=25)
        plt.yticks(range(self._nrows), row_labels, fontsize=25)
        if show_actions:
            for action in self._actions:
                if action.move_direction == Direction.UP:
                    x = action.top_left_cell_col
                    y = action.top_left_cell_row
                    dx = 0
                    dy = -1 * action.move_len
                elif action.move_direction == Direction.RIGHT:
                    x = action.top_left_cell_col + action.car_len - 1
                    y = action.top_left_cell_row
                    dx = action.move_len
                    dy = 0
                elif action.move_direction == Direction.DOWN:
                    x = action.top_left_cell_col
                    y = action.top_left_cell_row + action.car_len - 1
                    dx = 0
                    dy = action.move_len
                else:
                    x = action.top_left_cell_col
                    y = action.top_left_cell_row
                    dx = -1 * action.move_len
                    dy = 0
                plt.arrow(x=x, y=y, dx=dx, dy=dy, length_includes_head=True, head_width=0.25, head_length=0.3
                          )

        returns = []
        if return_cv2mat:
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='raw', dpi=dpi)
            io_buf.seek(0)
            mat = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
            io_buf.close()

            # canvas = FigureCanvas(fig)
            # canvas.draw()
            # # mat = np.array(canvas.renderer.buffer_rgba())
            # mat = np.array(canvas.renderer._renderer)
            mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
            returns.append(mat)
        if show_img:
            plt.show(block=False)
        else:
            plt.close(fig)

        # Return
        if len(returns) == 0:
            return None
        elif len(returns) == 1:
            return returns[0]
        else:
            return returns

    def show_18_colors(self):
        if self._nrows != 6 or self._ncols != 6:
            raise ValueError('You can not show the 18 colors if the board size is not (6,6).')
        # List 18 colors, background color is inserted between each one
        board = np.array(
            [0, -1, 1, -1, 2, -1, -1, 3, -1, 4, -1, 5, 6, -1, 7, -1, 8, -1, -1, 9, -1, 10, -1, 11, 12, -1, 13, -1, 14,
             -1, -1, 15, -1, 16, -1, 17]).reshape(6, 6)
        self.set_board(board=board)
        self.show_board()

    # Input: 2d arr
    #     [car_idx, top_left_cell_row, top_left_cell_col, length, orientation]
    #     - length: between 2 and 3
    #     - orientation: 1 for horizontal, 2 for vertical
    def transform_car_info_into_board(self, arr2d: Union[Sequence, str]):
        # If input is str, we need to do parsing
        if isinstance(arr2d, str):
            result = []
            for line in arr2d.split('\n'):
                if line != '':
                    arr = line.strip().split(' ')
                    result.append([int(x) for x in arr if x != ''])
            arr2d = np.array(result)
        #
        board = self.empty_board()
        for arr in arr2d:
            car_idx, top_left_cell_row, top_left_cell_col, length, orientation = arr
            for i in range(length):
                target_row = top_left_cell_row + i if orientation == 2 else top_left_cell_row
                target_col = top_left_cell_col + i if orientation == 1 else top_left_cell_col
                # Check if target outside the board
                if not 0 <= target_row <= self._nrows or not 0 <= target_col <= self._ncols:
                    raise ValueError(
                        'The car (index: {}) will occupy the grid ({},{}) which is outside the board of size\
                         ({},{}).'.format(
                            car_idx, target_row, target_col, self._nrows, self._ncols)
                    )
                    # Check if already put a car
                if board[target_row, target_col] != self.BACKGROUND_IDX:
                    raise ValueError(
                        'The car (index: {}) will occupy the grid ({},{}) that there is already a car there.'.format(
                            car_idx, target_row, target_col, self._nrows, self._ncols
                        ))
                # Fill it
                board[target_row, target_col] = car_idx
        self.set_board(board=board)

    # Get all legal actions from the current state
    def get_legal_actions(self) -> List[Action]:
        board = self._board
        # From top-left
        checked_grid = np.full((self._nrows, self._ncols), False)  # Value of explored grids are True
        height, width = board.shape
        _actions = []
        for i in range(height):
            for j in range(width):
                # Check if explored
                if checked_grid[i][j]:
                    continue
                # Check if no car is on this grid
                if board[i][j] == self.BACKGROUND_IDX:
                    checked_grid[i, j] = True
                    # print(f'({i},{j}) empty and explored.')
                    continue
                # Get its car index
                car_idx = board[i, j]
                # Get its orientation
                orientation = 1 if j + 1 < width and board[i, j + 1] == car_idx else 2
                # Know whether its length is 3
                if orientation == 1:
                    length = 3 if j + 2 < width and board[i, j + 2] == car_idx else 2
                else:
                    length = 3 if i + 2 < height and board[i + 2, j] == car_idx else 2
                # Find all legal actions (along two directions)
                # Up/Left
                for direction in ('Up/Left', 'Down/Right'):
                    # Add or subtract depending on direction
                    if direction == 'Up/Left':
                        try_point = [i, j - 1] if orientation == 1 else [i - 1, j]
                        move_direction = Direction.LEFT if orientation == 1 else Direction.UP
                    else:
                        try_point = [i, j + length] if orientation == 1 else [i + length, j]
                        move_direction = Direction.RIGHT if orientation == 1 else Direction.DOWN
                    if not (0 <= try_point[0] < self._nrows) or not (0 <= try_point[1] < self._ncols):
                        continue
                    # See if there are empty grids
                    while board[try_point[0]][try_point[1]] == self.BACKGROUND_IDX:
                        #
                        if direction == 'Up/Left':
                            move_len = abs(try_point[1] - j) if orientation == 1 else abs(try_point[0] - i)
                        else:
                            move_len = abs(try_point[1] - j - length + 1) if orientation == 1 else abs(
                                try_point[0] - i - length + 1)
                        # Add an action
                        _actions.append(
                            Action(car_idx=car_idx,
                                   top_left_cell_row=i,
                                   top_left_cell_col=j,
                                   car_len=length,
                                   move_direction=move_direction,
                                   move_len=move_len
                                   ))
                        # Move more
                        if direction == 'Up/Left':
                            if orientation == 1:
                                try_point[1] -= 1
                            else:
                                try_point[0] -= 1
                        else:
                            if orientation == 1:
                                try_point[1] += 1
                            else:
                                try_point[0] += 1

                        if not (0 <= try_point[0] < self._nrows) or not (0 <= try_point[1] < self._ncols):
                            break
                # Record those are explored
                for ii in range(length):
                    iii = i + ii if orientation == 2 else i
                    jjj = j + ii if orientation == 1 else j
                    checked_grid[iii][jjj] = True
        return _actions

    def apply_action(self, action: Action,
                     return_board_and_not_renew: bool = False):
        board = self._board
        # Move the number of grids that equals the car length
        for i in range(0, action.car_len):
            # Find the new grid that one grid of the car is going to move to
            if action.move_direction == Direction.UP:
                cur_row = action.top_left_cell_row + i
                cur_col = action.top_left_cell_col
                row = cur_row - action.move_len
                col = cur_col
            elif action.move_direction == Direction.RIGHT:
                cur_row = action.top_left_cell_row
                cur_col = action.top_left_cell_col + action.car_len - 1 - i
                row = cur_row
                col = cur_col + action.move_len
            elif action.move_direction == Direction.DOWN:
                cur_row = action.top_left_cell_row + action.car_len - 1 - i
                cur_col = action.top_left_cell_col
                row = cur_row + action.move_len
                col = cur_col
            else:  # elif action.move_direction == Direction.LEFT:
                cur_row = action.top_left_cell_row
                cur_col = action.top_left_cell_col + i
                row = cur_row
                col = cur_col - action.move_len
            # Check if the target grid outside the board
            if not (0 <= row < self._nrows) or not (0 <= col < self._ncols):
                raise ValueError('The grid ({},{}) is outside the board. There may be some errors in action part.\
                                 '.format(row, col))
            # Check if there's other car on that grid to ensure the action is valid
            if board[row][col] != self.BACKGROUND_IDX and board[row][col] != action.car_idx:
                raise ValueError(
                    'There is a car (idx={}) on the grid ({},{}) that the car (idx={}) is going to move there. \
                    There may be some errors in action part.'.format(board[row][col], row, col, action.car_idx))
            # Erase the current grid and fill the new grid
            board[row][col] = action.car_idx
            board[cur_row][cur_col] = self.BACKGROUND_IDX

        #
        self.set_board(board=board, renew_actions=True)
        self._last_action = action

    def is_solved(self):
        # When car (idx=0) arrive (2,4), (2,5)
        if self._board[2][4] == 0 and self._board[2][5] == 0:
            return True
        else:
            return False

    def encode(self, mark_depth: bool = False, depth: int = -1) -> str:
        if mark_depth and depth < 0:
            raise ValueError('You should provide depth if you want to mark depth.')
        if mark_depth:
            encoded = '{}'.format(depth)
        else:
            encoded = ''
        encoded = ''
        for i in range(self._nrows):
            for j in range(self._ncols):
                encoded += " " + str(self._board[i][j])
        return encoded

    @staticmethod
    def board_exists(boards: Sequence['RushHourPuzzle'], s: 'RushHourPuzzle') -> bool:
        for node in boards:
            if np.array_equal(node.board, s.board):
                return True
        return False

    @staticmethod
    def board_encoded_exists(boards_encoded: SortedSet, s_encoded: str) -> bool:
        return s_encoded in boards_encoded
