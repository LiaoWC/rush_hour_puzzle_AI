# Reference:
# https://stackoverflow.com/questions/10194482/custom-matplotlib-plot-chess-board-like-table-with-colored-cells
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from typing import Sequence, Union, Tuple
import copy
import cv2
import io


class Direction:
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Action:
    def __init__(self, car_idx: int, top_left_cell_row: int, top_left_cell_col: int, car_len: int,
                 move_direction: Union[Direction, int],
                 move_len: int):
        self.car_idx = car_idx
        self.top_left_cell_row = top_left_cell_row
        self.top_left_cell_col = top_left_cell_col
        self.car_len = car_len
        self.move_direction = move_direction
        self.move_len = move_len


class RushHourPuzzleBoard:
    # -1: background color
    # 0~17: car colors (max number of car is 18 since min length of a car is 2.)
    # Reference: https://matplotlib.org/stable/gallery/color/named_colors.html
    BACKGROUND_IDX = -1
    BACKGROUND_COLOR = 'lightgray'
    # Car with index 0 is the car you want to move out the board
    CAR_COLOR = [
        'red',
        'aqua',
        'yellow',
        'greenyellow',
        'orange',
        'violet',
        'darkolivegreen',
        'dodgerblue',
        'royalblue',
        'paleturquoise',
        'darkgoldenrod',
        'hotpink',
        'cornflowerblue',
        'linen',
        'olive',
        'turquoise',
        'violet',
        'mediumslateblue'
    ]

    def __init__(self, n_rows: int = 6, n_cols: int = 6):
        # Initialization
        self.n_rows, self.n_cols = n_rows, n_cols
        self.board = self.empty_board()
        self.legal_actions = self.get_legal_actions()

    def empty_board(self) -> np.ndarray:
        return np.full((self.n_rows, self.n_cols), self.BACKGROUND_IDX)

    def set_board(self, board: Sequence):
        # Ensure format
        board = np.array(board)
        # Check size
        n_rows, n_cols = board.shape
        if n_rows != self.n_rows or n_cols != self.n_cols:
            raise ValueError(
                'Board size must be the same as ({},{}) but got ({},{}).'.format(self.n_rows, self.n_cols, n_rows,
                                                                                 n_cols))
        # Set
        self.board = board
        # Renew legal actions
        self.legal_actions = self.get_legal_actions()

    def show_board(self, show_img: bool = True, show_legal_actions=False, return_cv2mat=False, side_length: int = 10,
                   dpi: int = 80):
        # Reference:
        #     https://stackoverflow.com/questions/34975972/how-can-i-make-a-video-from-array-of-images-in-matplotlib
        # side_length's unit is "inch"

        # Set colors
        cmap = colors.ListedColormap([self.BACKGROUND_COLOR] + self.CAR_COLOR)
        bounds = [-1.5 + x for x in range(len(self.CAR_COLOR) + 2)]  # [-1.5, -0.5, 0.5, ......, car_max_idx + 0.5]
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)

        fig = plt.figure(figsize=(side_length, side_length), dpi=dpi)

        plt.imshow(self.board, cmap=cmap, norm=norm)

        row_labels = range(self.n_rows)
        col_labels = range(self.n_cols)
        plt.xticks(range(self.n_cols), col_labels, fontsize=25)
        plt.yticks(range(self.n_rows), row_labels, fontsize=25)
        if show_legal_actions:
            for action in self.legal_actions:
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
        if self.n_rows != 6 or self.n_cols != 6:
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
                if not 0 <= target_row <= self.n_rows or not 0 <= target_col <= self.n_cols:
                    raise ValueError(
                        'The car (index: {}) will occupy the grid ({},{}) which is outside the board of size\
                         ({},{}).'.format(
                            car_idx, target_row, target_col, self.n_rows, self.n_cols)
                    )
                    # Check if already put a car
                if board[target_row, target_col] != self.BACKGROUND_IDX:
                    raise ValueError(
                        'The car (index: {}) will occupy the grid ({},{}) that there is already a car there.'.format(
                            car_idx, target_row, target_col, self.n_rows, self.n_cols
                        ))
                # Fill it
                board[target_row, target_col] = car_idx
        self.set_board(board=board)

    # Get all legal actions from the current state
    def get_legal_actions(self):
        board = self.board
        # From top-left
        checked_grid = np.full((self.n_rows, self.n_cols), False)  # Value of explored grids are True
        height, width = board.shape
        legal_actions = []
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
                    if not (0 <= try_point[0] < self.n_rows) or not (0 <= try_point[1] < self.n_cols):
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
                        legal_actions.append(
                            Action(car_idx=car_idx,
                                   top_left_cell_row=i,
                                   top_left_cell_col=j,
                                   car_len=length,
                                   move_direction=move_direction,
                                   move_len=move_len
                                   ))
                        # Move more
                        if orientation == 1:
                            try_point[1] += 1
                        else:
                            try_point[0] += 1

                        if not (0 <= try_point[0] < self.n_rows) or not (0 <= try_point[1] < self.n_cols):
                            break
                # Record those are explored
                for ii in range(length):
                    iii = i + ii if orientation == 2 else i
                    jjj = j + ii if orientation == 1 else j
                    checked_grid[iii][jjj] = True
        return legal_actions

    def apply_action(self, action: Action):
        board = self.board
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
            if not (0 <= row < self.n_rows) or not (0 <= col < self.n_cols):
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
        self.set_board(board=board)

    def is_terminal(self):
        # When car (idx=0) arrive (2,4), (2,5)
        if self.board[2][4] == 0 and self.board[2][5] == 0:
            return True
        else:
            return False


##########################################
def make_video(name: str, size: Tuple[int, int], cv2mats: Sequence[npt.ArrayLike], fps: int = 1.25):
    video = cv2.VideoWriter(f'{name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for cv2mat in cv2mats:
        video.write(cv2mat)
    cv2.destroyAllWindows()
    video.release()


def board_exists(path: Sequence[RushHourPuzzleBoard], s: RushHourPuzzleBoard) -> bool:
    for node in path:
        if np.array_equal(node.board, s.board):
            return True
    return False


def dfs_tree_non_recursive(source_board: RushHourPuzzleBoard):
    path = []
    waiting_stack = [source_board]
    sol_found = False
    max_path_len = 1000
    while len(waiting_stack) != 0:
        s = waiting_stack.pop()

        if not board_exists(path, s):
            path.append(s)
        else:
            continue

        # Check if terminal
        if s.is_terminal():
            sol_found = True
            break

        # Check current length of path
        if len(path) >= max_path_len:
            break

        for action in s.legal_actions:
            new_s = copy.deepcopy(s)
            new_s.apply_action(action=action)
            waiting_stack.append(new_s)
    print('Whether found:', sol_found)
    if sol_found:
        return path
    else:
        return None


a = RushHourPuzzleBoard()

eg_input = '''
            0 2 3 2 1
            1 0 0 3 1
            2 1 3 3 1
            3 3 1 2 1
            4 5 0 3 1
            5 1 0 2 2
            7 1 2 2 2
            8 3 3 3 2
            9 4 4 2 2
            10 2 5 2 2
            11 4 5 2 2'''

# aaa = []
#
a.transform_car_info_into_board(eg_input)
# aaa.append(a.show_board(show_legal_actions=True, return_cv2mat=True))
#
# a.apply_action(a.legal_actions[0])
# aaa.append(a.show_board(show_legal_actions=True, return_cv2mat=True))
# a.apply_action(a.legal_actions[0])
# aaa.append(a.show_board(show_legal_actions=True, return_cv2mat=True))
# a.apply_action(a.legal_actions[0])
# aaa.append(a.show_board(show_legal_actions=True, return_cv2mat=True))
# a.apply_action(a.legal_actions[0])
# aaa.append(a.show_board(show_legal_actions=True, return_cv2mat=True))
# a.apply_action(a.legal_actions[0])
# aaa.append(a.show_board(show_legal_actions=True, return_cv2mat=True))
# a.apply_action(a.legal_actions[0])
# aaa.append(a.show_board(show_legal_actions=True, return_cv2mat=True))
# a.apply_action(a.legal_actions[0])
# aaa.append(a.show_board(show_legal_actions=True, return_cv2mat=True))
# a.apply_action(a.legal_actions[0])
# aaa.append(a.show_board(show_legal_actions=True, return_cv2mat=True))


#
path_found = dfs_tree_non_recursive(source_board=a)
if path_found:
    img_to_be_video = []
    for state in path_found:
        img_to_be_video.append(state.show_board(show_legal_actions=True, show_img=False, return_cv2mat=True))
    make_video('dfs', (img_to_be_video[0].shape[0], img_to_be_video[0].shape[1]), img_to_be_video, fps=2)

# 1. S -> all legal actions
# 2. S ____> S' after do an action

# TODO: receive car info from stdout "<" symbol
# Prepare for making a video
