from .rush_hour_puzzle import RushHourPuzzle, Action, Direction, DirectionT
from typing import List, Tuple, Sequence, Union, Optional, Set
import copy
import random

RED_CAR_ROW = 2
MIN_CAR_LEN = 2
MAX_CAR_LEN = 3
RED_CAR_IDX = 0
N_ROWS = 6
N_COLS = 6


# Tuple: (car_idx, move_direction)
def encode_car_idx_move_direction(car_idx: int, move_direction: DirectionT) -> str:
    return str(car_idx) + ' ' + str(move_direction)


def find_all_blocking_cars_and_its_suitable_move(rhp: RushHourPuzzle) \
        -> List[Tuple[int, int, int, int, List[Tuple[DirectionT, int]]]]:
    assert MIN_CAR_LEN == 2
    assert MAX_CAR_LEN == 3
    # [(top-left-cell-row, top-left-cell-col, car_len, [(direction, move_len), (...), ...]), (...), ...]
    result: List[Tuple[int, int, int, int, List[Tuple[DirectionT, int]]]] = []
    car_indices = []
    for j in range(rhp.ncols - 1, -1, -1):
        car_idx = rhp.board[RED_CAR_ROW][j]

        if car_idx == RushHourPuzzle.BACKGROUND_IDX:
            continue

        if car_idx == RED_CAR_IDX:
            break

        if car_idx in car_indices:
            raise ValueError('There should be no horizontal car between red car and the gateway.')
        car_indices.append(car_idx)

        car_len = 1
        left_top_cell_row = RED_CAR_ROW

        # Up
        for i in range(RED_CAR_ROW - 1, -1, -1):
            if rhp.board[i][j] == car_idx:
                car_len += 1
                left_top_cell_row = i

        # Down
        for i in range(RED_CAR_ROW + 1, rhp.nrows):
            if rhp.board[i][j] == car_idx:
                car_len += 1

        # Find its suitable move to let us move red car out
        suitable_moves = []
        if car_len == 3:
            suitable_moves.append((Direction.DOWN, RED_CAR_ROW - left_top_cell_row + 1))
        elif car_len == 2:
            if left_top_cell_row == 1:
                suitable_moves.append((Direction.UP, 1))
            else:
                # Check if the top two grid contains a car
                if not (rhp.board[0][j] != RushHourPuzzle.BACKGROUND_IDX and (rhp.board[0][j] == rhp.board[1][j])):
                    suitable_moves.append((Direction.UP, 2))
            # check if there's a car blocking down
            if not (rhp.board[4][j] != RushHourPuzzle.BACKGROUND_IDX and (
                    rhp.board[3][j] == rhp.board[4][j] or rhp.board[4][j] == rhp.board[5][j])):
                suitable_moves.append((Direction.DOWN, RED_CAR_ROW - left_top_cell_row + 1))
        else:
            raise ValueError('Invalid car length:', car_len)

        result.append((car_idx, left_top_cell_row, j, car_len, suitable_moves))
    return result


def touch_car(rhp: RushHourPuzzle,
              touch_direction: DirectionT,
              touch_row: int,
              touch_col: int) -> List[Tuple[int, int, int, DirectionT, int, int]]:
    # return: List of tuple of (car_idx, top_left_cell_row, top_left_cell_col, direction, move_len, car_len)
    try:
        car_idx = rhp.board[touch_row][touch_col]
    except:
        pass
    if car_idx == RushHourPuzzle.BACKGROUND_IDX:
        return []  # No car to touch
    # Get car len and position and orientation
    if touch_direction == Direction.UP:
        if touch_row != 0 and rhp.board[touch_row - 1][touch_col] == car_idx:
            orientation = 2  # 2 is vertical
        else:
            orientation = 1
    elif touch_direction == Direction.DOWN:
        if touch_row != (N_ROWS - 1) and rhp.board[touch_row + 1][touch_col] == car_idx:
            orientation = 2
        else:
            orientation = 1
    elif touch_direction == Direction.RIGHT:
        if touch_col != (N_COLS - 1) and rhp.board[touch_row][touch_col + 1] == car_idx:
            orientation = 1
        else:
            orientation = 2
    else:
        if touch_col != 0 and rhp.board[touch_row][touch_col - 1] == car_idx:
            orientation = 1
        else:
            orientation = 2

    car_len = 1
    top_left_cell_row = touch_row
    top_left_cell_col = touch_col
    if touch_direction in [Direction.UP, Direction.DOWN]:
        if orientation == 1:
            for j in range(touch_col + 1, N_COLS):
                if rhp.board[touch_row][j] == car_idx:
                    car_len += 1
            for j in range(touch_col - 1, -1, -1):
                if rhp.board[touch_row][j] == car_idx:
                    car_len += 1
                    top_left_cell_col = j
        else:
            if touch_direction == Direction.UP:
                for i in range(touch_row - 1, -1, -1):
                    if rhp.board[i][touch_col] == car_idx:
                        car_len += 1
                        top_left_cell_row = i
            else:
                for i in range(touch_row + 1, N_ROWS):
                    if rhp.board[i][touch_col] == car_idx:
                        car_len += 1
    else:
        if orientation == 2:
            for i in range(touch_row + 1, N_ROWS):
                if rhp.board[i][touch_col] == car_idx:
                    car_len += 1
            for i in range(touch_row - 1, -1, -1):
                if rhp.board[i][touch_col] == car_idx:
                    car_len += 1
                    top_left_cell_row = i
        else:
            if touch_direction == Direction.LEFT:
                for j in range(touch_col - 1, -1, -1):
                    if rhp.board[touch_row][j] == car_idx:
                        car_len += 1
                        top_left_cell_row = j
            else:
                for j in range(touch_col + 1, N_COLS):
                    if rhp.board[j][touch_col] == car_idx:
                        car_len += 1

    # Get where to go
    results = []
    if touch_direction in [Direction.UP, Direction.DOWN]:
        if orientation == 1:
            # Left
            left_len = touch_col
            if left_len >= car_len:
                occur_car = {}
                for jj in range(0, touch_col):
                    other_car_idx = rhp.board[touch_row][jj]
                    occur_car[other_car_idx] = occur_car.get(other_car_idx, 0) + 1
                segment_car_total_len = 0
                for key in occur_car:
                    if occur_car[key] >= 2 and key not in [RushHourPuzzle.BACKGROUND_IDX, car_idx]:
                        segment_car_total_len += occur_car[key]
                if left_len - segment_car_total_len >= car_len:
                    results.append((car_idx, top_left_cell_row, top_left_cell_col, Direction.LEFT,
                                    top_left_cell_col + car_len - touch_col, car_len))
            # Right
            right_len = N_COLS - touch_col - 1
            if right_len >= car_len:
                occur_car = {}
                for jj in range(touch_col + 1, N_COLS):
                    other_car_idx = rhp.board[touch_row][jj]
                    occur_car[other_car_idx] = occur_car.get(other_car_idx, 0) + 1
                segment_car_total_len = 0
                for key in occur_car:
                    if occur_car[key] >= 2 and key not in [RushHourPuzzle.BACKGROUND_IDX, car_idx]:
                        segment_car_total_len += occur_car[key]
                if right_len - segment_car_total_len >= car_len:
                    results.append((car_idx, top_left_cell_row, top_left_cell_col, Direction.RIGHT,
                                    touch_col - top_left_cell_col + 1, car_len))
        else:
            # Vertical
            if touch_direction == Direction.UP:
                vertical_len = touch_row
                if vertical_len >= car_len:
                    occur_car = {}
                    for ii in range(0, touch_row):
                        other_car_idx = rhp.board[ii][touch_col]
                        occur_car[other_car_idx] = occur_car.get(other_car_idx, 0) + 1
                    segment_car_total_len = 0
                    for key in occur_car:
                        if occur_car[key] >= 2 and key not in [RushHourPuzzle.BACKGROUND_IDX, car_idx]:
                            segment_car_total_len += occur_car[key]
                    if vertical_len - segment_car_total_len >= car_len:
                        results.append((car_idx, top_left_cell_row, top_left_cell_col, Direction.UP,
                                        top_left_cell_row + car_len - touch_row, car_len))
            else:
                vertical_len = touch_row
                if vertical_len >= car_len:
                    occur_car = {}
                    for ii in range(touch_row + 1, N_ROWS):
                        other_car_idx = rhp.board[ii][touch_col]
                        occur_car[other_car_idx] = occur_car.get(other_car_idx, 0) + 1
                    segment_car_total_len = 0
                    for key in occur_car:
                        if occur_car[key] >= 2 and key not in [RushHourPuzzle.BACKGROUND_IDX, car_idx]:
                            segment_car_total_len += occur_car[key]
                    if vertical_len - segment_car_total_len >= car_len:
                        results.append((car_idx, top_left_cell_row, top_left_cell_col, Direction.DOWN,
                                        touch_row - top_left_cell_row + 1, car_len))
    else:
        if orientation == 2:
            # Up
            up_len = touch_row
            if up_len >= car_len:
                occur_car = {}
                for ii in range(0, touch_row):
                    other_car_idx = rhp.board[ii][touch_col]
                    occur_car[other_car_idx] = occur_car.get(other_car_idx, 0) + 1
                segment_car_total_len = 0
                for key in occur_car:
                    if occur_car[key] >= 2 and key not in [RushHourPuzzle.BACKGROUND_IDX, car_idx]:
                        segment_car_total_len += occur_car[key]
                if up_len - segment_car_total_len >= car_len:
                    results.append((car_idx, top_left_cell_row, top_left_cell_col, Direction.UP,
                                    top_left_cell_row + car_len - touch_row, car_len))
            # Down
            down_len = N_ROWS - touch_row - 1
            if down_len >= car_len:
                occur_car = {}
                for ii in range(touch_row + 1, N_ROWS):
                    other_car_idx = rhp.board[ii][touch_col]
                    occur_car[other_car_idx] = occur_car.get(other_car_idx, 0) + 1
                segment_car_total_len = 0
                for key in occur_car:
                    if occur_car[key] >= 2 and key not in [RushHourPuzzle.BACKGROUND_IDX, car_idx]:
                        segment_car_total_len += occur_car[key]
                if down_len - segment_car_total_len >= car_len:
                    results.append((car_idx, top_left_cell_row, top_left_cell_col, Direction.DOWN,
                                    touch_row - top_left_cell_row + 1, car_len))
        else:
            # Horizontal
            if touch_direction == Direction.LEFT:
                horizontal_len = touch_col
                if horizontal_len >= car_len:
                    occur_car = {}
                    for jj in range(0, touch_col):
                        other_car_idx = rhp.board[touch_row][jj]
                        occur_car[other_car_idx] = occur_car.get(other_car_idx, 0) + 1
                    segment_car_total_len = 0
                    for key in occur_car:
                        if occur_car[key] >= 2 and key not in [RushHourPuzzle.BACKGROUND_IDX, car_idx]:
                            segment_car_total_len += occur_car[key]
                    if horizontal_len - segment_car_total_len >= car_len:
                        results.append((car_idx, top_left_cell_row, top_left_cell_col, Direction.LEFT,
                                        top_left_cell_col + car_len - touch_col, car_len))
            else:
                horizontal_len = N_COLS - touch_col - 1
                if horizontal_len >= car_len:
                    occur_car = {}
                    for jj in range(touch_col + 1, N_COLS):
                        other_car_idx = rhp.board[touch_row][jj]
                        occur_car[other_car_idx] = occur_car.get(other_car_idx, 0) + 1
                    segment_car_total_len = 0
                    for key in occur_car:
                        if occur_car[key] >= 2 and key not in [RushHourPuzzle.BACKGROUND_IDX, car_idx]:
                            segment_car_total_len += occur_car[key]
                    if horizontal_len - segment_car_total_len >= car_len:
                        results.append((car_idx, top_left_cell_row, top_left_cell_col, Direction.RIGHT,
                                        touch_col - top_left_cell_col + 1, car_len))
    return results


def move_car(rhp: RushHourPuzzle,
             car_idx: int,
             top_left_cell_row: int,
             top_left_cell_col: int,
             direction: DirectionT,
             car_len: int,
             move_len: int,
             expected_sets: List[Set[str]]) -> List[Set[str]]:
    hello = random.randint(1, 9999999)
    print(hello, 'GET:', expected_sets)
    # Add to set
    encoded = encode_car_idx_move_direction(car_idx=car_idx, move_direction=direction)
    all_repeated_so_far = True
    for expected_set in expected_sets:
        if encoded not in expected_set:
            expected_set.add(encoded)
            all_repeated_so_far = False
    if all_repeated_so_far:
        return []
    # Check if touching others
    all_sets: List[Set[str]] = []
    for m in range(1, move_len + 1):
        #
        if direction == Direction.UP:
            touch_row = top_left_cell_row - m
            touch_col = top_left_cell_col
        elif direction == Direction.RIGHT:
            touch_row = top_left_cell_row
            touch_col = top_left_cell_col + car_len - 1 + m
        elif direction == Direction.DOWN:
            touch_row = top_left_cell_row + car_len - 1 + m
            touch_col = top_left_cell_col
        else:
            touch_row = top_left_cell_row
            touch_col = top_left_cell_col - m
        if not (0 <= touch_row < N_ROWS) or not (0 <= touch_col < N_COLS):
            continue
        #
        car_moving = touch_car(rhp=rhp, touch_direction=direction, touch_row=touch_row, touch_col=touch_col)
        #
        print('**********', len(car_moving))
        for data in car_moving:
            ret = move_car(rhp=rhp,
                           car_idx=data[0],
                           top_left_cell_row=data[1],
                           top_left_cell_col=data[2],
                           direction=data[3],
                           car_len=data[5],
                           move_len=data[4],
                           expected_sets=copy.deepcopy(expected_sets))
            all_sets += ret
    rtn = all_sets if len(all_sets) > 0 else expected_sets
    print(hello, 'RETURN:', rtn)
    return rtn


class HeuristicFunc:
    @staticmethod
    def n_directly_block(rhp: RushHourPuzzle) -> float:
        board = rhp.board
        num_of_not_empty_not_red_car = 0
        for i in range(6):
            if board[2][5 - i] == 0:
                break
            if board[2][5 - i] != rhp.BACKGROUND_IDX:
                num_of_not_empty_not_red_car += 1
        return float(num_of_not_empty_not_red_car)

    @staticmethod
    def h2(rhp: RushHourPuzzle) -> float:
        ret = find_all_blocking_cars_and_its_suitable_move(rhp=rhp)
        print(ret)

        expected_sets: List[Set[str]] = [set()]
        result_sets: List[Set[str]] = []
        for item in ret:
            car_idx, top_left_cell_row, top_left_cell_col, car_len, suitable_moves = item
            for move in suitable_moves:
                direction, move_len = move
                result_sets += move_car(rhp=rhp,
                                        car_idx=car_idx,
                                        top_left_cell_row=top_left_cell_row,
                                        top_left_cell_col=top_left_cell_col,
                                        direction=direction,
                                        car_len=car_len,
                                        move_len=move_len,
                                        expected_sets=expected_sets)
        print(result_sets)
        # TODO:
        return float(min([len(result_set) for result_set in result_sets]))


if __name__ == '__main__':
    import pathlib, yaml

    config = yaml.safe_load(pathlib.Path('rush_hour_puzzle/config.yaml').read_text())
    puzzle = RushHourPuzzle(config=config)

    eg_input = '''
0 2 1 2 1
1 0 1 2 1
2 3 3 3 1
3 5 2 3 1
4 0 0 3 2
5 3 2 2 2
6 0 3 3 2
7 4 5 2 2
'''
    puzzle.transform_car_info_into_board(eg_input)
    puzzle.show_board()

    print(HeuristicFunc.h2(puzzle))
