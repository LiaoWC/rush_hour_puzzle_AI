import yaml
import argparse
import os
import time
from pathlib import Path
from .rush_hour_puzzle import RushHourPuzzle, Action, Direction
from .search import SearchEngine, SearchStats, Algorithm, SearchVersion
import sys
import cv2
from typing import Tuple, Sequence, List
from numpy.typing import ArrayLike
from .heuristic import HeuristicFunc


def make_path_video(name: str, size: Tuple[int, int], cv2mats: Sequence[ArrayLike], fps: int = 4):
    video = cv2.VideoWriter(f'{name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for cv2mat in cv2mats:
        video.write(cv2mat)
    cv2.destroyAllWindows()
    video.release()


def get_heuristic_func_list() -> List:
    return [func for func in dir(HeuristicFunc) if callable(getattr(HeuristicFunc, func)) and '__' not in func]


if __name__ == '__main__':
    #
    print('Arguments:', sys.argv)

    # Parser
    default_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.yaml')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=default_config_path, help='Path to the config file.')
    algorithm_help = """
        Algorithm for solving the puzzle. 
        Available options: BFS, DFS, IDS, A_STAR, IDA_STAR.
        You can use number to indicate the algorithm you'd like to use:
          (1) for BFS,
          (2) for DFS,
          (3) for IDS,
          (4) for A_STAR,
          (5) for IDA_STAR
        """
    heuristic_func_help = 'Pick heuristic function. ("None" or "none" for not using.) Available heuristic functions: {}'.format(
        ', '.join(get_heuristic_func_list())) + '.'
    parser.add_argument('-a', '--algorithm', required=True, help=algorithm_help)
    parser.add_argument('-v', '--version', required=True, help='Choose tree version or graph version.')
    parser.add_argument('-f', '--input-file', help='Input board.', required=True)
    parser.add_argument('-d', '--max-depth', help='Max depth.', type=int, required=True)
    parser.add_argument('-e', '--max-explored', help='Max number of nodes explored.', type=int, required=True)
    parser.add_argument('-u', '--heuristic-func', help=heuristic_func_help, required=True)
    parser.add_argument('-m', '--make-video',
                        help='Choose whether making an video of path if finding the solution. Type "true", "yes", "t", or "y" to choose making an video.',
                        required=True)
    args = parser.parse_args()

    # Load config
    config = yaml.safe_load(Path(args.config).read_text())

    # Instantiate puzzle
    puzzle = RushHourPuzzle(config=config)

    # Choose algorithm
    algorithm = args.algorithm
    if algorithm in ['1', Algorithm.BFS]:
        algorithm = Algorithm.BFS
    elif algorithm in ['2', Algorithm.DFS]:
        algorithm = Algorithm.DFS
    elif algorithm in ['3', Algorithm.IDS]:
        algorithm = Algorithm.IDS
    elif algorithm in ['4', Algorithm.A_STAR]:
        algorithm = Algorithm.A_STAR
    elif algorithm in ['5', Algorithm.IDA_STAR]:
        algorithm = Algorithm.IDA_STAR
    else:
        raise ValueError('Algorithm "{}" is not valid.'.format(algorithm))

    # Choose version (graph or tree)
    version = args.version
    if version not in ['tree', 'graph']:
        raise ValueError('Version "{}" is not valid.'.format(version))

    # Choose heuristic func
    if args.heuristic_func and args.heuristic_func not in ['None', 'none']:
        heuristic_func = getattr(HeuristicFunc, args.heuristic_func)
    else:
        heuristic_func = None

    # Choose max depth
    max_depth = args.max_depth

    # Choose max number of explored nodes
    max_n_explored = args.max_explored

    # Choose whether make an video if finding the solution
    if args.make_video in ['true', 't', 'yes', 'y']:
        make_video = True
    else:
        make_video = False

    # eg_input = '''
    #             0 2 3 2 1
    #             1 0 0 3 1
    #             2 1 3 3 1
    #             3 3 1 2 1
    #             4 5 0 3 1
    #             5 1 0 2 2
    #             6 3 0 2 2
    #             7 1 2 2 2
    #             8 3 3 3 2
    #             9 4 4 2 2
    #             10 2 5 2 2
    #             11 4 5 2 2'''
    # puzzle.transform_car_info_into_board(eg_input)

    # Read input file and set board
    puzzle.transform_car_info_into_board(Path(args.input_file).read_text())

    start_time = time.time()

    make_video = True
    algorithm = Algorithm.A_STAR
    max_depth = 99999
    version = SearchVersion.GRAPH
    heuristic_func = HeuristicFunc.n_directly_block

    #
    search_engine = SearchEngine(config=config,
                                 algorithm=algorithm,
                                 version=version,
                                 source_puzzle=puzzle,
                                 heuristic_fun=heuristic_func,
                                 max_depth=max_depth,
                                 max_n_explored=max_n_explored)

    search_stats: SearchStats = search_engine.run()

    # stat: SearchStats = search(config=config,
    #                            algorithm='BFS',
    #                            version='graph',
    #                            source_board=a,
    #                            heuristic_fun=None,
    #                            max_depth=1000,
    #                            max_n_explored=8000000)
    # path_found = search_with_heuristic(source_board=a, max_depth=20, max_n_visited=100000, heuristic_fun=jjj)
    #
    print("--- {:.3f} seconds for searching---".format(time.time() - start_time))

    print('Whether found solution:', search_stats.sol_found)
    if search_stats.sol_found:
        # data = stat.path
        path = search_stats.path

        if len(path) > 0:
            print('=== Solution ===')
            print('car_index, new_row, new_col')
            for state in path[1:]:
                move_direction = state.last_action.move_direction
                move_len = state.last_action.move_len
                car_idx = state.last_action.car_idx
                #
                new_row = state.last_action.top_left_cell_row
                new_col = state.last_action.top_left_cell_col
                if move_direction == Direction.UP:
                    new_row -= move_len
                elif move_direction == Direction.DOWN:
                    new_row += move_len
                elif move_direction == Direction.RIGHT:
                    new_col += move_len
                elif move_direction == Direction.LEFT:
                    new_col -= move_len
                else:
                    raise ValueError('Invalid move_direction: {}'.format(move_direction))
                #
                print(car_idx, new_row, new_col)

        print('Output solution done!')

        if make_video:
            start_time = time.time()
            img_to_be_video = []
            for state in path:
                img_to_be_video.append(state.show_board(show_actions=True, show_img=False, return_cv2mat=True))
            print('--- {} seconds for preparing for the video ---'.format(time.time() - start_time))
            video_name = input('Video name:')
            make_path_video(video_name, (img_to_be_video[0].shape[0], img_to_be_video[0].shape[1]), img_to_be_video,
                            fps=4)

    print('Number of nodes expanded:', search_stats.n_expand)
    print('Length of explored:', len(search_stats.explored))

    print('Done!')

    # 1. S -> all legal actions
    # 2. S ____> S' after do an action

    # TODO: receive car info from stdout "<" symbol
    # Prepare for making a video
