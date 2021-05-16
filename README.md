# Rush Hour Puzzle Search Algorithms


<!--img src="Rush%20Hour%20Puzzle%20Search%20Algorithms%2064c5b59631d144b3af53e7de2dfb18ae/myplot.png" alt="rush_hour_puzzle" style="display: block;  margin-left:auto; margin-right:auto; max-width: 30px;"-->  
<img src="Rush%20Hour%20Puzzle%20Search%20Algorithms%2064c5b59631d144b3af53e7de2dfb18ae/myplot.png" alt="rush_hour_puzzle" height="150px" width="150px">  

## Problem Setting[1]

We will use a standard 6x6 board with a number of cars. The cells are identified using the
(row,column) coordinates, with (0,0) at the top-left corner. With the only exit at the right side of row 2 (3rd
row from top), the goal is to clear the way for the red car to exit.

The initial state of a game is given by the layout of the cars on the board. This will be given in plain text, with
each line representing a car and consisting of its index, top-left cell (row, column), length (2 or 3), and
orientation (1 for horizontal and 2 for vertical).

Goal state: Any state where the red car has its top-left cell at (row=2, column=4).
Solution: A sequence of car movements. Each action is specified by a tuple of <car_index, new_row,
new_column>. You can output your solution in plain text.

Heuristics: A common heuristic function for this puzzle is the "blocking heuristic", that is, the number of cars
directly blocking the way of the red car to the exit. Its value for the example board above is one.

## Experiments

- Algorithms: `BFS`, `DFS`, `IDS`, `A*`, and `IDA*`. Each has `graph` & `tree` search versions.
- Heuristic functions: `blocking cars` & `h2`. ("block cars" is the one provided in the assignment spec. "h2" is designed by me and will be introduced later.)
- The below experiments all use `L26.txt` as input puzzle.
- Note that due to layout restrictions, the fonts in the graphs may be too small, but you can zoom in the graphs to get better experience.

### Graph Search Version (All find a solution successfully.)

![method_result_table](Rush%20Hour%20Puzzle%20Search%20Algorithms%2064c5b59631d144b3af53e7de2dfb18ae/table1.png)

#### How the Number of Nodes Stored in the Container Changes with the Number of Explored

![](Rush%20Hour%20Puzzle%20Search%20Algorithms%2064c5b59631d144b3af53e7de2dfb18ae/12.png)
![](Rush%20Hour%20Puzzle%20Search%20Algorithms%2064c5b59631d144b3af53e7de2dfb18ae/11.png)
![](Rush%20Hour%20Puzzle%20Search%20Algorithms%2064c5b59631d144b3af53e7de2dfb18ae/13.png)

Although `IDS` takes much more time than `DFS`, its **space use** is significantly **smaller** than `DFS`'s. Max number of nodes in `IDS`'s container is 140, which is much lower than 2560 in `DFS`.

#### Why the Size of Explored Set Changes with the Number of Explored Nodes in IDS is very large?

Thanks for 彭敍溶's suggestion, he let me know that to get an **optimal** solution in IDS, I should also keep **depth** information in the explored set to prevent the following situation:

![](Rush%20Hour%20Puzzle%20Search%20Algorithms%2064c5b59631d144b3af53e7de2dfb18ae/flow1.png)

IDS Comparison(All find a solution successfully.)
![IDS Comparison(All find a solution successfully.)](Rush%20Hour%20Puzzle%20Search%20Algorithms%2064c5b59631d144b3af53e7de2dfb18ae/table2.png)

![](Rush%20Hour%20Puzzle%20Search%20Algorithms%2064c5b59631d144b3af53e7de2dfb18ae/14.png)

Therefore, iterative-deepening without keeping depth may be a compromise version between DFS and IDS because its time and space use are between those two.

#### Evaluation

- Number of expanded nodes:

    ![](Rush%20Hour%20Puzzle%20Search%20Algorithms%2064c5b59631d144b3af53e7de2dfb18ae/eq1.png)

- Time used:

    ![](Rush%20Hour%20Puzzle%20Search%20Algorithms%2064c5b59631d144b3af53e7de2dfb18ae/eq2.png)

### Tree search Version

In the tree search version, I restrict the number of expanded nodes to **1,000,000**, since enormous number of nodes will be expanded and cause running out of computer's memory. The experiment result is that **all** combinations **fail** to find the solution in the restriction of only 1,000,000 expanded nodes. (Tested algorithms: DFS, BFS, IDS, and A*, IDA* with two kinds of heuristic functions.)

## Design of Heuristic Function (`h2`)

In this heuristic function, I use backward method to calculate the total number of how many cars must be moved towards which direction if I want to move a blocking car to let red car pass through. Please refer to the codes if you're interesting its implementing details. The below is a simple example and the following is the simple description:

![Rush%20Hour%20Puzzle%20Search%20Algorithms%2064c5b59631d144b3af53e7de2dfb18ae/Untitled_Design.png](Rush%20Hour%20Puzzle%20Search%20Algorithms%2064c5b59631d144b3af53e7de2dfb18ae/Untitled_Design.png =250x250)

1. Find all blocking car, do the following steps to each of them. (Purple car)
2. Find positions to move this car to let red pass through. (Purple car should move down.)
3. Find what cars will be influenced. (Green car is touched.)
4. For each influenced car, do step 2 & 3 until terminated. Finally, count how many car need to be move towards what direction. (e.g. Yellow should be moved left. Move left ≠ move right.)

In this example, total need-move is 5 (those five arrows in the image). This heuristic function gives this state a value: 5.

#### Evaluation of h2

Speed: It's sometimes faster and sometimes slower than `blocking cars`. Heuristic function `h2`  in `IDA*` has fewer expanded nodes but takes more time than `blocking cars`. It may because calculate `h2`  function takes more time than that of `blocking cars`.

Space: It takes more space to stored the nodes in the container.

## Discussing

#### Why are tree search version algorithms difficult to find a solution?

If there are any **repeated** states, an infinitive loop may occur. Thus, depth-first search may not find a solution forever. As for bread-first algorithms, they can find a solution theoretically, but it takes lots of time to deal with repeated states.

I make an small experiment to see how large is tree search's expanding number. Limit the number of explored nodes to 500.  Input data is still `L26.txt`. Calculate all explored nodes' average number of expanded nodes. I got:

- BFS graph search: 1.22
- BFS graph search: 7.22

Besides time use, tree search take lots of space, too. The two graphs right show us that the space which the tree search needs rise up dramatically with the increasing of explored nodes.

![](Rush%20Hour%20Puzzle%20Search%20Algorithms%2064c5b59631d144b3af53e7de2dfb18ae/15.png)

#### How to implement explored set & is it worth it?

Hash table is good when the number of elements is small but we cannot promise it won't collision. Another data structure like red-black tree is also suitable. The insertion and finding existence take O(log n) time.

Use explored set is a wise choice in this game since there may be some repeated states and the branching factor is large. But, in some cases (like the IDS problem mentioned above), the space for explored set may be larger than the space for storing nodes in container.

## Reference and Source
1. The L26.txt is from course "AI Capstone" in NYCU taught by Tsaipei Wang. Problem setting and some rules are from this course homework project's spec.
