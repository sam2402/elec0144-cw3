from dataclasses import dataclass
from math import exp
import random
import copy

# GLOBAL VARIABLES

# Possible actions and the corresponding change in the row and column
# Eg. to go up, subtract one from the row and do nothing to the column
DIRECTIONS = {
    "up":    (-1, 0),
    "right": (0, 1),
    "down":  (1, 0),
    "left":  (0, -1),
}

LIVING_REWARD = 1
DISCOUNT_FACTOR = 0.775
LEARNING_RATE = 0.15

@dataclass
class Cell:
    '''
    Class representing each cell in the grid
    It stores whether the agent can visit it (an obstacle or not), whether it is a terminal state and what the reward is for visiting it

    Below is an example of how to create and use a cell that is visitable, non-terminal and has a reward of 3 for visiting it:
    >>> cell1 = Cell(True, False, 3)
    >>> cell1.visitable
    True
    >>> cell1.terminal
    False
    >>> cell1.reward
    3
    '''
    visitable: bool
    terminal: bool
    reward: int

# TYPES

# The Q-table is stored as a 2D-list of dictionaries.
# The keys of the dict are "up", "right", "down", "left", and the values are the Q-values for the respective action
TableType = list[list[dict[str, float]]]

# The grid is stored as a 2D-list of Cell objects
GridType = list[list[Cell]]

Number = float | int

def clamp(x: Number , low: Number, high: Number) -> Number:
    '''
    Clamps x to be between low and high.
    If x is greater than high, high is returned
    If x is less than low, low is returned
    If x is between low and high, x is returned
    '''
    return max(min(x, high), low)

def e_greedy(t: int, rate: float | None, e_min: float | None) -> float:
    '''
    Epsilon function as used in the epsilon-greedy method
    Its value is constrained to be between 1 and e_min.
    It exponentially decays as t increases
    '''
    _rate = 0.016 if rate is None else rate
    _e_min = 0.7 if e_min is None else e_min
    return max(exp(-_rate*t), _e_min)

def has_converged(curr_table: TableType, new_table: TableType, precision: int = 4) -> bool:
    '''
    Returns True if the curr_table has the same Q-values for all cells and actions as new_table
    Otherwise returns False
    '''

    # Iterate over all pairs of cells in curr_table and new_table
    for curr_row, new_row in zip(curr_table, new_table):
        for curr_cell, new_cell in zip(curr_row, new_row):
            for direction in DIRECTIONS: # If the cells don't have the same value to 'precision' decimal places for each of the actions return False
                if round(curr_cell[direction], precision) != round(new_cell[direction], precision):
                    return False
    return True

def calculate_new_position(grid: GridType, pos: (int, int), direction: str) -> (int, int):
    '''
    Returns the new position co-ordinates given the underlying grid, current position and direction to move in
    '''
    
    # Clamping the new row and column co-ordinates ensures they are between 0 and the number rows/columns respectively 
    new_row = clamp(pos[0] + DIRECTIONS[direction][0], 0, len(grid)-1)
    new_col = clamp(pos[1] + DIRECTIONS[direction][1], 0, len(grid[0])-1)
    
    # Return the new position if it is visitable otherwise return the old position
    return (new_row, new_col) if grid[new_row][new_col].visitable else pos

def get_best_action_grid(table: TableType, grid: GridType) -> list[str]:
    '''
    Returns a viewable grid where each cell shows the action with the highest Q value
    If the cell is an obstacle or terminal then '*' is displayed.
    Cells on the same row are separated by tabs
    Below is an example output:
    [
        "right   right   right   *",
        "up      *       up      *",
        "right   right   up      left"
    ]
    '''

    action_grid = [] 

    # Iterate over all pairs of cells in Q-table and grid (containing data on each cell)
    for table_row, grid_row in zip(table, grid):
        row = ""
        for table_cell, grid_cell in zip(table_row, grid_row):
            # For each cell, append the action with the highest Q value to the row 
            # table_cell is a dictionary where the keys are actions and the values are Q-values 
            # max(table_cell, key=table_cell.get) returns the key in table_cell with the highest value
            # If the cell is an obstacle or terminal then '*' is appended
            row += max(table_cell, key=table_cell.get) if (grid_cell.visitable and not grid_cell.terminal) else "*"
            row += "\t"
        action_grid.append(row[:-1]) # Remove the final '\t' and append the remainder of the row to the action_grid list
    return action_grid

def get_viewable_table(table: TableType) -> list[str]:
    '''
    Returns a human readable version of the Q-table
    Each row of the table shows the cell number, along with each action's Q-values for that cell
    Below is an example output:

    cell        up          right       down        left
    1           -0.5449     3.1216      -0.0337     0.3667
    2           1.194       4.5799      1.4589      0.8747
    3           6.2         0.4924      3.6936      2.2394
    4           -6.8619     -0.5069     -0.297      3.0324
    ...
    '''

    viewable_table = []
    cell_number = 1

    # Cells are numbered 1 to n with 1 being in the bottom left and n being at the top right
    # This requires the rows to be iterated over in reverse order from bottom to top
    for row in table[::-1]:
        for cell in row:

            # Get the Q-value of each action rounded to 4 decimal places and join them together with tabs
            # Prepend to that the cell number and a tab
            viewable_row = "\t".join([str(cell_number), *(str(round(cell[direction], 4)) for direction in DIRECTIONS)])
            viewable_table.append(viewable_row.expandtabs(12)) # Set the length of a tab to be 12 spaces to provide sufficient padding
            cell_number += 1
    return viewable_table
            

def run_episode(grid: GridType, curr_table: TableType, t: int, rate: float = None, e_min: float = None) -> TableType:
    '''
    Run a learning episode on the grid, with curr_table as initial Q-table and as the epoch number
    Return the new Q-table after running the Q-learning algorithm until a terminal state is reached.
    '''
    
    pos = (len(grid)-1, 0) # The starting state is always the bottom left cell
    new_table = copy.deepcopy(curr_table) # curr_table is a compound object so a deepcopy must be created to avoid reusing stale objects
    
    # Perform value iterations to solve the MDP until a terminal state is reached
    while True:
        a = random.random() # A random number between 0 and 1

        # A random action is taken if a < e(t) or all Q-values for the current state are equal
        # Otherwise the action with the greatest Q-value is used
        direction = random.choice(list(DIRECTIONS)) if a < e_greedy(t, rate, e_min) or len(set(new_table[pos[0]][pos[1]].values())) == 1 else \
                    max(new_table[pos[0]][pos[1]], key=new_table[pos[0]][pos[1]].get)
        new_row, new_col = calculate_new_position(grid, pos, direction)
        new_cell = grid[new_row][new_col]

        # Calculate the reward for the action by subtracting the living reward from the cell reward if the new state is not terminal
        reward = new_cell.reward - (LIVING_REWARD if not new_cell.terminal else 0)

        # curr_q is the current Q-value for the current state and chosen action
        curr_q = new_table[pos[0]][pos[1]][direction]

        # sample_q is the reward of going from the current state to the next state plus the discounted Q-value of the best action when in the next state
        sample_q = reward + (DISCOUNT_FACTOR*max(new_table[new_row][new_col].values()))

        # Adjust the Q-value by the learning rate multiplied by the difference between the sample and current Q-value
        new_table[pos[0]][pos[1]][direction] += LEARNING_RATE*(sample_q-curr_q)

        # If the next state is terminal then return the current Q-table
        # Otherwise set the current state to be the new state and perform another iteration
        if new_cell.terminal == True:
            return new_table
        pos = (new_row, new_col)

def run_q_learning(grid: GridType, rate = None, e_min = None) -> (list[str], list[str]):
    '''
    Given a 2D array of Cell objects (each of which contains data on its reward and whether it is terminal or an obstacle),
    Run the tabular Q-learning algorithm and return:
        - The resultant Q-table showing the Q-values of all the state-action pair after convergence
        - A grid showing the best action for each cell
    '''
    # The Q-table is represented as a 2D list that has the same form as the grid-world.
    # Each cell is a dictionary that contains a key for each action - the value for each action is is the Q-value for taking that action
    curr_table = [[{action: 0 for action in DIRECTIONS} for _ in row] for row in grid]
    new_table = None
    t = 0

    # While at least one Q-value has changed between the previous and current iteration, run an episode of Q-learning
    # Increment the value of t for each iteration so the e-greedy function decays
    while True:
        new_table = run_episode(grid, curr_table, t, rate, e_min)
        if has_converged(curr_table, new_table):
            # When all values in the Q-table have converged, return a human readable version of the Q-table
            # and a grid where each cell shows the action with the highest Q value
            return get_viewable_table(new_table), get_best_action_grid(new_table, grid)
        curr_table = new_table
        t += 1

    
if __name__ == "__main__":
    table, action_grid = run_q_learning([
        [Cell(True, False, 0), Cell(True,  False, 0), Cell(True, False, 0), Cell(True, True,  10)],
        [Cell(True, False, 0), Cell(False, False, 0), Cell(True, False, 0), Cell(True, True, -10)],
        [Cell(True, False, 0), Cell(True,  False, 0), Cell(True, False, 0), Cell(True, False, 0) ],
    ])
    
    # Output the best action grid
    print("___________________________________________________________")
    print("BEST ACTIONS")
    print("If a cell is an obstacle or terminal then '*' is displayed")
    print("")
    print(*action_grid, sep="\n")
    print("")

    # Output the Q-table
    print("___________________________________________________________")
    print("Q-TABLE")
    print("")
    print(("\t".join(["cell", *DIRECTIONS])).expandtabs(12))
    print(*table, sep="\n")

    '''
    Uncomment the code below to produce a graph showing the correctness of the algorithm for varying e_min and and decay rate values

    WARNING: Executing this code will take ~45 minutes
    '''
    # import matplotlib.pyplot as plt
    # import numpy as np
    # from matplotlib import cm
    # from matplotlib.ticker import LinearLocator

    # RUNS = 100
    # rates = np.arange(0.005, 0.02, 0.001)
    # e_mins = np.arange(0.05, 0.8, 0.02)
    # X = []
    # Y = []
    # correctness = []
    # i = 0
    # for rate in rates:
    #     print("Total runs: ", i)
    #     for e_min in e_mins:
    #         correct = 0
    #         for _ in range(RUNS):
    #             i += 1
    #             table, action_grid = run_q_learning([
    #                 [Cell(True, False, 0), Cell(True,  False, 0), Cell(True, False, 0), Cell(True, True,  10)],
    #                 [Cell(True, False, 0), Cell(False, False, 0), Cell(True, False, 0), Cell(True, True, -10)],
    #                 [Cell(True, False, 0), Cell(True,  False, 0), Cell(True, False, 0), Cell(True, False, 0) ],
    #             ])
    #             if action_grid[0] ==  "right\tright\tright\t*" and \
    #             action_grid[1] ==  "up\t*\tup\t*" and \
    #             (action_grid[2] == "up\tright\tup\tleft" or action_grid[2] == "right\tright\tup\tleft"):
    #                 correct += 1
    #         X.append(rate)
    #         Y.append(e_min)
    #         correctness.append((correct*100)/RUNS)

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Z = correctness

    # surf = ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm_r,
    #                     linewidth=0, antialiased=False)

    # ax.set_zlim(88, 100)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter('{x:.02f}')

    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # ax.set_xlabel("Rate")
    # ax.set_ylabel("ε min")
    # ax.set_zlabel("Percentage correct")

    # plt.show()

    '''
    Uncomment the code below to produce a graph showing the correctness of the algorithm for varying learning rates

    WARNING: Executing this code will take ~1 minute
    '''
    # import matplotlib.pyplot as plt
    # import numpy as np

    # RUNS = 100
    # rates = np.arange(0.05, 0.9, 0.05)
    # correctness = []
    # i = 0
    # for rate in rates:
    #     print("Total runs: ", i)
    #     correct = 0
    #     LEARNING_RATE = rate
    #     for _ in range(RUNS):
    #         i += 1
    #         table, action_grid = run_q_learning([
    #             [Cell(True, False, 0), Cell(True,  False, 0), Cell(True, False, 0), Cell(True, True,  10)],
    #             [Cell(True, False, 0), Cell(False, False, 0), Cell(True, False, 0), Cell(True, True, -10)],
    #             [Cell(True, False, 0), Cell(True,  False, 0), Cell(True, False, 0), Cell(True, False, 0) ],
    #         ])
    #         if action_grid[0] ==  "right\tright\tright\t*" and \
    #         action_grid[1] ==  "up\t*\tup\t*" and \
    #         (action_grid[2] == "up\tright\tup\tleft" or action_grid[2] == "right\tright\tup\tleft"):
    #             correct += 1
    #     correctness.append((correct*100)/RUNS)

    # trend = np.polyfit(rates, correctness, 5)

    # plt.plot(rates, correctness, 'o')

    # trendpoly = np.poly1d(trend) 

    # plt.plot(rates, trendpoly(rates))

    # plt.xlabel("Learning Rate")
    # plt.ylabel("Percentage Correct")

    # plt.show()

    '''
    Uncomment the code below to produce a graph showing the correctness of the algorithm for varying discount factors

    WARNING: Executing this code will take ~1 minute
    '''
    # import matplotlib.pyplot as plt
    # import numpy as np

    # RUNS = 100
    # discount_factors = np.arange(0.5, 0.99, 0.025)
    # correctness = []
    # i = 0
    # for discount_factor in discount_factors:
    #     print("Total runs: ", i)
    #     correct = 0
    #     DISCOUNT_FACTOR = discount_factor
    #     for _ in range(RUNS):
    #         i += 1
    #         table, action_grid = run_q_learning([
    #             [Cell(True, False, 0), Cell(True,  False, 0), Cell(True, False, 0), Cell(True, True,  10)],
    #             [Cell(True, False, 0), Cell(False, False, 0), Cell(True, False, 0), Cell(True, True, -10)],
    #             [Cell(True, False, 0), Cell(True,  False, 0), Cell(True, False, 0), Cell(True, False, 0) ],
    #         ])
    #         if action_grid[0] ==  "right\tright\tright\t*" and \
    #         action_grid[1] ==  "up\t*\tup\t*" and \
    #         (action_grid[2] == "up\tright\tup\tleft" or action_grid[2] == "right\tright\tup\tleft"):
    #             correct += 1
    #     correctness.append((correct*100)/RUNS)

    # trend = np.polyfit(discount_factors, correctness, 5)

    # plt.plot(discount_factors, correctness, 'o')

    # trendpoly = np.poly1d(trend) 

    # plt.plot(discount_factors, trendpoly(discount_factors))

    # plt.xlabel("Discount Factor")
    # plt.ylabel("Percentage Correct")

    # plt.show()
