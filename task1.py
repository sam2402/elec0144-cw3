from dataclasses import dataclass
from math import exp
import random
import copy

DIRECTIONS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
LIVING_REWARD = 1
DISCOUNT_FACTOR = 0.95
LEARNING_RATE = 0.1

@dataclass
class Cell:
    visitable: bool
    terminal: bool
    reward: int

TableType = list[list[dict[str, float]]]
GridType = list[list[Cell]]

def e_greedy(t):
    e_start = 1
    e_min = 0.3
    rate = 0.05
    return max(e_start*exp(-rate*t), e_min)

def has_converged(curr_table: TableType, new_table: TableType):
    if new_table is None:
        return False
    
    for curr_row, new_row in zip(curr_table, new_table):
        for curr_cell, new_cell in zip(curr_row, new_row):
            for direction in DIRECTIONS.keys():
                if round(curr_cell[direction], 4) != round(new_cell[direction], 4):
                    return False
    return True

def get_new_position(grid: GridType, pos: tuple[int, int], direction):
    new_row = max(min(
        pos[0] + DIRECTIONS[direction][0],
        len(grid)-1
    ), 0)
    new_col = max(min(
        pos[1] + DIRECTIONS[direction][1],
        len(grid[0])-1
    ), 0)
    return (new_row, new_col) if grid[new_row][new_col].visitable else pos

def get_best_action_grid(table: TableType, grid: GridType) -> list[str]:
    action_grid = []
    for table_row, grid_row in zip(table, grid):
        row = ""
        for table_cell, grid_cell in zip(table_row, grid_row):
            row += max(table_cell, key=table_cell.get) if (grid_cell.visitable and not grid_cell.terminal) else "*"
            row += "\t"
        action_grid.append(row)
    return action_grid

def get_viewable_table(table: TableType) -> list[str]:
    viewable_table = []
    cell_index = 1
    for row in table[::-1]:
        for cell in row:
            viewable_table.append(f"{cell_index}\t{round(cell['up'], 4)}\t{round(cell['right'], 4)}\t{round(cell['down'], 4)}\t{round(cell['left'], 4)}".expandtabs(12))
            cell_index += 1
    return viewable_table
            

def run_episode(grid: GridType, curr_table: TableType, t: int) -> TableType:
    pos = (len(grid)-1, 0)
    new_table = copy.deepcopy(curr_table)
    while True:
        a = random.random()
        direction = random.choice(list(DIRECTIONS.keys())) if a < e_greedy(t) or len(set(curr_table[pos[0]][pos[1]].values())) == 1 else \
                    max(curr_table[pos[0]][pos[1]], key=curr_table[pos[0]][pos[1]].get)
        new_pos = get_new_position(grid, pos, direction)
        new_cell = grid[new_pos[0]][new_pos[1]]
        reward = new_cell.reward - (LIVING_REWARD if not new_cell.terminal else 0)
        sample_q_i = reward + (DISCOUNT_FACTOR*max(curr_table[new_pos[0]][new_pos[1]].values()))
        prev_q_i = curr_table[pos[0]][pos[1]][direction]
        q_i_next = prev_q_i + LEARNING_RATE*(sample_q_i-prev_q_i)
        new_table[pos[0]][pos[1]][direction] = q_i_next
        if new_cell.terminal == True:
            return new_table
        pos = new_pos

def run_q_learning(grid) -> tuple[list[str], list[list[str]]]:
    curr_table = [[{"up":0, "right":0, "down":0, "left":0} for _ in row] for row in grid]
    new_table = None
    t = 0

    while True:
        new_table = run_episode(grid, curr_table, t)
        if has_converged(curr_table, new_table):
            print(t)
            return get_viewable_table(new_table), get_best_action_grid(new_table, grid)
        curr_table = new_table
        t += 1
    
if __name__ == "__main__":
    table, action_grid = run_q_learning([
        [Cell(True, False, 0), Cell(True, False, 0),  Cell(True, False, 0), Cell(True, True, 10) ],
        [Cell(True, False, 0), Cell(False, False, 0), Cell(True, False, 0), Cell(True, False, -10)],
        [Cell(True, False, 0), Cell(True, False, 0),  Cell(True, False, 0), Cell(True, False, 0)],
    ])
    print("BEST ACTIONS")
    print(*action_grid, sep="\n")
    print("")

    print("TABLE")
    print(("\t".join(["cell", *DIRECTIONS.keys()])).expandtabs(12))
    print(*table, sep="\n")