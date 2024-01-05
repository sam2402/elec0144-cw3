# A* and Dijkstra algorithms combined implementation

from typing import Iterator, List, Set, Tuple
import math
import random


# Defines the grid to be 20x20 in size
GRID_SIZE = 20


def zero(position: int, target: int) -> float:
    # Dijkstra's algorithm is the same as using A* algorithm
    # with a heuristic function that always returns 0
    return 0.0


def euclidean_distance(position_1: int, position_2: int) -> float:
    # Returns the Euclidean distance between two positions
    # after turning the positions into (row, column) coordinates

    row_1 = (position_1 - 1) // GRID_SIZE
    column_1 = (position_1 - 1) % GRID_SIZE

    row_2 = (position_2 - 1) // GRID_SIZE
    column_2 = (position_2 - 1) % GRID_SIZE

    # Calculate the Euclidean distance
    return math.sqrt((row_1 - row_2) ** 2 + (column_1 - column_2) ** 2)


def inside_grid(row: int, column: int) -> bool:
    # Returns whether the position is inside the grid
    return 0 <= row < GRID_SIZE and 0 <= column < GRID_SIZE


def neighbours(position: int, target: int, obstacles: Set[int], heuristic) -> Iterator[int]:
    # Returns the neighbours of the current grid position which can be moved to

    # Turn the position into a (row, column) coordinate
    # to make it easier to find the neighbours
    # (0, 0) is the top left corner of the grid
    row = (position - 1) // GRID_SIZE
    column = (position - 1) % GRID_SIZE

    # Try all eight directions from the current cell
    for row_delta in [-1, 0, 1]:
        for column_delta in [-1, 0, 1]:
            # Skip the position we are already in
            if row_delta == 0 and column_delta == 0:
                continue

            new_row = row + row_delta
            new_column = column + column_delta

            # Skip the position if it is outside the grid
            if not inside_grid(new_row, new_column):
                continue

            # Convert the (row, column) coordinate back into a position
            new_position = GRID_SIZE * new_row + new_column + 1

            # Skip the position if there is an obstacle there
            if new_position in obstacles:
                continue

            # Calculate the cost of moving to the new position
            # Equal to 1 for straight moves and sqrt(2) for diagonal moves
            g = math.sqrt(row_delta * row_delta +
                          column_delta * column_delta)

            h = heuristic(new_position, target)

            # Return the new position and the cost of moving to it
            yield new_position, g + h, g


def a_star(start: int,  target: int, obstacles: Set[int], heuristic):
    # Create the front set and visited set
    front = {(heuristic(start, target), 0, start, None)}
    visited = {}

    front_log = [front.copy()]
    visited_log = [visited.copy()]

    # Repeat while the front set is not empty
    while front:
        # Find the node in the front set with the minimum cost
        node = min(front, key=lambda n: n[0])
        f, g, position, previous = node

        # Add the node to the visited set and remove it from the front set
        visited[position] = (f, g, previous)
        front.remove(node)

        # Optimisation: stop if we have chosen the target position
        # This is not necessary for Dijkstra's algorithm to work, however
        # it prevents us from exploring the rest of the grid unnecessarily
        if position == target:
            front_log.append(front.copy())
            visited_log.append(visited.copy())
            break

        # For every neighbouring position in the grid
        for neighbour, edge_f, edge_g in neighbours(position, target, obstacles, heuristic):
            neighbour_node = (g + edge_f, g + edge_g, neighbour, position)

            # Skip the neighbour if it has already been visited
            if neighbour in visited:
                continue

            # Add the neighbour node to the front set if it is not already there
            # Or, adjust the cost if the new cost is lower
            add_to_front_set = True

            for front_node in list(front):
                # Skip nodes in the front set which do not refer to
                # the same position as the neighbour we are interested in
                if front_node[2] != neighbour_node[2]:
                    continue

                if front_node[0] <= neighbour_node[0]:
                    # The neighbour is already in the front set and the new cost is not lower
                    add_to_front_set = False
                else:
                    # The new cost is lower
                    front.remove(front_node)

                break

            # Add the neighbour to the front set if it wasn't already
            # there, or if it was there but the new cost is lower
            if add_to_front_set:
                front.add(neighbour_node)

        # Save the evolution of the front set and visited set
        front_log.append(front.copy())
        visited_log.append(visited.copy())

    # Print the cost and reconstruct the path
    print("Cost:", visited[target][1])

    path = []
    while target:
        # Repeatedly find the previous position of the target
        # until we reconstruct the entire from from the start
        path.insert(0, target)
        if target in visited:
            target = visited[target][2]
        else:
            target = None

    print("Path:", path)
    return path, visited


def make_table(start, target, obstacles, path, visited):
    # This function generates the LaTeX code for the grid showing
    # the result of the pathfinding process. It uses the arguments
    # to highlight the start, target, obstacles, and visited positions
    # in different colours.

    lines = []
    lines.append("\\begin{center}")
    lines.append("\\vspace{1em}")
    lines.append(
        "\\begin{tabular}" + "{|" + "|".join(list("c" * GRID_SIZE)) + "|}")
    lines.append("  \\hline")

    for line in range(GRID_SIZE):
        l = []

        for i in range(1, GRID_SIZE + 1):
            n = GRID_SIZE * line + i

            if n in obstacles:
                l.append("\\cellcolor{gray!35}")
            elif n in path:
                l.append("\\cellcolor{blue!25}")
            elif n in visited:
                l.append("\\cellcolor{blue!10}")
            else:
                l.append("")

            if n == start:
                l[-1] += "$\\triangleright$"
            elif n == target:
                l[-1] += "$\\oplus$"

        line = " & ".join(l) + " \\\ \\hline"
        lines.append("  " + line)

    lines.append("\\end{tabular}")
    lines.append("\\captionof{figure}{PUT CAPTION HERE}")
    lines.append("\\end{center}")

    return "\n".join(lines)


# Runs the pathfinding with both Dijkstra's algorithm and the A* algorithm

NUM_OBSTACLES = 200

# Generate a random set of obstacle positions
obstacles = set(random.randint(1, GRID_SIZE * GRID_SIZE)
                for _ in range(NUM_OBSTACLES))

start = None
target = None

# Generate a random start and target position, making
# sure they are not at positions with obstacles
while start is None or target is None or start in obstacles or target in obstacles:
    start = random.randint(1, GRID_SIZE * GRID_SIZE)
    target = random.randint(1, GRID_SIZE * GRID_SIZE)


# Dijkstra
# (Using the a_star function with a heuristic that always returns 0
# is equivalent to Dijkstra's algorithm)
print("\nDijkstra's Algorithm:")
path, visited = a_star(start, target, obstacles, heuristic=zero)
print("Dijkstra explored", len(visited), "positions")

dijkstra_visited = len(visited)

# Save the LaTeX code for the grid to a file
with open("dijkstra_large_table.txt", "w") as file:
    table = make_table(start, target, obstacles, path, visited)
    file.write(table)

# A*
print("\nA* Algorithm:")
path, visited = a_star(start, target, obstacles, heuristic=euclidean_distance)
print("A* explored", len(visited), "positions")

a_star_visited = len(visited)

# Save the LaTeX code for the grid to a file
with open("a_star_large_table.txt", "w") as file:
    table = make_table(start, target, obstacles, path, visited)
    file.write(table)


# Compare the number of positions explored by Dijkstra's algorithm and the A* algorithm
visited_percentage_difference = (
    dijkstra_visited - a_star_visited) / dijkstra_visited * 100
print("\nA* explored", str(round(visited_percentage_difference, 2)) + "%",
      "fewer positions than Dijkstra's algorithm")
