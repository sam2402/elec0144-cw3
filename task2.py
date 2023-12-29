# Dijkstra's Algorithm

from typing import Iterator, List, Set, Tuple
import math


# Defines the grid to be 6x6 in size
GRID_SIZE = 6


def inside_grid(row: int, column: int) -> bool:
    # Returns whether the position is inside the grid
    return 0 <= row < GRID_SIZE and 0 <= column < GRID_SIZE


def neighbours(position: int, obstacles: Set[int]) -> Iterator[int]:
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
            cost = math.sqrt(row_delta * row_delta +
                             column_delta * column_delta)

            # Return the new position and the cost of moving to it
            yield new_position, cost


def save_front_set_and_visited_log(front_log: List, visited_log: List, filename: str):
    # Save the evolution of the front set and visited set to a txt file
    with open(filename, "w") as file:
        for i in range(len(front_log)):
            file.write(f"Iteration {i}:\n")
            file.write(f"Front set: {front_log[i]}\n")

            visited_set = set((cost, position, previous)
                              for position, (cost, previous) in visited_log[i].items())
            file.write(f"Visited set: {visited_set}\n\n")

    print("Saved the evolution of the front set and visited set to", filename)


def dijkstra(start: int,  target: int, obstacles: Set[int], filename: str):
    # Create the front set and visited set
    front = {(0, start, None)}
    visited = {}

    front_log = [front.copy()]
    visited_log = [visited.copy()]

    # Repeat while the front set is not empty
    while front:
        # Find the node in the front set with the minimum cost
        node = min(front, key=lambda n: n[0])
        cost, position, previous = node

        # Add the node to the visited set and remove it from the front set
        visited[position] = (cost, previous)
        front.remove(node)

        # Optimisation: stop if we have chosen the target position
        # This is not necessary for Dijkstra's algorithm to work, however
        # it prevents us from exploring the rest of the grid unnecessarily
        if position == target:
            front_log.append(front.copy())
            visited_log.append(visited.copy())
            break

        # For every neighbouring position in the grid
        for neighbour, edge_cost in neighbours(position, obstacles):
            neighbour_node = (cost + edge_cost, neighbour, position)

            # Skip the neighbour if it has already been visited
            if neighbour in visited:
                continue

            # Add the neighbour node to the front set if it is not already there
            # Or, adjust the cost if the new cost is lower
            add_to_front_set = True

            for front_node in list(front):
                # Skip nodes in the front set which do not refer to
                # the same position as the neighbour we are interested in
                if front_node[1] != neighbour_node[1]:
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
    print("Cost:", visited[target][0])

    path = []
    while target:
        # Repeatedly find the previous position of the target
        # until we reconstruct the entire from from the start
        path.insert(0, target)
        if target in visited:
            target = visited[target][1]
        else:
            target = None

    print("Path:", path)
    save_front_set_and_visited_log(front_log, visited_log, filename)


# Run Dijkstra's algorithm on the 6x6 grid
obstacles = {2, 10, 11, 20, 21, 27}
start = 5
target = 32

print("Running Dijkstra's algorithm on the 6x6 grid")
dijkstra(start, target, obstacles, "dijkstra_1.txt")

# Add the new obstacle and run Dijkstra's algorithm again
obstacles.add(19)
print("\nRunning Dijkstra's algorithm on the 6x6 grid with an extra obstacle at position 19")
dijkstra(start, target, obstacles, "dijkstra_2.txt")
