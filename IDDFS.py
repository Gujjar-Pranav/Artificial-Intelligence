# Import necessary modules
import random
import time
import csv
from copy import deepcopy
from random import shuffle

# Define the Node class for representing a node in the search tree
class Node:
    def __init__(self, state, parent=None, action=None, depth=0):
        """
        Initialize a node with the given state, parent node, action, and depth.
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = depth

    def __eq__(self, other):
        """
        Check if two nodes are equal by comparing their states.
        """
        return self.state == other.state

    def __hash__(self):
        """
        Generate a hash value for a node based on its state.
        """
        return hash(tuple(tuple(row) for row in self.state[2]))

# Function to generate a template state by shuffling the given state for the puzzle
def template_state_generator():

    state = list(reversed(range(9)))
    shuffle(state)
    matrix = []
    for i in range(0, 9, 3):
        matrix.append(state[i:i + 3])

    # Find the position of the blank tile (0)
    blank_tile_pos_row = None
    blank_tile_pos_col = None
    for i, row in enumerate(matrix):
        if 0 in row:
            blank_tile_pos_row = i
            blank_tile_pos_col = row.index(0)
            break

    return [blank_tile_pos_row, blank_tile_pos_col, matrix]

# Function to generate child nodes by moving the blank tile in different directions
def get_children(node):
    children = []
    x, y = node.state[0], node.state[1]  # Blank tile position
    blnk_tile_move_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Direction of blank tile: Up, Down, Left, Right
    node_state = node.state  # Store node state to avoid repeated access
    for dx, dy in blnk_tile_move_directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(node_state[2]) and 0 <= ny < len(node_state[2][0]):
            new_state = deepcopy(node_state)
            new_state[2][x][y], new_state[2][nx][ny] = new_state[2][nx][ny], new_state[2][x][y]
            new_state[0], new_state[1] = nx, ny
            children.append(Node(new_state, node, (dx, dy), node.depth + 1))
    return children

# Function to perform Iterative Deepening Depth-First Search (IDDFS) till the maz. depth is reached.
def iddfs(root, goal, max_depth):
    nodes_opened = 0  # Counter for opened nodes
    visited = set()  # Initialize visited set outside the loop
    for depth in range(max_depth + 1):
        result, nodes_opened = dls(root, goal, depth, visited, nodes_opened)
        if result is not None:
            return result, nodes_opened
        visited.clear()  # Clear visited set for the next iteration
    return None, nodes_opened

# Function to perform Depth-Limited Search (DLS) up to a certain depth
def dls(node, goal, depth, visited, nodes_opened):
    if node.depth > depth:
        return None, nodes_opened
    nodes_opened += 1
    state_hash = tuple(map(tuple, node.state[2]))  # Convert state to hashable form
    visited.add(state_hash)
    if node.state == goal:
        return node, nodes_opened
    for child in get_children(node):
        child_state_hash = tuple(map(tuple, child.state[2]))  # Convert child state to hashable form
        if child_state_hash not in visited:
            result, nodes_opened = dls(child, goal, depth, visited, nodes_opened)
            if result is not None:
                return result, nodes_opened
    return None, nodes_opened

# Function to solve the puzzle using IDDFS and return relevant information
def solve_puzzle(start_state, goal_state):
    max_depth = 30
    root = Node(start_state)  # Create a root node with the initial state
    start_time = time.time()
    # Perform IDDFS to find the solution
    solution, nodes_opened = iddfs(root, goal_state, max_depth)
    end_time = time.time()

    # Determine if a solution is found and calculate the number of moves
    found_solution = bool(solution)
    moves = solution.depth if solution else max_depth
    computing_time = end_time - start_time

    return start_state, found_solution, moves, nodes_opened, computing_time

# Function to save the results of solving puzzle instances into a CSV file
def save_results_to_csv(results, filename="IDDFS_output.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the header row for the CSV file
        writer.writerow(["seed", "Case_Id", "Start_State", "Solution_Found", "Num_Moves", "Num_Nodes_Opened", "Computing_Time"])
        # Write the results for each puzzle instance into the CSV file
        for case_num, (start_state, solution_found, num_moves, num_nodes_opened, computing_time) in results:
            start_state_str = str(start_state)
            solution_found_int = int(solution_found)
            # Write the results for the current puzzle instance into the CSV file
            writer.writerow([seed, case_num, start_state_str, solution_found_int, num_moves, num_nodes_opened, computing_time])

# Random seed , Initial state, Goal state, Start state details
seed = 924  # Student ID
random.seed(seed)
state = list(reversed(range(9)))
goal_state = [1, 1, [[1, 2, 3], [8, 0, 4], [7, 6, 5]]]
# Generate 10 random start states for the puzzle
start_states = [template_state_generator() for _ in range(10)]

# Solve each puzzle instance and collect the results
results = [(i, solve_puzzle(start_state, goal_state)) for i, start_state in enumerate(start_states, 1)]
save_results_to_csv(results)
