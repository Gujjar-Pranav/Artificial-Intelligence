# Import necessary library for IDA* Algorithms
import random
import time
import csv
from random import shuffle
from copy import deepcopy

class Node:
    def __init__(self, state, parent=None, action=None, depth=0, actual_cost=0, heuristic=0):
        # Initialize the Node with the following:
        # given state
        # parent node in the search tree,
        # action taken to reach this node,
        # depth of the node in the search tree.

        self.state = state
        self.parent = parent
        self.action = action
        self.depth = depth
        self.actual_cost = actual_cost  # g(n): actual cost to reach this node
        self.heuristic = heuristic  # h(n): heuristic estimate cost to reach the goal from current node
        self.total_cost = self.actual_cost + self.heuristic  # f(n) = g(n) + h(n), f-score is total cost

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

# Function for Template state by shuffling the given state
def template_state_generator():

    state = list(reversed(range(9)))
    shuffle(state)
    matrix = []
    for i in range(0, 9, 3):
        matrix.append(state[i:i + 3])  # Convert the template state into a matrix (3x3)

    blank_tile_pos_row = None
    blank_tile_pos_col = None
    for i, row in enumerate(matrix):
        if 0 in row:
            blank_tile_pos_row = i
            blank_tile_pos_col = row.index(0)
            break

    return [blank_tile_pos_row, blank_tile_pos_col, matrix]

#Function to generate child nodes by moving the blank tile in various directions
def get_children(node, goal_state):
    children = []
    x, y, matrix = node.state[0], node.state[1], node.state[2]
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Blank tile moving directions: right, left, down, up
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(matrix) and 0 <= ny < len(matrix[0]):
            new_state = deepcopy(node.state)  # Create a deepcopy to avoid modifying the original state
            new_state[0], new_state[1] = nx, ny
            new_state[2] = [row[:] for row in matrix]  # Deep copy of the matrix
            new_state[2][x][y], new_state[2][nx][ny] = new_state[2][nx][ny], new_state[2][x][y]  # Swap tiles
            new_node = Node(new_state, node, (dx, dy), node.depth + 1)  # Create a new node with the updated state
            new_node.heuristic = heuristic(new_state, goal_state)  # Calculate heuristic for the new node
            new_node.actual_cost = new_node.depth  # Increment the actual cost by 1
            new_node.total_cost = new_node.actual_cost + new_node.heuristic  # Calculate the total cost
            children.append(new_node)  # Add the new node to the list of children
    return children

# Function for calculate heuristic value (manhattan distance) for the given state
def heuristic(state, goal_state):
    goal_positions = {}
    for i, row in enumerate(goal_state[2]):
        for j, num in enumerate(row):
            goal_positions[num] = (i, j)  # Store the position of each number in the goal state

    total_dist = 0
    for i, row in enumerate(state[2]):
        for j, num in enumerate(row):
            if num != 0:  # Exclude the blank tile (0)
                goal_pos = goal_positions[num]
                total_dist += abs(i - goal_pos[0]) + abs(j - goal_pos[1])  # Calculate Manhattan distance

    return total_dist

# Function to perform IDA* until the max. threshold is reached.
def ida_star(root, goal, max_threshold):
    visited = set()
    nodes_opened = 0
    for threshold in range(max_threshold + 1):
        result, nodes = depth_bound_search_path(root, goal, threshold, visited, 0)
        nodes_opened += nodes
        if result is not None:
            return result, nodes_opened
        visited.clear()  # Clear visited set for the next iteration
    return None, nodes_opened

#Function to perform search path with deth bound using ida* algorithm
def depth_bound_search_path(node, goal, threshold, visited, nodes_opened):
    nodes_opened += 1
    if node.total_cost > threshold:
        return None, nodes_opened
    visited.add(node)
    if node.state == goal:
        return node, nodes_opened
    for child in get_children(node, goal):
        if child not in visited:
            result, nodes_opened = depth_bound_search_path(child, goal, threshold, visited, nodes_opened)
            if result is not None:
                return result, nodes_opened
    return None, nodes_opened

# Function to solve the sliding puzzle problem using ida* algorithm.
def solve_puzzle(start_state, goal_state):
    root = Node(start_state)
    root.heuristic = heuristic(start_state, goal_state)
    root.actual_cost = root.depth
    root.total_cost = root.actual_cost + root.heuristic
    max_threshold = 30
    start_time = time.time()
    solution, nodes_opened = ida_star(root, goal_state, max_threshold)  # Apply IDA* algorithm to find the solution
    end_time = time.time()

    moves = solution.depth if solution else max_threshold
    computing_time = end_time - start_time
    return start_state, solution is not None, moves, nodes_opened, computing_time

# Function to save the results of solving puzzle instances into a csv file.
def save_results_to_csv(results, seed, filename="IDASTAR_output.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header row for CSV file
        writer.writerow(["Seed", "Case_Id", "Start_State", "Solution_Found", "Num_Moves", "Num_Nodes_Opened", "Computing_Time"])
        # Write data rows for each puzzle instance
        for case_num, (start_state, solution_found, num_moves, num_nodes_opened, computing_time) in results:
            start_state_str = str(start_state)
            solution_found_int = 1 if solution_found else 0
            writer.writerow([seed, case_num, start_state_str, solution_found_int, num_moves, num_nodes_opened, computing_time])

# Random Seed , Initial state, Goal state , Start state details as below
seed = 924
random.seed(seed)
state = list(reversed(range(9)))
goal_state = [1, 1, [[1, 2, 3], [8, 0, 4], [7, 6, 5]]]
start_states = [template_state_generator() for _ in range(10)]

# Solve puzzles and collect results
results = [(i, solve_puzzle(start_state, goal_state)) for i, start_state in enumerate(start_states, 1)]
save_results_to_csv(results, seed)
