#Hunter Zhao 
#with reference to https://www.samyzaf.com/ML/rl/qmaze.html

import matplotlib.pyplot as plt 
import numpy as np
import random
#from random import randrange

maze = ([
    [1., 1., 1., 1., 1.],
    [1., 0., 1., 1., 1.],
    [1., 1., 1., 1., 1.],
    [1., 1., 1., 0., 1.],
    [1., 1., 1., 1., 1.]
    ])
visited_cell = 0.8 #track visited cells
rat_mark = 0.5 #track current rat cell
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
#actions
actions = {
    UP: "up",
    RIGHT: "right",
    DOWN: "down",
    LEFT: "left",
    }
epsilon = 0.1

class Qmaze(object):
    def __init__(self, maze, rat=(0,0)):
        self.maze1 = np.array(maze)
        max_rows,max_cols = self.maze1.shape    #gets diemensions of maze
        #print(max_rows,max_cols)
        self.goal = (max_rows - 1,max_cols - 1) #where the goal is
        self.create_table()
        self.start(rat)

    def create_table(self):
        max_rows,max_cols = self.maze1.shape
        rows = max_rows * max_cols
        cols = 4
        self.q_table = [[0 for i in range(cols)] for j in range(rows)]  #creates a q table with rows the size of the maze and cols with actions
        #print(self.q_table)

    def get_empty(self, max_rows,max_cols, maze1):
        array = []
        for r in range(max_rows):
            for c in range(max_cols):
                if maze1[r,c] == 1.0:
                    cell = (r,c)
                    array.append(cell)
        return array

    def start(self, rat):
        self.rat = rat
        self.maze = np.copy(self.maze1)
        row, col = rat
        #print(row, col)
        self.maze[row,col] = visited_cell   #mark starting cell as visited
        self.state = (row, col, "start")    #tracks the current state of the rat
        self.min_reward = -1 * self.maze.size #the minimum reward before restart
        self.total_reward = 0 #current overall reward
        self.visited = set()    #visited cells

    def update(self, action):
        nrow, ncol, nmode = self.state  #get current state of rat
        nrows, ncols = self.maze.shape  #get maze shape

        if self.maze[nrow, ncol] > 0.0:
            self.visited.add((nrow, ncol))  # mark visited cell

        valid_actions = self.get_actions(nrow, ncol, nrows, ncols)  #get valid actions

        if not valid_actions:
            nmode = "empty"
        elif action in valid_actions:
            nmode = "valid"
            if action == UP:
                nrow -= 1
            elif action == RIGHT:
                ncol += 1
            if action == DOWN:
                nrow += 1
            elif action == LEFT:
                ncol -= 1
        else:   #action not recognized
            nmode = "invalid"
        
        #new state after action
        self.state = (nrow, ncol, nmode)

    def get_actions(self, row, col, nrows, ncols):
        actions = [0, 1, 2, 3]
        if row == 0:
            actions.remove(0)
        elif row == nrows - 1:
            actions.remove(2)

        if col == 0:
            actions.remove(3)
        elif col == ncols - 1:
            actions.remove(1)

        if row > 0 and self.maze[row - 1, col] == 0.0:
            actions.remove(0)
        if row<nrows-1 and self.maze[row + 1, col] == 0.0:
            actions.remove(2)

        if col > 0 and self.maze[row, col - 1] == 0.0:
            actions.remove(3)
        if col<ncols - 1 and self.maze[row, col + 1] == 0.0:
            actions.remove(1)

        return actions

    def get_reward(self):
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows-1 and rat_col == ncols-1:
            return 1.0
        if mode == "empty":
            return self.min_reward
        if mode == "invalid":
            return -7.5
        if (rat_row, rat_col) in self.visited:
            return -2.5
        if mode == "valid":
            return -0.04

    def game_status(self):
        if self.total_reward <= self.min_reward:
            return "loss"
        rat_row, rat_col, _ = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows - 1 and rat_col == ncols - 1:
            return "won"
        return "continue"

    def observe(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r,c] > 0.0:
                    canvas[r,c] = 1.0
        # draw the rat
        row, col, _ = self.state
        canvas[row, col] = rat_mark
        envstate = canvas.reshape((1, -1))
        return envstate

    def move(self, action):
        self.update(action)
        self.reward = self.get_reward()  #gets reward of action
        self.total_reward += self.reward    #adds reward of action to overall reward
        status = self.game_status() #get game status
        envstate = self.observe()   #get maze state
        return envstate, self.reward, status

    #goes through all actions to check for highest qvalue and then output that action
    def get_move(self, row, col, max_rows, max_cols):
        greatest_reward = -99999
        greatest_action = []
        output = 0
        for x in range(4):
            current_q_value = self.q_table[(row * max_rows) + col][x]
            if(current_q_value > greatest_reward):
                greatest_action = [x]
                output = greatest_action[0]
                greatest_reward = current_q_value
            elif(current_q_value == greatest_reward):
                greatest_action.append(x)
        #print(len(greatest_action))
        if(len(greatest_action) > 1):
            chosen = random.randrange(len(greatest_action))
            output = greatest_action[chosen]
        return output

    #goes through a set of valid actions in order to choose action with highest qvalue
    def rec_get_move(self, row, col, max_rows, max_cols):
        valid = self.get_actions(row, col, max_rows, max_cols)
        greatest_reward = -99999
        greatest_action = []
        output = 0
        for x in valid:
            current_q_value = self.q_table[(row * max_rows) + col][x]
            if(current_q_value > greatest_reward):
                greatest_action = [x]
                output = greatest_action[0]
                greatest_reward = current_q_value
            elif(current_q_value == greatest_reward):
                greatest_action.append(x)
        #print(len(greatest_action))
        if(len(greatest_action) > 1):
            chosen = random.randrange(len(greatest_action))
            output = greatest_action[chosen]
        return output

    #recursively calls itself to find the max reward based off of n
    def getFuture(self, cur_row, cur_col, tot_row, tot_col, n):
        discount_factor = 0.95
        if n == 1:
            #print((cur_row * tot_row) + cur_col)
            return max(self.q_table[(cur_row * tot_row) + cur_col]) * discount_factor
        elif n > 1:
            best = self.rec_get_move(cur_row, cur_col, tot_row, tot_col)
            if best == 0:
                return max(self.q_table[(cur_row * tot_row) + cur_col][0], self.getFuture(cur_row - 1, cur_col, tot_row, tot_col, n-1)) * discount_factor
            elif best == 1:
                return max(self.q_table[(cur_row * tot_row) + cur_col][1], self.getFuture(cur_row, cur_col + 1, tot_row, tot_col, n-1)) * discount_factor
            elif best == 2:
                return max(self.q_table[(cur_row * tot_row) + cur_col][2], self.getFuture(cur_row + 1, cur_col, tot_row, tot_col, n-1)) * discount_factor
            elif best == 3:
                return max(self.q_table[(cur_row * tot_row) + cur_col][3], self.getFuture(cur_row, cur_col - 1, tot_row, tot_col, n-1)) * discount_factor
    
    #updates qtable with the q learning algorithim
    def qUpdate(self, prev_row, prev_col, new_move):
        max_rows,max_cols = self.maze1.shape
        rat_row, rat_col, _ = self.state
        old = self.q_table[(prev_row * max_rows) + prev_col][new_move]
        learning_rate = .5
        reward = self.reward
        discount_factor = 0.95
        future_value = self.getFuture(rat_row, rat_col, max_rows, max_cols, n)
        #print("new move", new_move)
        self.q_table[(prev_row * max_rows) + prev_col][new_move] = old + learning_rate * (reward + future_value - old)
        #print(self.q_table)

def show(new_maze):
    plt.ion()
    nrows, ncols = new_maze.maze.shape
    canvas = np.copy(new_maze.maze)
    for row, col in new_maze.visited:
        canvas[row, col] = 0.6
    rat_row, rat_col, _ = new_maze.state
    canvas[rat_row, rat_col] = 0.3   # rat cell
    canvas[nrows-1, ncols-1] = 0.9 # cheese cell
    plt.imshow(canvas, cmap = "gray")
    #plt.show()
    #plt.pause(.001)

def play_game(qmaze):
    while(1):   #runs forever until stopped
        qmaze.start((0,0))  #restarts the maze with rat at the coordinates
        a = 1   #keep track of game status
        while(a):
            show(new_maze)
            max_rows, max_cols = qmaze.maze1.shape  #maze shape
            prev_row, prev_col, _ = qmaze.state     #rat state
            if(epsilon > random.random()):
                new_move = random.randrange(4)  #epsilon exploration
            else:
                new_move = qmaze.get_move(prev_row, prev_col, max_rows, max_cols)   #gets next move based on current qtable values
            canvas, reward, game_over = qmaze.move(new_move)    #tracks maze status
            qmaze.qUpdate(prev_row, prev_col, new_move)     #updates qtable
            print(qmaze.q_table)
            print("\n")
            if(game_over == "loss" or game_over == "won"):
                a = 0

n = input("Enter n")
n = int(n)
new_maze = Qmaze(maze)
play_game(new_maze)
#show(new_maze)
#canvas, reward, game_over = new_maze.move(UP)
#print(new_maze.state)
#print("reward=", reward)
#print("canvas=", canvas)
#show(new_maze)
