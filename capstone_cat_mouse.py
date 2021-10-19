#Hunter Zhao 

import sys
import matplotlib.pyplot as plt 
import numpy as np
import numpy as gfg
import random
#from random import randrange

maze = ([
    [0., 0., 0., 0., 0.],
    [0., -1., 0., 0., 0.],
    [0., 0., 3., 0., 0.],
    [0., 0., 0., -1., 0.],
    [0., 0., 0., 0., 0.]
    ])
#visited_cell_m = 0.8 #track visited cells
rat_mark = 2 #track current rat cell
#visited_cell_c = 
cat_mark = 1
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
    def __init__(self, maze):
        self.maze1 = np.array(maze)
        max_rows,max_cols = self.maze1.shape    #gets diemensions of maze
        #print(max_rows,max_cols)
        self.goal = (max_rows - 1,max_cols - 1) #where the goal is
        #self.empty_cells = self.get_empty(max_rows, max_cols, self.maze1)   #get all empty cells
        #self.empty_cells.remove(self.goal)  #remove goal from empty cell array
        #print(self.empty_cells)
        #self.create_table()
        self.start()

    def create_table(self):
        max_rows,max_cols = self.maze1.shape
        rows = max_rows * max_cols
        cols = 4
        self.m_q_table = np.zeros((rows, rows, cols))
        #print(self.m_q_table)
        self.c_q_table = np.zeros((rows, rows, cols))  #creates a 3d array with 25x25x4 as q table

    def get_empty(self, max_rows,max_cols, maze1):
        array = []
        for r in range(max_rows):
            for c in range(max_cols):
                if maze1[r,c] == 0.0:
                    cell = (r,c)
                    array.append(cell)
        return array

    def start(self):
        #self.rat = rat
        #self.cat = cat
        self.maze = np.copy(self.maze1)
        #rrow, rcol = rat
        #crow, ccol = cat
        #print(row, col)
        #self.maze[row,col] = visited_cell_m   #mark starting cell as visited
        #self.mstate = (rrow, rcol, "start")    #tracks the current state of the rat
        #self.cstate = (crow, ccol, "start")     #tracks the current start of the cat
        #self.min_reward = -1 * self.maze.size #the minimum reward before restart
        #self.m_total_reward = 0 #current mouse overall reward
        #self.c_total_reward = 0 #current cat overall reward
        #self.visited = set()    #visited cells

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

        if row > 0 and self.maze[row - 1, col] == -1.0:
            actions.remove(0)
        if row<nrows-1 and self.maze[row + 1, col] == -1.0:
            actions.remove(2)

        if col > 0 and self.maze[row, col - 1] == -1.0:
            actions.remove(3)
        if col<ncols - 1 and self.maze[row, col + 1] == -1.0:
            actions.remove(1)

        return actions

    def game_status(self, mouse, cat):
        #if self.total_reward <= self.min_reward:
        #    return "loss"
        rat_row, rat_col, _ = mouse.state
        cat_row, cat_col, _ = cat.state
        nrows, ncols = self.maze.shape
        global win
        global loss
        if rat_row == nrows - 1 and rat_col == ncols - 1 and mouse.cheese == 1:
            win = win + 1
            return "won"
        if rat_row == cat_row and rat_col == cat_col:
            loss = loss + 1
            return "loss"
        return "continue"

    #goes through all actions to check for highest qvalue and then output that action
    def get_move(self, m_row, m_col, max_rows, max_cols, c_row, c_col, agent):
        greatest_reward = -99999
        greatest_action = []
        output = 0
        for x in range(4):
            current_q_value = agent.q_table[(m_row * max_rows) + m_col][(c_row * max_rows) + c_col][x]
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
    
class Mouse:
    def __init__(self, name):
        self.name = name
        self.q_table = np.zeros((25, 25, 4))
        self.start()

    def start(self):
        self.state = (0, 0, "start")
        self.cheese = 0
        #self.total_reward = 0
        self.list = []

    def update(self, action, qmaze):
        nrow, ncol, nmode = self.state  #get current state of rat
        nrows, ncols = qmaze.maze.shape  #get maze shape

        valid_actions = qmaze.get_actions(nrow, ncol, nrows, ncols)  #get valid actions

        if not valid_actions:
            nmode = "empty"
        if nrow == 2 and ncol == 2 and self.cheese == 0:
            self.cheese = 1
            nmode = "cheese"
            qmaze.maze[2][2] == 0.
            if action == UP:
                nrow -= 1
            elif action == RIGHT:
                ncol += 1
            if action == DOWN:
                nrow += 1
            elif action == LEFT:
                ncol -= 1
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

    def get_reward(self, qmaze, cat):
        rat_row, rat_col, mode = self.state
        cat_row, cat_col, __ = cat.state
        nrows, ncols = qmaze.maze.shape
        if rat_row == cat_row and rat_col == cat_col:
            return -100
        if rat_row == nrows-1 and rat_col == ncols-1 and self.cheese == 1:
            return 100.0
        if mode == "empty":
            return -100
        if mode == "cheese":
            return 20.
        if mode == "invalid":
            return -7.5
        if mode == "valid":
            return -0.04
        #print("rip")

    #goes through a set of valid actions in order to choose action with highest qvalue
    def rec_get_move(self, row, col, max_rows, max_cols, cat_row, cat_col, qmaze):
        valid = qmaze.get_actions(row, col, max_rows, max_cols)
        greatest_reward = -99999
        greatest_action = []
        output = 0
        for x in valid:
            current_q_value = self.q_table[(row * max_rows) + col][(cat_row * max_rows) + cat_col][x]
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
    def getFuture(self, cur_row, cur_col, tot_row, tot_col, cat_row, cat_col, qmaze, n):
        if n == 1:
            #print((cur_row * tot_row) + cur_col)
            return max(self.q_table[(cur_row * tot_row) + cur_col][(cat_row * tot_row) + cat_col]) *.95
        elif n > 1:
            best = self.rec_get_move(cur_row, cur_col, tot_row, tot_col, cat_row, cat_col, qmaze)
            if best == 0:
                return max(self.q_table[(cur_row * tot_row) + cur_col][(cat_row * tot_row) + cat_col][0], self.getFuture(cur_row - 1, cur_col, tot_row, tot_col, cat_row, cat_col, qmaze, n-1)) *.95
            elif best == 1:
                return max(self.q_table[(cur_row * tot_row) + cur_col][(cat_row * tot_row) + cat_col][1], self.getFuture(cur_row, cur_col + 1, tot_row, tot_col, cat_row, cat_col, qmaze, n-1)) *.95
            elif best == 2:
                return max(self.q_table[(cur_row * tot_row) + cur_col][(cat_row * tot_row) + cat_col][2], self.getFuture(cur_row + 1, cur_col, tot_row, tot_col, cat_row, cat_col, qmaze, n-1)) *.95
            elif best == 3:
                return max(self.q_table[(cur_row * tot_row) + cur_col][(cat_row * tot_row) + cat_col][3], self.getFuture(cur_row, cur_col - 1, tot_row, tot_col, cat_row, cat_col, qmaze, n-1)) *.95

    #updates qtable with the q learning algorithim
    def qUpdate(self, prev_row, prev_col, new_move, cat, qmaze):
        max_rows,max_cols = qmaze.maze1.shape
        rat_row, rat_col, _ = self.state
        cat_row, cat_col, _ = cat.state
        old = self.q_table[(prev_row * max_rows) + prev_col][(cat_row * max_rows) + cat_col][new_move]
        learning_rate = .5
        reward = self.reward
        #discount_factor = 0.95
        future_value = self.getFuture(rat_row, rat_col, max_rows, max_cols, cat_row, cat_col, qmaze, n)
        #print("new move", new_move)
        self.q_table[(prev_row * max_rows) + prev_col][(cat_row * max_rows) + cat_col][new_move] = old + learning_rate * (reward + future_value - old)
        #print(self.q_table)

    def move(self, action, qmaze, cat):
        self.update(action, qmaze)
        self.reward = self.get_reward(qmaze, cat)  #gets reward of action
        self.list.append(action)
        #print(self.reward)
        #self.total_reward += self.reward    #adds reward of action to overall reward

class Cat:
    def __init__(self, name):
        self.name = name
        self.q_table = np.zeros((25, 25, 4))
        self.start()
    
    def start(self):
        self.state = (4, 4, "start")
        self.list = []

    def update(self, action, qmaze):
        nrow, ncol, nmode = self.state  #get current state of cat
        nrows, ncols = qmaze.maze.shape  #get maze shape

        valid_actions = qmaze.get_actions(nrow, ncol, nrows, ncols)  #get valid actions

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

    def get_reward(self, qmaze, mouse):
        cat_row, cat_col, mode = self.state
        rat_row, rat_col, __ = mouse.state
        nrows, ncols = qmaze.maze.shape
        if rat_row == cat_row and rat_col == cat_col:
            return 100
        if rat_row == nrows-1 and rat_col == ncols-1 and mouse.cheese == 1:
            return -100.0
        if mode == "empty":
            return -100
        if mode == "invalid":
            return -7.5
        if mode == "valid":
            return -0.04

    #goes through a set of valid actions in order to choose action with highest qvalue
    def rec_get_move(self, row, col, max_rows, max_cols, cat_row, cat_col, qmaze):
        valid = qmaze.get_actions(cat_row, cat_col, max_rows, max_cols)
        greatest_reward = -99999
        greatest_action = []
        output = 0
        for x in valid:
            current_q_value = self.q_table[(row * max_rows) + col][(cat_row * max_rows) + cat_col][x]
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
    def getFuture(self, rat_row, rat_col, tot_row, tot_col, cur_row, cur_col, qmaze, n):
        discount_factor = .95
        if n == 1:
            #print((cur_row * tot_row) + cur_col)
            return max(self.q_table[(rat_row * tot_row) + rat_col][(cur_row * tot_row) + cur_col]) * discount_factor
        elif n > 1:
            best = self.rec_get_move(rat_row, rat_col, tot_row, tot_col, cur_row, cur_col, qmaze)
            if best == 0:
                return max(self.q_table[(rat_row * tot_row) + rat_col][(cur_row * tot_row) + cur_col][0], self.getFuture(rat_row, rat_col, tot_row, tot_col, cur_row - 1, cur_col, qmaze, n-1)) * discount_factor
            elif best == 1:
                return max(self.q_table[(rat_row * tot_row) + rat_col][(cur_row * tot_row) + cur_col][1], self.getFuture(rat_row, rat_col, tot_row, tot_col, cur_row, cur_col + 1, qmaze, n-1)) * discount_factor
            elif best == 2:
                return max(self.q_table[(rat_row * tot_row) + rat_col][(cur_row * tot_row) + cur_col][2], self.getFuture(rat_row, rat_col, tot_row, tot_col, cur_row + 1, cur_col, qmaze, n-1)) * discount_factor
            elif best == 3:
                return max(self.q_table[(rat_row * tot_row) + rat_col][(cur_row * tot_row) + cur_col][3], self.getFuture(rat_row, rat_col, tot_row, tot_col, cur_row, cur_col - 1, qmaze, n-1)) * discount_factor

    #updates qtable with the q learning algorithim
    def qUpdate(self, prev_row, prev_col, new_move, mouse, qmaze):
        max_rows,max_cols = qmaze.maze1.shape
        rat_row, rat_col, _ = mouse.state
        cat_row, cat_col, _ = self.state
        old = self.q_table[(rat_row * max_rows) + rat_col][(prev_row * max_rows) + prev_col][new_move]
        learning_rate = .5
        reward = self.reward
        #discount_factor = 0.95
        future_value = self.getFuture(rat_row, rat_col, max_rows, max_cols, cat_row, cat_col, qmaze, n)
        #print("new move", new_move)
        self.q_table[(rat_row * max_rows) + rat_col][(prev_row * max_rows) + prev_col][new_move] = old + learning_rate * (reward + future_value - old)
        #print(self.q_table)

    def move(self, action, qmaze, cat):
        self.update(action, qmaze)
        self.reward = self.get_reward(qmaze, cat)  #gets reward of action
        self.list.append(action)
        #print(self.reward)
        #self.total_reward += self.reward    #adds reward of action to overall reward

def show(new_maze, mouse, cat):
    plt.ion()
    nrows, ncols = new_maze.maze.shape
    canvas = np.copy(new_maze.maze)
    rat_row, rat_col, _ = mouse.state
    cat_row, cat_col, _ = cat.state
    canvas[rat_row, rat_col] = -3   # rat cell
    canvas[cat_row, cat_col] = -2  # cat cell
    canvas[nrows-1, ncols-1] = 2 # exit cell
    plt.imshow(canvas)
    #plt.show()
    plt.pause(.1)

def play_game(qmaze, mouse, cat):
    #while(1):   #runs forever until stopped
    qmaze.start()  #restarts the maze with rat and cat at the coordinates
    mouse.start()
    cat.start()
    a = 1   #keep track of game status
    #b = random.randrange(10000)
    while(a):
        #if b == 500:
            #show(qmaze, mouse, cat)
            #epsilon = epsilon - .01
        max_rows, max_cols = qmaze.maze1.shape  #maze shape
        m_prev_row, m_prev_col, _ = mouse.state     #rat state
        c_prev_row, c_prev_col, _ = cat.state       #cat state
        if(epsilon > random.random()):
            new_move = random.randrange(4)  #epsilon exploration
        else:
            new_move = qmaze.get_move(m_prev_row, m_prev_col, max_rows, max_cols, c_prev_row,c_prev_col, mouse)   #gets next move based on current qtable values
        mouse.move(new_move, qmaze, cat)    #tracks maze status
        status = qmaze.game_status(mouse, cat) #get game status
        #envstate = qmaze.observe(mouse, cat)   #get maze state
        mouse.qUpdate(m_prev_row, m_prev_col, new_move, cat, qmaze)     #updates qtable
        #print(qmaze.q_table)
        #print("\n")
        if(status == "loss" or status == "won"):
            a = 0
        else:
            if(epsilon > random.random()):
                new_move = random.randrange(4)  #epsilon exploration
            else:
                new_move = qmaze.get_move(m_prev_row, m_prev_col, max_rows, max_cols, c_prev_row, c_prev_col, cat)   #gets next move based on current qtable values
            cat.move(new_move, qmaze, mouse)    #tracks maze status
            status = qmaze.game_status(mouse, cat) #get game status
            #envstate = qmaze.observe(mouse, cat)   #get maze state
            cat.qUpdate(c_prev_row, c_prev_col, new_move, mouse, qmaze)    #updates qtable
            if(status == "loss"):
                mouse.reward = -100
                mouse.qUpdate(m_prev_row, m_prev_col, new_move, cat, qmaze)
            #print(qmaze.q_table)
            #print("\n")
            if(status == "loss" or status == "won"):
                a = 0
        
    strm = "".join(str(e) for e in mouse.list)
    strc = "".join(str(e) for e in cat.list)
    #print(strm + " " + strc)
    return

n = 1
win = 1
loss = 1
games = 0
winsList = []
gamesList =[]

def main(args = None):
    args = sys.argv
    #n = int(n)
    new_maze = Qmaze(maze)
    mouse = Mouse("mouse")
    cat = Cat("cat")
    global games
    while(games < 100000):
        play_game(new_maze, mouse, cat)
        games = games + 1
        if((games % 1000) == 0):
            winsList.append(win/(games))
            gamesList.append(games)
    plt.plot(gamesList, winsList)
    plt.title('Mouse Winrate Vs Total Games')
    plt.xlabel('Games')
    plt.ylabel('Mouse Winrate')
    plt.show()        

if __name__ == "__main__":
    main()

#show(new_maze)
#canvas, reward, game_over = new_maze.move(UP)
#print(new_maze.state)
#print("reward=", reward)
#print("canvas=", canvas)
#show(new_maze)
