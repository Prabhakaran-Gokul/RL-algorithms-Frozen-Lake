from Parameters import * #import parameters
import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.n_row, self.n_col = None, None
        self.states = None 
        self.curr_row, self.curr_col = 0 , 0
        self.actions = [LEFT, DOWN, RIGHT, UP]
        self.map = None
        self.build_environment()
        print("Environment initialised with {} X {} map\n".format(self.n_row, self.n_col))

    def build_environment(self):
        if self.grid_size == 4:
            raw_map = MAPS["4x4"]
        elif self.grid_size == 10:
            raw_map = MAPS["10x10"]
        else:
            print("raise exception here")
        
        self.map = np.asarray(raw_map, dtype="c")
        self.n_row, self.n_col = self.map.shape
        self.states = list(range(self.n_row * self.n_col))

    #function to convert row and col to state
    def convert_row_col_to_state(self, row, col):
        return row * self.n_col + col

    def convert_state_to_row_col(self, state):
        return (state // self.n_row, state % self.n_row)

    def get_reward(self, letter):
        if letter == b"G":
            return 1
        elif letter == b"H":
            return -1
        else:
            return 0

    def is_episode_done(self, new_letter):
        if new_letter == b"G" or new_letter == b"H":
            return True
        return False

    def step(self, action):
        if action == LEFT:
            self.curr_col = max(self.curr_col - 1, 0)
        elif action == DOWN:
            self.curr_row = min(self.curr_row + 1, self.n_row - 1)
        elif action == RIGHT:
             self.curr_col = min(self.curr_col + 1, self.n_col - 1)     
        elif action == UP:
            self.curr_row = max(self.curr_row - 1, 0)
        
        new_state = self.convert_row_col_to_state(self.curr_row, self.curr_col)
        new_letter = self.map[self.curr_row, self.curr_col]
        reward = self.get_reward(new_letter)
        done = self.is_episode_done(new_letter)
        return (new_state, reward, done)

    def reset(self):
        self.curr_row = 0
        self.curr_col = 0
        return self.convert_row_col_to_state(self.curr_row, self.curr_col)

    def render(self, path):
        blue = [0, 255, 255]
        black = [0, 0 , 0]
        green = [0, 255, 0]
        red = [255, 0, 0]
        yellow = [255, 255, 51]
        map_list =  [[[state.decode("utf-8")] for state in row] for row in self.map]
        for row in range(len(map_list)):
            for col in range(len(map_list)):
                if map_list[row][col][0] == "F":
                    map_list[row][col] = blue
                elif map_list[row][col][0] == "H":
                    map_list[row][col] = black
                elif map_list[row][col][0] == "S":
                    map_list[row][col] = green
                elif map_list[row][col][0] == "G":
                    map_list[row][col] = red  
        
        for state in path:
            row, col = self.convert_state_to_row_col(state)
            map_list[row][col] = yellow

        map_array = np.asarray(map_list, dtype=np.int32)
        fig = plt.figure(figsize=(8,8))
        #TODO add in the arrows
        plt.arrow(0, 0, 1, 0, length_includes_head = True, head_width=0.2, head_length=0.2)
        plt.imshow(map_array)
        plt.show()