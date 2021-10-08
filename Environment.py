from Parameters import * 
import numpy as np
import matplotlib.pyplot as plt

class Environment:
    """
    Environment class is used to represent a Frozen Lake
    """
    def __init__(self, grid_size):
        """
        Initialises the environment with the given grid size

        Parameters:
        ----------
        grid_size: int
            The size of the grid. It is assumed that only a square grid is required.
        """

        self.grid_size = grid_size
        self.n_row, self.n_col = None, None
        self.states = None 
        self.curr_row, self.curr_col = 0 , 0
        self.actions = [LEFT, DOWN, RIGHT, UP]
        self.map = None
        self.build_environment()
        print("Environment initialised with {} X {} map\n".format(self.n_row, self.n_col))

    def build_environment(self):
        """
        Builds the forzen lake environment and populates the Environment class attributes
        """
        if self.grid_size == 4:
            raw_map = MAPS["4x4"]
        elif self.grid_size == 10:
            raw_map = MAPS["10x10"]
        else:
            print("Invalid Map")
        
        #Converting the input map into an numpy array and extracting information(number of states, shape of the map)
        self.map = np.asarray(raw_map, dtype="c")
        self.n_row, self.n_col = self.map.shape
        self.states = list(range(self.n_row * self.n_col))

    #function to convert row and col to state
    def convert_row_col_to_state(self, row, col):
        """
        Converts row and column to the specific state in the map

        Parameters:
        ----------
        row: int
            The row of the map 
        col: int 
            The column of the map
        """
        return row * self.n_col + col

    def convert_state_to_row_col(self, state):
        """
        Converts state of a map to row and column 
        
        Parameters:
        ----------
        state: int 
            The state in that map that needs to be convert into row and column 
        """
        return (state // self.n_row, state % self.n_row)

    def get_reward(self, letter):
        """
        Returns the Reward depending on the letter provided 
        
        Parameters:
        ----------
        letter: bytes
            Each letter corresponds to a certain reward. F == 0, G == 1, H == -1
            where F is frozen (ie ice), H is hole, G is the goal (firsbee)

        Returns:
        -------
        int
            The reward 
        """
        if letter == b"G":
            return 1
        elif letter == b"H":
            return -1
        else:
            return 0

    def is_episode_done(self, new_letter):
        """
        Checks if the episode is finished
        
        Parameters:
        ----------
        new_letter: bytes
            Each letter corresponds to a certain reward. F == 0, G == 1, H == -1
            where F is frozen (ie ice), H is hole, G is the goal (firsbee)
        
        Returns:
        -------
        boolean 
            
        """
        if new_letter == b"G" or new_letter == b"H":
            return True
        return False

    def step(self, action):
        """
        Moves the agent to an adjacent grid 
        
        Parameters:
        ----------
        action: int
            The Direction the agents wants to move

        Returns:
        -------
        tuple
            A tuple containing the new state the agent is in, the reward the agent gets and a boolean value to 
            see if the episode is finished 
        """
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

    # Resets the environment by placing the agent back to the starting point
    def reset(self):
        self.curr_row = 0
        self.curr_col = 0
        return self.convert_row_col_to_state(self.curr_row, self.curr_col)

    #Renders a map with obstacles, start and the goal location. A arrow is drawn to show the optimal policy
    def render(self, path):
        blue = [0, 255, 255] #ice colour 
        black = [0, 0 , 0] #obstacle colour
        green = [0, 255, 0] #start location color 
        red = [255, 0, 0] #goal location color

        #The following block adds color to the given map
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

        #Visulaise the map
        map_array = np.asarray(map_list, dtype=np.int32)
        fig = plt.figure(figsize=(8,8))
        self.draw_grid_arrow(path)
        plt.imshow(map_array)
        plt.grid(True)
        plt.subplots_adjust(left=0.25)
        plt.show()

    # Draws an arrow leading from the start point to the goal
    def draw_grid_arrow(self, path):
        for i in range(len(path) - 1):
            start_state_coordinate = self.convert_state_to_row_col(path[i])
            end_state_coordinate = self.convert_state_to_row_col(path[i+1])
            dx, dy = (end_state_coordinate[1] - start_state_coordinate[1]), (end_state_coordinate[0] - start_state_coordinate[0])
            
            #Draw a line with an arrow head only at the final state
            if (i == len(path) -2):
                plt.arrow(start_state_coordinate[1], start_state_coordinate[0], dx, dy, length_includes_head = True, head_width=0.2, head_length=0.2)
            else: 
                plt.arrow(start_state_coordinate[1], start_state_coordinate[0], dx, dy, length_includes_head = False)