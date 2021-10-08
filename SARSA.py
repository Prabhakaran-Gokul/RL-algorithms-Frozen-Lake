from Parameters import *
import numpy as np
from tqdm import tqdm
from Rl_Model_Free_Methods import RL_Model_Free_Methods
import time

class SARSA(RL_Model_Free_Methods):
    def __init__(self, env, rg, epsilon, gamma, learning_rate):
        RL_Model_Free_Methods.__init__(self, env, rg, epsilon, gamma)
        self.learning_rate = learning_rate
        self.name = "SARSA"

    #runs the SARSA algorithm
    def run(self, num_of_episodes):
        #tqdm module is used here to display the progress of our training, via a progress bar
        start_time = time.time() # start timer
        print ("SARSA algorithm is starting...")
        for i in tqdm(range(num_of_episodes)):
            step = 0
            state = self.env.reset() #resets the enviroment. Agent goes back to the starting point
            action = np.random.choice(self.actions, p = self.policy_table[state]) #choose a policy according to the policy table
            while True:
                step += 1
                if step > MAX_STEPS:
                    break
                new_state , reward, done = self.env.step(action)
                new_action = np.random.choice(self.actions, p = self.policy_table[new_state])  #choose a policy according to the policy table
                
                #apply SARSA update rule
                new_Q_value = self.Q_values[(new_state, new_action)]
                self.Q_values[(state, action)] +=  self.learning_rate * (reward + self.gamma * new_Q_value - self.Q_values[(state, action)])
                
                self.update_policy_table(state) #update the policy table using the epsilon greedy policy

                if done:
                    if reward == 1: 
                        self.success_count += 1
                    if reward == -1:
                        self.failure_count += 1
                    
                    self.total_reward += reward
                    break
                
               
                state = new_state
                action = new_action

            #if agent reaches the goal for the first time, add the corrosponding episode number to the Results Generator
            if not self.first_successful_episode_reached and reward == 1:
                self.rg.SARSA_results["First_successful_episode"].append(i)
                self.first_successful_episode_reached = True

            #Generate an optimal path and check if it is the first time it reaches the goal. if yes, append the path and the current episode number to Results generator 
            optimal_path = self.get_optimal_path()
            if optimal_path[-1] == self.env.n_row * self.env.n_col - 1: #goal state
                if not self.first_successful_policy_reached:
                    self.rg.SARSA_results["First_successful_policy"].append([optimal_path, i])
                    self.first_successful_policy_reached = True

            self.rg.SARSA_results["Average_Reward"].append(self.total_reward / (i + 1))
            self.rg.SARSA_results["Steps"].append(step)
            self.rg.SARSA_results["Success_Failure_Count"].append([self.success_count, self.failure_count])
        
        end_time = time.time() #end timer
        self.rg.SARSA_results["Computation_time"].append(end_time - start_time) #add the calculated computational time to Results Generator
        #if the agent does not each the goal or does not generate an optimal policy, add -1 in the following format to the Results Generator to indicate failure
        if not self.first_successful_policy_reached:
            self.rg.SARSA_results["First_successful_policy"].append([[-1], -1])
        if not self.first_successful_episode_reached:
            self.rg.SARSA_results["First_successful_episode"].append(-1)
        return self.Q_values

    # Generates an episode and returns a list containing multiple steps (ie. [state, action, reward])
    def generate_episode(self, policy_table):
        episode = [] #a list containing lists of [state, action, reward] for every step
        state = self.env.reset() #send the agent to the starting point
        step = 0
        #steps will stop generating when the NAX_STEPS is reached or when the agent reaches a hole or a goal
        while True:
            step += 1
            if step > MAX_STEPS:
                break
            action = np.random.choice(self.actions, p = policy_table[state])
            new_state , reward, done = self.env.step(action)
            episode.append([state, action, reward])
            if done:
                break
            state = new_state
        return episode 