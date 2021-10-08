from Parameters import *
import numpy as np
from tqdm import tqdm
from Rl_Model_Free_Methods import RL_Model_Free_Methods
import time

class MonteCarlo(RL_Model_Free_Methods):
    def __init__(self, env, rg, epsilon, gamma):
        RL_Model_Free_Methods.__init__(self, env, rg, epsilon, gamma)
        self.name = "Monte_carlo"

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
            #get the next state the agent is going to and create a step
            action = np.random.choice(self.actions, p = policy_table[state])
            new_state , reward, done = self.env.step(action)
            episode.append([state, action, reward])
            if done:
                if reward == 1: 
                    self.success_count += 1
                if reward == -1:
                    self.failure_count += 1
                self.total_reward += reward 
                break
            state = new_state
        
        self.rg.Monte_carlo_results["Success_Failure_Count"].append([self.success_count, self.failure_count])
        self.rg.Monte_carlo_results["Steps"].append(step)

        return episode 

    #runs the monte carlo algorithm
    def run(self, num_of_episodes):
        #tqdm module is used here to display the progress of our training, via a progress bar
        start_time = time.time() #start timer
        print ("First visit Monte Carlo control without Exploring starts algorithm is starting...")
        for i in tqdm(range(num_of_episodes)):
            #An Episode is generated
            episode = self.generate_episode(self.policy_table)
            G = 0

            self.rg.Monte_carlo_results["Average_Reward"].append(self.total_reward / (i + 1)) #add the current average reward to results generator
            for t in reversed(range(0, len(episode))):
                state , action, reward = episode[t]
                G = self.gamma * G + reward #Get the G values
                if (state, action) not in [(x[0], x[1]) for x in episode[0:t]]: # if (state, action) is not in any of the steps prior to this current step, then consider it as first visit
                    self.returns[(state, action)] += G
                    self.num_of_visits[(state, action)] += 1
                    self.Q_values[(state, action)] = self.returns[(state, action)] / self.num_of_visits[(state, action)] #Average the G values for a particular state and action
                    
                    self.update_policy_table(state) #update the policy table using the epsion greedy policy
                    
            #Generate an optimal path and check if it is the first time it reaches the goal. if yes, append the path and the current episode number to Results generator 
            optimal_path = self.get_optimal_path()
            if optimal_path[-1] == self.env.n_row * self.env.n_col - 1: #goal state
                if not self.first_successful_policy_reached:
                    self.rg.Monte_carlo_results["First_successful_policy"].append([optimal_path, i])
                    self.first_successful_policy_reached = True
            
            #check if it is the first time agent goal. if yes, append the current episode number to Results generator 
            if not self.first_successful_episode_reached and episode[-1][-1] == 1:
                self.rg.Monte_carlo_results["First_successful_episode"].append(i)
                self.first_successful_episode_reached = True

        
        end_time = time.time() #end timer
        self.rg.Monte_carlo_results["Computation_time"].append(end_time - start_time) #add the computation time to Results Generator
        #if the agent does not each the goal or does not generate an optimal policy, add -1 in the following format to the Results Generator to indicate failure
        if not self.first_successful_policy_reached:
            self.rg.Monte_carlo_results["First_successful_policy"].append([[-1], -1])
        if not self.first_successful_episode_reached:
            self.rg.Monte_carlo_results["First_successful_episode"].append(-1)
        return self.Q_values