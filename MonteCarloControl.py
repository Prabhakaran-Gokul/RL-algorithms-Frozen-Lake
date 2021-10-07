from Parameters import *
import numpy as np
from tqdm import tqdm
from Rl_Model_Free_Methods import RL_Model_Free_Methods
import time

class MonteCarlo(RL_Model_Free_Methods):
    def __init__(self, env, rg, epsilon, gamma):
        RL_Model_Free_Methods.__init__(self, env, rg, epsilon, gamma)
        self.name = "Monte_carlo"

    def generate_episode(self, policy_table):
        episode = [] #a list containing lists of [state, action, reward] for every step
        state = self.env.reset()
        step = 0
        while True:
            step += 1
            if step > MAX_STEPS:
                break
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


    def run(self, num_of_episodes):
        #tqdm module is used here to display the progress of our training, via a progress bar
        start_time = time.time()
        print ("First visit Monte Carlo control without Exploring starts algorithm is starting...")
        for i in tqdm(range(num_of_episodes)):
            episode = self.generate_episode(self.policy_table)
            G = 0

            self.rg.Monte_carlo_results["Average_Reward"].append(self.total_reward / (i + 1))
            for t in reversed(range(0, len(episode))):
                state , action, reward = episode[t]
                G = self.gamma * G + reward
                if (state, action) not in [(x[0], x[1]) for x in episode[0:t]]:
                    self.returns[(state, action)] += G
                    self.num_of_visits[(state, action)] += 1
                    self.Q_values[(state, action)] = self.returns[(state, action)] / self.num_of_visits[(state, action)]
                    
                    self.update_policy_table(state)
                    
            optimal_path = self.get_optimal_path()
            if optimal_path[-1] == self.env.n_row * self.env.n_col - 1: #goal state
                if not self.first_successful_policy_reached:
                    self.rg.Monte_carlo_results["First_successful_policy"].append([optimal_path, i])
                    self.first_successful_policy_reached = True

            if not self.first_successful_episode_reached and episode[-1][-1] == 1:
                self.rg.Monte_carlo_results["First_successful_episode"].append(i)
                self.first_successful_episode_reached = True

        end_time = time.time()
        self.rg.Monte_carlo_results["Computation_time"].append(end_time - start_time)
        if not self.first_successful_policy_reached:
            self.rg.Monte_carlo_results["First_successful_policy"].append([[-1], -1])
        if not self.first_successful_episode_reached:
            self.rg.Monte_carlo_results["First_successful_episode"].append(-1)
        return self.Q_values