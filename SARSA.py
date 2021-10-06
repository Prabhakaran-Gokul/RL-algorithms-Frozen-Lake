from Parameters import *
from Environment import Environment
import numpy as np
from tqdm import tqdm
from Rl_Model_Free_Methods import RL_Model_Free_Methods
import time

class SARSA(RL_Model_Free_Methods):
    def __init__(self, env, rg, epsilon, gamma, learning_rate):
        RL_Model_Free_Methods.__init__(self, env, rg, epsilon, gamma)
        self.learning_rate = learning_rate
        self.name = "SARSA"

    def run(self, num_of_episodes):
        #tqdm module is used here to display the progress of our training, via a progress bar
        start_time = time.time()
        print ("SARSA algorithm is starting...")
        for i in tqdm(range(num_of_episodes)):
            step = 0
            state = self.env.reset()
            action = np.random.choice(self.actions, p = self.policy_table[state])
            while True:
                step += 1
                if step > MAX_STEPS:
                    break
                new_state , reward, done = self.env.step(action)
                new_action = np.random.choice(self.actions, p = self.policy_table[new_state])
                
                #apply SARSA update rule
                new_Q_value = self.Q_values[(new_state, new_action)]
                self.Q_values[(state, action)] +=  self.learning_rate * (reward + self.gamma * new_Q_value - self.Q_values[(state, action)])
                best_action = self.get_best_action(state)
                
                for a in self.actions:
                    if a == best_action:
                        self.policy_table[state][a] = 1 - self.epsilon + self.epsilon/len(self.actions)
                    else: 
                        self.policy_table[state][a] = self.epsilon/len(self.actions)
                if done:
                    if reward == 1: 
                        self.success_count += 1
                    if reward == -1:
                        self.failure_count += 1
                    
                    self.total_reward += reward
                    break

                state = new_state
                action = new_action
            
            self.rg.SARSA_results["Average_Reward"].append(self.total_reward / (i + 1))
        
        end_time = time.time()
        self.rg.SARSA_results["Computation_time"].append(end_time - start_time)
        return self.Q_values

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
                break
            state = new_state
        return episode 

# if __name__ == "__main__":
#     env = Environment(grid_size=GRID_SIZE)
#     sarsa = SARSA(env, epsilon=EPSILON, gamma=GAMMA, learning_rate=LEARNING_RATE)
#     Q_values = sarsa.run(NUMBER_OF_EPISODES)
#     sarsa.write_to_txt_file(Q_values, sarsa.name)
#     print(sarsa.success_count, sarsa.failure_count)
#     print(sarsa.test_policy(sarsa.policy_table))
#     print(sarsa.get_optimal_path())
#     sarsa.rg.plot_average_reward_vs_episode(sarsa.name)
