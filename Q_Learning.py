from Parameters import *
from Rl_Model_Free_Methods import RL_Model_Free_Methods
import numpy as np
from tqdm import tqdm
import time

class Q_Learning(RL_Model_Free_Methods):
    def __init__(self, env, rg, epsilon, gamma, learning_rate):
        RL_Model_Free_Methods.__init__(self, env, rg, epsilon, gamma)
        self.learning_rate = learning_rate
        self.name = "Q_Learning"

    def run(self, num_of_episodes):
        #tqdm module is used here to display the progress of our training, via a progress bar
        start_time = time.time()
        print ("Q_Learning algorithm is starting...")
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
                
                #apply q_learning update rule     
                new_Q_value = max(self.Q_values[(new_state, a)] for a in self.actions)
                self.Q_values[(state, action)] +=  self.learning_rate * (reward + self.gamma * new_Q_value - self.Q_values[(state, action)])
                
                self.update_policy_table(state)
                
                if done:
                    if reward == 1: 
                        self.success_count += 1
                    if reward == -1:
                        self.failure_count += 1

                    self.total_reward += reward
                    break
                
                state = new_state
                action = new_action
            
            if not self.first_successful_episode_reached and reward == 1:
                self.rg.Q_Learning_results["First_successful_episode"].append(i)
                self.first_successful_episode_reached = True

            optimal_path = self.get_optimal_path()
            if optimal_path[-1] == self.env.n_row * self.env.n_col - 1: #goal state
                if not self.first_successful_policy_reached:
                    self.rg.Q_Learning_results["First_successful_policy"].append([optimal_path, i])
                    self.first_successful_policy_reached = True

            self.rg.Q_Learning_results["Average_Reward"].append(self.total_reward / (i + 1))
            self.rg.Q_Learning_results["Steps"].append(step)
            self.rg.Q_Learning_results["Success_Failure_Count"].append([self.success_count, self.failure_count])
        
        end_time = time.time()
        self.rg.Q_Learning_results["Computation_time"].append(end_time - start_time)
        if not self.first_successful_policy_reached:
            self.rg.Q_Learning_results["First_successful_policy"].append([[-1], -1])
        if not self.first_successful_episode_reached:
            self.rg.Q_Learning_results["First_successful_episode"].append(-1)
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
#     q_learning = Q_Learning(env, epsilon=EPSILON, gamma=GAMMA, learning_rate=LEARNING_RATE)
#     Q_values = q_learning.run(NUMBER_OF_EPISODES)
#     q_learning.write_to_txt_file(Q_values, q_learning.name)
#     print(q_learning.success_count, q_learning.failure_count)
#     print(q_learning.test_policy(q_learning.policy_table))
#     print(q_learning.get_optimal_path())
#     q_learning.rg.plot_average_reward_vs_episode(q_learning.name)