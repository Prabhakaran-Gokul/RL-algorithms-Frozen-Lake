from Parameters import *
from Environment import Environment
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
                    #TODO make a function for the code below update_policy_table(state)
                    best_action = self.get_best_action(state)
                    for a in self.actions:
                        if a == best_action:
                            self.policy_table[state][a] = 1 - self.epsilon + self.epsilon/len(self.actions)
                        else: 
                            self.policy_table[state][a] = self.epsilon/len(self.actions)

        end_time = time.time()
        self.rg.Monte_carlo_results["Computation_time"].append(end_time - start_time)
        return self.Q_values

# if __name__ == "__main__":
#     env = Environment(grid_size=GRID_SIZE)
#     monte_carlo = MonteCarlo(env, epsilon=EPSILON, gamma=GAMMA)
#     Q_values = monte_carlo.run(NUMBER_OF_EPISODES)
#     monte_carlo.write_to_txt_file(Q_values, monte_carlo.name)
#     print(monte_carlo.success_count, monte_carlo.failure_count)
#     print(monte_carlo.test_policy(monte_carlo.policy_table))
#     print(monte_carlo.get_optimal_path())
#     # monte_carlo.rg.plot_average_reward_vs_episode(monte_carlo.name)
#     monte_carlo.rg.plot_computation_time()
