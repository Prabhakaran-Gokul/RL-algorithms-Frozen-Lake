from Parameters import *
from Environment import Environment
import numpy as np
from tqdm import tqdm

class MonteCarlo:
    def __init__(self, env, epsilon, gamma):
        self.epsilon = epsilon
        self.gamma = gamma
        self.env = env
        self.actions = self.env.actions
        self.states = self.env.states
        self.Q_values, self.returns, self.num_of_visits, self.policy_table = self.initialise_Q_table_and_policy_table()

        #to measure success and failures of episodes
        self.success_count = 0 
        self.failure_count = 0

    def initialise_Q_table_and_policy_table(self):
        Q_values, returns, num_of_visits, policy_table = {}, {}, {}, {}
        for state in self.states:
            policy_table[state] = [1/len(self.actions)] * len(self.actions)
            for action in self.actions:
                Q_values[(state, action)] = 0
                returns[(state, action)] = 0
                num_of_visits[(state, action)] = 0
        return Q_values, returns, num_of_visits, policy_table

    # def epsilon_greedy_policy(self, state):
    #     best_action = max(self.actions, key = lambda x: self.Q_values[(state, x)])
    #     action_probabilities = [self.epsilon/len(self.actions)] * len(self.actions)
    #     action_probabilities[best_action] = 1 - self.epsilon + self.epsilon/len(self.actions)
    #     chosen_action = np.random.choice(self.actions, p = action_probabilities)
    #     return chosen_action

    # def random_policy(self):
    #     return np.random.randint(0,4)

    def get_best_action(self, state):
        best_action = max(self.actions, key = lambda x: self.Q_values[(state, x)])
        return best_action

    # def generate_episode(self):
    #     episode = [] #a list containing lists of [state, action, reward] for every step
    #     state = self.env.reset()
    #     step = 0
    #     while True:
    #         step += 1
    #         if step > MAX_STEPS:
    #             break
    #         action = self.epsilon_greedy_policy(state)
    #         new_state , reward, done = self.env.step(action)
    #         episode.append([state, action, reward])
    #         if done:
    #             if reward == 1: 
    #                 self.success_count += 1
    #             if reward == -1:
    #                 self.failure_count += 1
    #             break
    #         state = new_state
    #     return episode 

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
                break
            state = new_state
        return episode 


    def run(self, num_of_episodes):
        #tqdm module is used here to display the progress of our training, via a progress bar
        print ("First visit Monte Carlo control without Exploring starts algorithm is starting...")
        for i in tqdm(range(num_of_episodes)):
            episode = self.generate_episode(self.policy_table)
            G = 0
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

        return self.Q_values

    def write_to_txt_file(self, content):
        counter = 1
        with open("Monte_carlo_Q_values.txt", "w") as f:
            for key, value in content.items():
                if counter != 4:
                    f.write("%s:%s  " % (key,value))
                    counter += 1
                else:
                    f.write("%s:%s  \n" % (key,value))
                    counter = 1

    def plot_results(self, success_count, failure_count):
        pass
    
    #function to generate the optimal path using the training data
    def get_optimal_path(self):
        path = []
        state = self.env.reset() 
        step = 0 
        path.append(state)
        while True:
            step += 1
            if step > MAX_STEPS:
                break
            action = self.get_best_action(state)
            new_state , reward, done = self.env.step(action)
            path.append(new_state)
            if len(set(path)) != len(path): #check if the robot visits some place more than twice in the path
                break
            if done:
                break
            state = new_state
        return path

    def test_policy(self, policy):
        reach_goal = 0
        no_of_episodes = 100
        for i in range(no_of_episodes):
            final_reward = self.generate_episode(policy)[-1][-1]
            if final_reward == 1:
                reach_goal += 1
        return reach_goal / no_of_episodes

if __name__ == "__main__":
    env = Environment(grid_size=GRID_SIZE)
    monte_carlo = MonteCarlo(env, epsilon=EPSILON, gamma=GAMMA)
    Q_values = monte_carlo.run(NUMBER_OF_EPISODES)
    monte_carlo.write_to_txt_file(Q_values)
    print(monte_carlo.success_count, monte_carlo.failure_count)
    print(monte_carlo.test_policy(monte_carlo.policy_table))
    print(monte_carlo.get_optimal_path())
