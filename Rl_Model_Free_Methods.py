from Parameters import *
import numpy as np

class RL_Model_Free_Methods():
    def __init__(self, env, rg, epsilon, gamma):
        self.epsilon = epsilon
        self.gamma = gamma
        self.env = env
        self.actions = self.env.actions
        self.states = self.env.states
        self.rg = rg
        self.Q_values, self.returns, self.num_of_visits, self.policy_table = self.initialise_Q_table_and_policy_table()

        #to measure success and failures of episodes
        self.success_count = 0 
        self.failure_count = 0
        self.total_reward = 0
        self.first_successful_policy_reached = False
        self.first_successful_episode_reached = False

    def initialise_Q_table_and_policy_table(self):
        Q_values, returns, num_of_visits, policy_table = {}, {}, {}, {}
        for state in self.states:
            policy_table[state] = [1/len(self.actions)] * len(self.actions)
            for action in self.actions:
                Q_values[(state, action)] = 0
                returns[(state, action)] = 0
                num_of_visits[(state, action)] = 0
        return Q_values, returns, num_of_visits, policy_table

    def get_best_action(self, state):
        Q_values_at_given_state = []
        for a in self.actions:
            Q_values_at_given_state.append(self.Q_values[(state, a)])
        best_actions = []
        for idx, q in enumerate(Q_values_at_given_state):
            if q == max(Q_values_at_given_state):
                best_actions.append(idx)
        best_action = np.random.choice(best_actions)             
        return best_action

    def update_policy_table(self, state):
        best_action = self.get_best_action(state)
        for a in self.actions:
            if a == best_action:
                self.policy_table[state][a] = 1 - self.epsilon + self.epsilon/len(self.actions)
            else: 
                self.policy_table[state][a] = self.epsilon/len(self.actions)
    
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

    def write_to_txt_file(self, content, algorithm_name):
        counter = 1
        with open(algorithm_name + "_Q_values.txt", "w") as f:
            for key, value in content.items():
                if counter != 4:
                    f.write("%s:%s  " % (key,value))
                    counter += 1
                else:
                    f.write("%s:%s  \n" % (key,value))
                    counter = 1