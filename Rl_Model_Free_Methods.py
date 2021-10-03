from Parameters import *
from Environment import Environment

class RL_Model_Free_Methods():
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