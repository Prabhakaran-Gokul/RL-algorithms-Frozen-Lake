import matplotlib.pyplot as plt
import numpy as np

class ResultsGenerator():
    def __init__(self):
        self.algorithm_names = ["Monte_carlo", "SARSA", "Q_Learning"]
        self.Monte_carlo_results, self.SARSA_results, self.Q_Learning_results = {}, {}, {}
        self.indicators = ["Average_Reward", "Computation_time", "Steps"]

    def initialise_results_table(self):
        for indicator in self.indicators:
            self.Monte_carlo_results[indicator] = []
            self.SARSA_results[indicator] = []
            self.Q_Learning_results[indicator] = []

    def plot_average_reward_vs_episode(self, algorithm_name):
        if algorithm_name == "Monte_carlo":
            average_reward = self.Monte_carlo_results["Average_Reward"]
        elif algorithm_name == "SARSA":
            average_reward = self.SARSA_results["Average_Reward"]
        elif algorithm_name == "Q_Learning":
            average_reward = self.Q_Learning_results["Average_Reward"]
        else:
            print("Invalid algorithm name")
        
        plt.plot(list(range(len(average_reward))), average_reward)
        plt.show()

    def plot_all_average_reward_vs_episode(self):
        average_reward = self.Monte_carlo_results["Average_Reward"]
        plt.plot(range(len(average_reward)), average_reward, 'r')
        
        average_reward = self.SARSA_results["Average_Reward"]
        plt.plot(range(len(average_reward)), average_reward, 'b')
        
        average_reward = self.Q_Learning_results["Average_Reward"]
        plt.plot(range(len(average_reward)), average_reward, 'g')

        plt.legend(self.algorithm_names)
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.title("Average Reward vs Episode")
        plt.show()


    def plot_computation_time(self):
        computation_time_data = [self.Monte_carlo_results["Computation_time"][0], self.SARSA_results["Computation_time"][0], self.Q_Learning_results["Computation_time"][0]]
        labels = self.algorithm_names
        plt.xticks(range(len(computation_time_data)), labels)
        plt.xlabel("Algorithms")
        plt.ylabel("Computational time")
        plt.title("Computational time of different algorithms")
        plt.bar(range(len(computation_time_data)), computation_time_data)
        plt.show()

    def plot_steps_vs_episode(self, algorithm_name):
        if algorithm_name == "Monte_carlo":
            steps = self.Monte_carlo_results["Steps"]
        if algorithm_name == "SARSA":
            steps = self.SARSA_results["Steps"]
        if algorithm_name == "Q_Learning":
            steps = self.Q_Learning_results["Steps"]
        
        plt.plot(range(len(steps)), steps)
        plt.legend(algorithm_name)
        plt.xlabel("Episode")
        plt.ylabel("Steps taken")
        plt.title("Steps Taken vs Episode")
        plt.show()

    def plot_all_steps_vs_episode(self):
        steps = self.Monte_carlo_results["Steps"]
        steps = np.cumsum(steps)
        plt.plot(range(len(steps)), steps, 'r')
        
        steps = self.SARSA_results["Steps"]
        steps = np.cumsum(steps)
        plt.plot(range(len(steps)), steps, 'b')
        
        steps = self.Q_Learning_results["Steps"]
        steps = np.cumsum(steps)
        plt.plot(range(len(steps)), steps, 'g')

        plt.legend(self.algorithm_names)
        plt.xlabel("Episode")
        plt.ylabel("Steps taken")
        plt.title("Steps Taken vs Episode")
        plt.show()