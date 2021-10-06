import matplotlib.pyplot as plt

class ResultsGenerator():
    def __init__(self):
        self.algorithm_names = ["Monte_carlo", "SARSA", "Q_Learning"]
        self.Monte_carlo_results, self.SARSA_results, self.Q_Learning_results = {}, {}, {}
        self.indicators = ["Average_Reward", "Computation_time"]

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
        pass

    def plot_computation_time(self):
        computation_time_data = [self.Monte_carlo_results["Computation_time"][0], self.SARSA_results["Computation_time"][0], self.Q_Learning_results["Computation_time"][0]]
        labels = self.algorithm_names
        plt.xticks(range(len(computation_time_data)), labels)
        plt.xlabel("Algorithms")
        plt.ylabel("Computational time")
        plt.title("Computational time of different algorithms")
        plt.bar(range(len(computation_time_data)), computation_time_data)
        plt.show()