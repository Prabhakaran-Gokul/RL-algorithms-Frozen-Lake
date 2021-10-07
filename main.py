from Parameters import *
from Environment import Environment
from MonteCarloControl import MonteCarlo
from SARSA import SARSA
from Q_Learning import Q_Learning
from ResultsGenerator import ResultsGenerator


if __name__ == "__main__":
    env = Environment(grid_size=GRID_SIZE)
    rg = ResultsGenerator()
    rg.initialise_results_table()
    monte_carlo = MonteCarlo(env, rg, epsilon=EPSILON, gamma=GAMMA)
    sarsa = SARSA(env, rg, epsilon=EPSILON, gamma=GAMMA, learning_rate=LEARNING_RATE)
    q_learning = Q_Learning(env, rg, epsilon=EPSILON, gamma=GAMMA, learning_rate=LEARNING_RATE)
    
    Q_values = monte_carlo.run(NUMBER_OF_EPISODES)
    monte_carlo.write_to_txt_file(Q_values, monte_carlo.name)
    print(monte_carlo.success_count, monte_carlo.failure_count)
    print(monte_carlo.test_policy(monte_carlo.policy_table))
    env.render(monte_carlo.get_optimal_path())

    Q_values = sarsa.run(NUMBER_OF_EPISODES)
    sarsa.write_to_txt_file(Q_values, sarsa.name)
    print(sarsa.success_count, sarsa.failure_count)
    print(sarsa.test_policy(sarsa.policy_table))
    env.render(sarsa.get_optimal_path())

    Q_values = q_learning.run(NUMBER_OF_EPISODES)
    q_learning.write_to_txt_file(Q_values, q_learning.name)
    print(q_learning.success_count, q_learning.failure_count)
    print(q_learning.test_policy(q_learning.policy_table))
    env.render(q_learning.get_optimal_path())

    rg.plot_all_average_reward_vs_episode()
    rg.plot_all_steps_vs_episode()
    rg.plot_computation_time()