from Parameters import *
from Environment import Environment
from MonteCarloControl import MonteCarlo
from SARSA import SARSA
from Q_Learning import Q_Learning
from ResultsGenerator import ResultsGenerator

def get_first_successful_episode_and_policy(env, rg):
    no_of_runs = 10
    for i in range(no_of_runs):
        monte_carlo = MonteCarlo(env, rg, epsilon=EPSILON, gamma=GAMMA)
        monte_carlo.run(NUMBER_OF_EPISODES)
        
        sarsa = SARSA(env, rg, epsilon=EPSILON, gamma=GAMMA, learning_rate=LEARNING_RATE)
        sarsa.run(NUMBER_OF_EPISODES)

        q_learning = Q_Learning(env, rg, epsilon=EPSILON, gamma=GAMMA, learning_rate=LEARNING_RATE)
        q_learning.run(NUMBER_OF_EPISODES)

    print (rg.Monte_carlo_results["First_successful_episode"])
    print (rg.Monte_carlo_results["First_successful_policy"])
    print ("\n")

    print (rg.SARSA_results["First_successful_episode"])
    print (rg.SARSA_results["First_successful_policy"])
    print ("\n")

    print (rg.Q_Learning_results["First_successful_episode"])
    print (rg.Q_Learning_results["First_successful_policy"])

def get_success_rate_of_algorithms(env, rg):
    monte_carlo_success_count = 0
    sarsa_success_count = 0
    q_learning_success_count = 0
    no_of_runs = 10
    for i in range(no_of_runs):
        monte_carlo = MonteCarlo(env, rg, epsilon=EPSILON, gamma=GAMMA)
        monte_carlo.run(NUMBER_OF_EPISODES)
        optimal_path = monte_carlo.get_optimal_path()
        if optimal_path[-1] == env.n_row * env.n_col - 1: #goal state
            monte_carlo_success_count += 1
        
        sarsa = SARSA(env, rg, epsilon=EPSILON, gamma=GAMMA, learning_rate=LEARNING_RATE)
        sarsa.run(NUMBER_OF_EPISODES)
        optimal_path = sarsa.get_optimal_path()
        if optimal_path[-1] == env.n_row * env.n_col - 1: #goal state
            sarsa_success_count += 1

        q_learning = Q_Learning(env, rg, epsilon=EPSILON, gamma=GAMMA, learning_rate=LEARNING_RATE)
        q_learning.run(NUMBER_OF_EPISODES)
        optimal_path = q_learning.get_optimal_path()
        if optimal_path[-1] == env.n_row * env.n_col - 1: #goal state
            q_learning_success_count += 1
    return [monte_carlo_success_count, sarsa_success_count, q_learning_success_count]


if __name__ == "__main__":
    env = Environment(grid_size=GRID_SIZE)
    rg = ResultsGenerator()
    rg.initialise_results_table()

    # get_first_successful_episode_and_policy()
    success_count = get_success_rate_of_algorithms(env, rg)
    print (success_count)
    rg.plot_computation_time()

    