import os.path

import numpy as np
import random
from timeit import default_timer as timer
import json
from tqdm import tqdm
from os import makedirs
import argparse

from gym.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv


np.random.seed(2)
sixteen = generate_random_map(16)
np.random.seed(44)
tvelve = generate_random_map(12)

np.random.seed(340)
eight = generate_random_map(8)
np.random.seed(340)
twenty_five = generate_random_map(25)

OUTPUT_DIR = "../outputs/trained_policies"
makedirs(OUTPUT_DIR, exist_ok=True)


if not os.path.exists("../outputs/maps.json"):
    MAPS = {
        "4x4": [
            "SFFF",
            "FHFH",
            "FFFH",
            "HFFG"
        ],
        # "8x8": [
        #     "SFFFFFFF",
        #     "FFFFFFFF",
        #     "FFFHFFFF",
        #     "FFFFFHFF",
        #     "FFFHFFFF",
        #     "FHHFFFHF",
        #     "FHFFHFHF",
        #     "FFFHFFFG"
        # ],
        "8x8": eight,
        "12x12": tvelve,
        "16x16": sixteen,
        "25x25": twenty_five
    }

    with open("../outputs/maps.json", "w") as output_file:
        json.dump(MAPS, output_file)
else:
    with open("../outputs/maps.json", "r") as input_file:
        MAPS = json.load(input_file)


def test_policy(env, policy, n_epoch=1000, n_episode=1000):
    rewards = []
    episode_counts = []
    for i in range(n_epoch):
        current_state, _ = env.reset()
        episode = 0
        episode_reward = 0
        done = False

        while not done and episode < n_episode:
            episode += 1
            act = int(policy[current_state])
            new_state, reward, done, _, _ = env.step(act)
            episode_reward += reward
            current_state = new_state

        rewards.append(episode_reward)
        episode_counts.append(episode)

    # all done
    mean_reward = sum(rewards) / len(rewards)
    mean_episode = sum(episode_counts) / len(episode_counts)
    return mean_reward, mean_episode, rewards, episode_counts

def value_iteration(env, discount=0.9, epsilon=1e-12, best_setting=False):
    start = timer()

    number_of_states = env.observation_space.n
    number_of_actions = env.action_space.n

    policy = np.zeros((1, number_of_states))
    value_list = np.zeros((1, number_of_states))    # New state values
    old_value_list = value_list.copy()              # Old state values

    episode = 0
    max_change = 1
    sigma = discount
    plot_data = {}

    # Phat: Convergence condition
    while max_change > epsilon:
        episode += 1
        plot_data[episode] = {}

        # For each state, check all possible actions
        start_iter = timer()
        for s in range(number_of_states):
            assigned_value = -np.inf
            # and select the action maximizing the reward
            for a in range(number_of_actions):
                # Get new state and its reward for each action
                total_cand_value = 0
                for prob, new_state, reward, done in env.P[s][a]:
                    # get new states value
                    value_new_state = old_value_list[0][new_state]
                    cand_value = reward if done else reward + sigma * value_new_state
                    total_cand_value += cand_value * prob

                if total_cand_value > assigned_value:
                    assigned_value = total_cand_value
                    policy[0][s] = a
                    value_list[0][s] = assigned_value

        changes = np.abs(value_list - old_value_list)
        max_change = np.max(changes)
        old_value_list = value_list.copy()

        plot_data[episode]["reward"] = value_list.mean()        # Mean of state values
        plot_data[episode]["policy"] = policy[0]
        plot_data[episode]["time"] = timer() - start_iter
        plot_data[episode]["max_delta"] = max_change            # Change in new and old state values

    end = timer()
    time_spent = end - start    # timedelta(seconds=end - start)
    print("Solved in: {} episodes and {} seconds".format(episode, time_spent))

    if best_setting:
        return policy[0], episode, time_spent, plot_data

    return policy[0], episode, time_spent
def policy_iteration(env, discount=0.9, epsilon=1e-3, best_setting=False):
    start = timer()

    number_of_states = env.observation_space.n
    number_of_actions = env.action_space.n

    # Generate a random policy
    policy = np.random.randint(number_of_actions, size=(1, number_of_states))
    value_list = np.zeros((1, number_of_states))
    old_value_list = value_list.copy()
    episode = 0
    sigma = discount
    plot_data = {}

    policy_stable = False
    while not policy_stable:
        episode += 1
        plot_data[episode] = {}
        start_iter = timer()

        eval_acc = True
        while eval_acc:
            eps = 0
            for s in range(number_of_states):
                v = value_list[0][s]        # first row
                a = policy[0][s]            # get the new value
                total_val_new_state = 0

                for prob, new_state, reward, done in env.P[s][a]:
                    value_new_state = value_list[0][new_state]
                    cand_value = reward if done else reward + sigma * value_new_state       # second row
                    total_val_new_state += cand_value * prob

                value_list[0][s] = total_val_new_state
                eps = max(eps, np.abs(v - value_list[0][s]))                                # third row

            if eps < epsilon:
                eval_acc = False

        changes = np.abs(value_list - old_value_list)
        max_change = np.max(changes)
        old_value_list = value_list.copy()

        # Check if all old actions are the best (i.e., no other actions yield better total reward)
        # If YES --> quit, if NOT --> update policy and continue with eval_acc
        policy_stable = True
        for s in range(number_of_states):
            old_action = policy[0][s]
            max_value = -np.inf     # get the argmax a here
            for a in range(number_of_actions):
                # get the new value
                total_cand_value = 0
                for prob, new_state, reward, done in env.P[s][a]:
                    value_new_state = value_list[0][new_state]
                    cand_value = reward if done else reward + sigma * value_new_state
                    total_cand_value += prob * cand_value
                if total_cand_value > max_value:
                    max_value = total_cand_value
                    policy[0][s] = a

            if old_action != policy[0][s]:
                policy_stable = False

        plot_data[episode]["reward"] = value_list.mean()
        plot_data[episode]["policy"] = policy[0]
        plot_data[episode]["time"] = timer() - start_iter
        plot_data[episode]["max_delta"] = max_change

    end = timer()
    time_spent = end - start    # timedelta(seconds=end - start)
    print("Solved in: {} episodes and {} seconds".format(episode, time_spent))

    if best_setting:
        return policy[0], episode, time_spent, plot_data

    return policy[0], episode, time_spent
def train_and_test_pi_vi(env, discounts=[0.9], epsilons=[1e-9], mute=False):
    # run value iteration
    vi_dict = {}
    for dis in discounts:
        vi_dict[dis] = {}
        for epsilon in epsilons:
            vi_dict[dis][epsilon] = {}

            # run value iteration
            vi_policy, vi_solve_iter, vi_solve_time = value_iteration(env, dis, epsilon)
            vi_mrews, vi_mepisode, _, _ = test_policy(env, vi_policy, n_episode=1000)
            vi_dict[dis][epsilon]["mean_reward"] = vi_mrews
            vi_dict[dis][epsilon]["mean_episode"] = vi_mepisode
            vi_dict[dis][epsilon]["iteration"] = vi_solve_iter
            vi_dict[dis][epsilon]["time_spent"] = vi_solve_time
            vi_dict[dis][epsilon]["policy"] = vi_policy.tolist()
            if not mute:
                print("Value iteration for {} discount and {} epsilon is done".format(dis, epsilon))
                print("Iteration: {} time: {}".format(vi_solve_iter, vi_solve_time))
                print("Mean reward: {} - mean episode: {}".format(vi_mrews, vi_mepisode))

    # run policy iteration
    pi_dict = {}
    for dis in discounts:
        pi_dict[dis] = {}
        for epsilon in epsilons:
            pi_dict[dis][epsilon] = {}

            pi_policy, pi_solve_iter, pi_solve_time = policy_iteration(env, dis, epsilon)
            pi_mrews, pi_mepisode, _, _ = test_policy(env, pi_policy)
            pi_dict[dis][epsilon]["mean_reward"] = pi_mrews
            pi_dict[dis][epsilon]["mean_episode"] = pi_mepisode
            pi_dict[dis][epsilon]["iteration"] = pi_solve_iter
            pi_dict[dis][epsilon]["time_spent"] = pi_solve_time
            pi_dict[dis][epsilon]["policy"] = pi_policy.tolist()
            if not mute:
                print("Policy iteration for {} discount and {} epsilon is done".format(dis, epsilon))
                print("Iteration: {} time: {}".format(pi_solve_iter, pi_solve_time))
                print("Mean reward: {} - mean episode: {}".format(pi_mrews, pi_mepisode))

    return vi_dict, pi_dict

def q_learning(env, discount=0.9, total_episodes=1e5, alpha=0.1, epsilon=1.0, eps_decay_rate=None, min_epsilon=0.01, best_setting=False):
    start = timer()

    number_of_states = env.observation_space.n
    number_of_actions = env.action_space.n

    qtable = np.zeros((number_of_states, number_of_actions))
    learning_rate = alpha
    gamma = discount

    if eps_decay_rate is None:
        eps_decay_rate = 1. / total_episodes

    rewards = []
    plot_data = {}
    old_value_list = np.zeros((1, number_of_states))    # Old state values

    for episode in tqdm(range(1, int(total_episodes) + 1, 1)):
        # reset the environment
        state, _ = env.reset()
        total_reward = 0
        plot_data[episode] = {}
        start_iter = timer()

        while True:
            # choose an action a in the current world state
            exp_exp_tradeoff = random.uniform(0, 1)

            # if greater than epsilon --> exploitation
            if exp_exp_tradeoff > epsilon:
                b = qtable[state, :]
                action = np.random.choice(np.where(b == b.max())[0])
                # TODO: WHY NOT THIS? --> Since if there are multiple max values --> Randomly select one rather than always selecting the first one
                # action = np.argmax(qtable[state, :])
            # else choose exploration (i.e., a random action)
            else:
                action = env.action_space.sample()

            # take action (a) and observe the outcome state (s') and reward (r)
            new_state, reward, done, info, _ = env.step(action)
            total_reward += reward

            # update Q(s,a) := Q(s,a) + lr [R(s,a) + gamma * max(Q (s', a') - Q(s,a))]
            if not done:
                qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
            else:
                qtable[state, action] = qtable[state, action] + learning_rate * (reward - qtable[state, action])

            # change state
            state = new_state

            # is it Done
            if done:
                break

        # reduce epsilon
        rewards.append(total_reward)
        epsilon = max(eps_decay_rate * epsilon, min_epsilon)
        # epsilon = max(max_epsilon - decay_rate * episode, min_epsilon)
        # print("New epsilon: {}".format(epsilon))

        policy = np.argmax(qtable, axis=1)
        value_list = np.max(qtable, axis=1)
        changes = np.abs(value_list - old_value_list)
        max_change = np.max(changes)
        old_value_list = value_list.copy()

        plot_data[episode]["reward"] = value_list.mean()
        plot_data[episode]["policy"] = policy
        plot_data[episode]["q-table"] = qtable
        plot_data[episode]["time"] = timer() - start_iter
        plot_data[episode]["max_delta"] = max_change

    end = timer()
    time_spent = end - start
    print("Solved in: {} episodes and {} seconds".format(total_episodes, time_spent))

    if best_setting:
        return np.argmax(qtable, axis=1), total_episodes, time_spent, qtable, rewards, plot_data

    return np.argmax(qtable, axis=1), total_episodes, time_spent, qtable, rewards
def train_and_test_q_learning(env, discounts=[0.9], total_episodes=[1e5], alphas=[0.1], epsilons=[0.9], epsilon_decay_rates=[0.01], mute=False):
    min_epsilon = 0.01
    q_dict = {}

    for dis in discounts:
        q_dict[dis] = {}
        for episode in total_episodes:
            q_dict[dis][episode] = {}
            for alpha in alphas:
                q_dict[dis][episode][alpha] = {}
                for epsilon in epsilons:
                    q_dict[dis][episode][alpha][epsilon] = {}
                    for epsilon_dr in epsilon_decay_rates:
                        q_dict[dis][episode][alpha][epsilon][epsilon_dr] = {}

                        # run q_learning
                        q_policy, q_solve_iter, q_solve_time, q_table, rewards = q_learning(env, dis, episode, alpha, epsilon, epsilon_dr, min_epsilon)
                        q_mrews, q_mepisode, _, _ = test_policy(env, q_policy, n_episode=int(episode))
                        q_dict[dis][episode][alpha][epsilon][epsilon_dr]["mean_reward"] = q_mrews
                        q_dict[dis][episode][alpha][epsilon][epsilon_dr]["mean_episode"] = q_mepisode
                        q_dict[dis][episode][alpha][epsilon][epsilon_dr]["iteration"] = q_solve_iter
                        q_dict[dis][episode][alpha][epsilon][epsilon_dr]["time_spent"] = q_solve_time
                        q_dict[dis][episode][alpha][epsilon][epsilon_dr]["policy"] = q_policy.tolist()
                        q_dict[dis][episode][alpha][epsilon][epsilon_dr]["q-table"] = q_table.tolist()    # Specific for Q-learning
                        q_dict[dis][episode][alpha][epsilon][epsilon_dr]["rewards"] = rewards             # Specific for Q-learning
                        if not mute:
                            print("gamma: {} total_episode: {} lr: {}, epsilon: {}, epsilon_dr: {}".format(dis, episode, alpha, epsilon, epsilon_dr))
                            print("Iteration: {} time: {}".format(q_solve_iter, q_solve_time))
                            print("Mean reward: {} - mean episode: {}".format(q_mrews, q_mepisode))
    return q_dict


def run_frozen_lake_vi_pi(grid_size=8):
    version = "small" if grid_size < 16 else "large"
    vi_file_path = f"{OUTPUT_DIR}/frozen_lake_{version}_vi_policies.json"
    pi_file_path = f"{OUTPUT_DIR}/frozen_lake_{version}_pi_policies.json"
    print(f"Running VI and PI for {version.upper()} Frozen Lake...")

    if not os.path.exists(vi_file_path) or not os.path.exists(pi_file_path):
        env = FrozenLakeEnv(desc=MAPS[f"{grid_size}x{grid_size}"])
        discounts = [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
        epsilons = [1e-3, 1e-6, 1e-9, 1e-12, 1e-15]
        vi_dict, pi_dict = train_and_test_pi_vi(env, discounts=discounts, epsilons=epsilons, mute=True)

        with open(vi_file_path, "w") as output_file:
            json.dump(vi_dict, output_file)
        with open(pi_file_path, "w") as output_file:
            json.dump(pi_dict, output_file)
    else:
        print(f"Existing policies found in {OUTPUT_DIR}")
    print(f"COMPLETE {version.upper()} FROZEN LAKE FOR VI & PI!!!")

def run_frozen_lake_q_learning(grid_size=8):
    version = "small" if grid_size < 16 else "large"
    q_file_path = f"{OUTPUT_DIR}/frozen_lake_{version}_Qlearning_policies.json"
    print(f"Running Q-Learning for {version.upper()} Frozen Lake...")

    if not os.path.exists(q_file_path):
        env = FrozenLakeEnv(desc=MAPS[f"{grid_size}x{grid_size}"])
        discounts = [0.75, 0.9, 0.999]
        episodes = [1e3, 1e4, 1e5]
        alphas = [0.01, 0.1]
        epsilons = [1.0, 0.75, 0.5]
        eps_decay_rates = [0.99, 0.999, 0.9999]

        q_dict = train_and_test_q_learning(env, discounts=discounts, total_episodes=episodes, alphas=alphas, epsilons=epsilons, epsilon_decay_rates=eps_decay_rates)

        with open(q_file_path, "w") as output_file:
            json.dump(q_dict, output_file)
    else:
        print(f"Existing policies found in {OUTPUT_DIR}")
    print(f"COMPLETE {version.upper()} FROZEN LAKE FOR Q-LEARNING!!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--q_learning', action='store_true')
    parser.add_argument('--large', action='store_true')
    args = parser.parse_args()

    if args.q_learning:
        run_frozen_lake_q_learning(grid_size=25 if args.large else 8)
    else:
        run_frozen_lake_vi_pi(grid_size=25 if args.large else 8)

