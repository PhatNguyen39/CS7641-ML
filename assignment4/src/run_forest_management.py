from hiive.mdptoolbox.mdp import QLearning
from mdptoolbox.mdp import ValueIteration, PolicyIteration, PolicyIterationModified
from mdptoolbox.example import forest

from itertools import product
import pandas as pd
import numpy as np
from numpy.random import choice

from tqdm import tqdm
from os import makedirs
import argparse

np.random.seed(340)

OUTPUT_DIR = "../outputs/trained_policies_new"
makedirs(OUTPUT_DIR, exist_ok=True)


def test_policy(P, R, policy, test_count=1000, gamma=0.9):
    num_state = P.shape[-1]
    total_episode = num_state * test_count

    # start in each state
    total_reward = 0
    for state in range(num_state):
        state_reward = 0
        for state_episode in range(test_count):
            episode_reward = 0
            disc_rate = 1
            while True:
                action = policy[state]                              # take step
                probs = P[action][state]                            # get next step using P
                candidates = list(range(len(P[action][state])))
                next_state = choice(candidates, 1, p=probs)[0]
                reward = R[state][action] * disc_rate               # get the reward
                episode_reward += reward
                disc_rate *= gamma                                  # when go back to 0 ended

                if next_state == 0:
                    break

            state_reward += episode_reward
        total_reward += state_reward

    return total_reward / total_episode


def train_vi(P, R, discounts=[0.9], epsilons=[1e-9], max_iter=1e6):
    vi_df = pd.DataFrame(columns=["Discount", "Epsilon", "Policy", "Iteration", "Time", "Reward", "Value Function"])

    for idx, (dis, epsilon) in tqdm(enumerate(product(discounts, epsilons)), total=len(discounts)*len(epsilons)):
        vi = ValueIteration(P, R, discount=dis, epsilon=epsilon, max_iter=max_iter)
        vi.run()
        reward = test_policy(P, R, vi.policy)
        print("{}: {}".format(idx, reward))

        info = [float(dis), float(epsilon), vi.policy, vi.iter, vi.time, reward, vi.V]
        df_length = len(vi_df)
        vi_df.loc[df_length] = info

    return vi_df


def train_pi(P, R, discounts=[0.9], epsilons=[1e-9], max_iter=1e6):
    pi_df = pd.DataFrame(columns=["Discount", "Epsilon", "Policy", "Iteration", "Time", "Reward", "Value Function"])

    for idx, (dis, epsilon) in tqdm(enumerate(product(discounts, epsilons)), total=len(discounts)*len(epsilons)):
        pi = PolicyIterationModified(P, R, discount=dis, epsilon=epsilon, max_iter=max_iter)
        pi.run()
        reward = test_policy(P, R, pi.policy)
        print("{}: {}".format(idx, reward))

        info = [float(dis), float(epsilon), pi.policy, pi.iter, pi.time, reward, pi.V]
        df_length = len(pi_df)
        pi_df.loc[df_length] = info

    return pi_df


def train_q_learning(P, R, discounts=[0.9], alphas=[0.1], epsilons=[1.0], epsilon_decays=[0.99], n_iters=[1e4]):
    q_df = pd.DataFrame(columns=["Iteration", "Alpha", "Epsilon", "Epsilon Decay", "Reward",
                                 "Time", "Policy", "Value Function", "Training Reward"])

    for idx, (n_iter, discount, epsilon, epsilon_decay, alpha) in tqdm(enumerate(product(n_iters, discounts, epsilons, epsilon_decays, alphas)),
                                                                       total=len(n_iters)*len(discounts)*len(epsilons)*len(epsilon_decays)*len(alphas)):
        q = QLearning(P, R, gamma=discount, alpha=alpha, epsilon=epsilon, epsilon_decay=epsilon_decay, n_iter=n_iter)
        q.run()
        reward = test_policy(P, R, q.policy)
        print("{}: {}".format(idx, reward))

        st = q.run_stats
        train_rews = [s['Reward'] for s in st]

        info = [n_iter, alpha, epsilon, epsilon_decay, reward, q.time, q.policy, q.V, train_rews]
        df_length = len(q_df)
        q_df.loc[df_length] = info

    return q_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--q_learning', action='store_true')
    parser.add_argument('--large', action='store_true')
    args = parser.parse_args()

    # TODO: HOW to define the probability `p` for wild fire and also the rewards `r1`, `r2`
    if args.large:
        P, R = forest(S=625, r1=100, r2=15, p=0.01)     # Large states
        version = "large"
    else:
        P, R = forest(S=8, r1=12, r2=6, p=0.1)         # Small states
        version = "small"

    if args.q_learning:
        print(f"Running Q-Learning for {version.upper()} Forest Management...")

        discounts = [0.75, 0.9, 0.999]
        iters = [1e3, 1e4, 1e5]
        alphas = [0.01, 0.1]
        epsilons = [1.0, 0.75, 0.5]
        epsilon_decay_rates = [0.99, 0.999, 0.9999]

        q_df = train_q_learning(P, R, discounts=discounts, alphas=alphas, epsilons=epsilons, epsilon_decays=epsilon_decay_rates, n_iters=iters)
        q_df.to_csv(f"{OUTPUT_DIR}/forest_{version}_Qlearning_policies.csv")

        print(f'Mean iteration = {q_df.groupby("Iteration").mean()}')
        print(f'Mean alphas = {q_df.groupby("Alpha").mean()}')
        print(f'Mean epsilon = {q_df.groupby("Epsilon").mean()}')
        print(f'Mean epsilon_decay = {q_df.groupby("Epsilon Decay").mean()}')

        best_policy = q_df.loc[q_df["Reward"] == q_df["Reward"].max()]
        print(f'Best policy is {best_policy.index.item()} with reward = {best_policy["Reward"]}')
        print(f"COMPLETE {version.upper()} FOREST MANAGEMENT FOR Q-LEARNING!!!")
    else:
        print(f"Running VI and PI for {version.upper()} Forest Management...")

        discounts = [0.5, 0.9, 0.95, 0.999]
        epsilons = [1e-3, 1e-6, 1e-9, 1e-12, 1e-15]

        vi_df = train_vi(P, R, discounts=discounts, epsilons=epsilons, max_iter=1e6)
        pi_df = train_pi(P, R, discounts=discounts, epsilons=epsilons, max_iter=1e6)

        vi_df.to_csv(f"{OUTPUT_DIR}/forest_{version}_vi_policies.csv")
        pi_df.to_csv(f"{OUTPUT_DIR}/forest_{version}_pi_policies.csv")

        print(f"COMPLETE {version.upper()} FOREST MANAGEMENT FOR VI & PI!!!")




