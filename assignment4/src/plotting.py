import os
import json

from os import makedirs
from tqdm import tqdm

from matplotlib import colors
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

from run_frozen_lake import value_iteration, policy_iteration, q_learning, test_policy
from gym.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv
from mdptoolbox.example import forest
from hiive.mdptoolbox.mdp import ValueIteration, PolicyIteration, PolicyIterationModified, QLearning


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def map_discretize(the_map):
    size = len(the_map)
    dis_map = np.zeros((size, size))
    for i, row in enumerate(the_map):
        for j, loc in enumerate(row):
            if loc == "S":
                dis_map[i, j] = 0
            elif loc == "F":
                dis_map[i, j] = 0
            elif loc == "H":
                dis_map[i, j] = -1
            elif loc == "G":
                dis_map[i, j] = 1
    return dis_map
def policy_numpy(policy):
    size = int(np.sqrt(len(policy)))
    pol = np.asarray(policy)
    pol = pol.reshape((size, size))
    return pol


def convert_dict_to_dicts(the_dict):
    # return for discount
    discount_rewards = {}
    discount_iterations = {}
    discount_times = {}

    for disc in the_dict:
        discount_rewards[disc] = []
        discount_iterations[disc] = []
        discount_times[disc] = []

        for eps in the_dict[disc]:
            discount_rewards[disc].append(the_dict[disc][eps]['mean_reward'])
            discount_iterations[disc].append(the_dict[disc][eps]['iteration'])
            discount_times[disc].append(the_dict[disc][eps]['time_spent'])

    # return for epsilon
    epsilon_rewards = {}
    epsilon_iterations = {}
    epsilon_times = {}
    for eps in the_dict["0.5"]:
        epsilon_rewards[eps] = []
        epsilon_iterations[eps] = []
        epsilon_times[eps] = []

        for disc in the_dict:
            epsilon_rewards[eps].append(the_dict[disc][eps]['mean_reward'])
            epsilon_iterations[eps].append(the_dict[disc][eps]['iteration'])
            epsilon_times[eps].append(the_dict[disc][eps]['time_spent'])

    return discount_rewards, discount_iterations, discount_times, epsilon_rewards, epsilon_iterations, epsilon_times
def convert_dict_to_df(the_dict):
    the_df = pd.DataFrame(columns=["Discount Rate", "Training Episodes", "Learning Rate",
                                   "Decay Rate", "Reward", "Time Spent"])
    for dis in tqdm(the_dict):
        for episode in the_dict[dis]:
            for lr in the_dict[dis][episode]:
                for epsilon in the_dict[dis][episode][lr]:
                    for epsilon_dr in the_dict[dis][episode][lr][epsilon]:
                        rew = the_dict[dis][episode][lr][epsilon][epsilon_dr]["mean_reward"]
                        iteration = the_dict[dis][episode][lr][epsilon][epsilon_dr]["iteration"]
                        time_spent = the_dict[dis][episode][lr][epsilon][epsilon_dr]["time_spent"]
                        dic = {"Discount Rate": dis,
                               "Training Episodes": episode,
                               "Learning Rate": lr,
                               "Epsilon": epsilon,
                               "Decay Rate": epsilon_dr,
                               "Reward": rew,
                               "Iteration": iteration,
                               "Time": time_spent}
                        the_df = the_df.append(dic, ignore_index=True)
    return the_df


def plot_policy_frozen_lake(method, method_version, grid_size, policy, iteration=None):
    fig, ax = plt.subplots(figsize=(9.6, 9.6), dpi=200)

    map_name = str(grid_size) + "x" + str(grid_size)
    data = map_discretize(MAPS[map_name])
    np_pol = policy_numpy(policy)
    plt.imshow(data, interpolation="nearest")

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    for i in range(np_pol[0].size):
        for j in range(np_pol[0].size):
            arrow = '\u2190'
            if np_pol[i, j] == 1:
                arrow = '\u2193'
            elif np_pol[i, j] == 2:
                arrow = '\u2192'
            elif np_pol[i, j] == 3:
                arrow = '\u2191'
            plt.text(j, i, arrow, ha="center", va="center", color="w", fontsize=18)

    if not os.path.exists(f"{OUTPUT_DIR}/{method_version}"):
        makedirs(f"{OUTPUT_DIR}/{method_version}", exist_ok=True)

    if iteration is not None:
        plt.savefig(f"{OUTPUT_DIR}/{method_version}/{method}-policy-iteration-{iteration}.png")
        plt.savefig(f"{OUTPUT_DIR}/{method_version}/{method}-policy-iteration-{iteration}.pdf")
    else:
        plt.savefig(f"{OUTPUT_DIR}/{method_version}/{method}-policy.png")
        plt.savefig(f"{OUTPUT_DIR}/{method_version}/{method}-policy.pdf")

    plt.close()
def plot_policy_forest_management(method, method_version, map_size, policy, iteration, groupby_policies, cond_values, rewards, ylabel):
    cmap = colors.ListedColormap(['green', 'brown'])
    fig, ax = plt.subplots(figsize=(16, 6), dpi=200)
    plt.title(f"Policy - {method} - Forest Management ({method_version.capitalize()}) - Brown = Cut, Green = Wait", fontsize=18)
    plt.xticks(fontsize=18)
    plt.xlabel('State', fontsize=18)
    plt.ylabel(ylabel, fontsize=18)

    if "Discount" not in ylabel:
        cond_values = [f"{cond_value} ({round(reward, 2)})" for cond_value, reward in zip(cond_values, rewards)]
    else:
        plt.ylabel("Discount", fontsize=18)

    # TODO: Disable this block if the y-axis is messed up
    if len(groupby_policies) < 5:
        empty_labels = [""] * len(cond_values)
        y_labels = [None] * (len(cond_values) + len(empty_labels))
        y_labels[::2] = cond_values
        y_labels[1::2] = empty_labels
    else:
        y_labels = cond_values

    ax.pcolor(np.array(groupby_policies), cmap=cmap, edgecolors='k', linewidths=0)
    ax.set_yticklabels(y_labels, fontsize=18)
    ax.tick_params(left=False)  # remove the ticks

    if not os.path.exists(f"{OUTPUT_DIR}/{method_version}"):
        makedirs(f"{OUTPUT_DIR}/{method_version}", exist_ok=True)

    if iteration is not None:
        fig.savefig(f"{OUTPUT_DIR}/{method_version}/{method}-policy-{ylabel}-iteration-{iteration}.png")
        fig.savefig(f"{OUTPUT_DIR}/{method_version}/{method}-policy-{ylabel}-iteration-{iteration}.pdf")
    else:
        fig.savefig(f"{OUTPUT_DIR}/{method_version}/{method}-policy-{ylabel}.png")
        fig.savefig(f"{OUTPUT_DIR}/{method_version}/{method}-policy-{ylabel}.pdf")

    plt.close(fig)
def plot_best_policy_forest_management(method, method_version, map_size, policy, iteration):
    cmap = colors.ListedColormap(['green', 'brown'])
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    plt.title(f"Policy - {method} - Forest Management ({method_version.capitalize()}) - Brown = Cut, Green = Wait", fontsize=18)
    plt.xticks(fontsize=18)
    plt.xlabel('State', fontsize=18)

    ax.pcolor(np.array([policy]), cmap=cmap, edgecolors='k', linewidths=0)
    ax.set_yticklabels([], fontsize=18)
    ax.tick_params(left=False)  # remove the ticks

    if not os.path.exists(f"{OUTPUT_DIR}/{method_version}"):
        makedirs(f"{OUTPUT_DIR}/{method_version}", exist_ok=True)

    if iteration is not None:
        fig.savefig(f"{OUTPUT_DIR}/{method_version}/{method}-policy-best-iteration-{iteration}.png")
        fig.savefig(f"{OUTPUT_DIR}/{method_version}/{method}-policy-best-iteration-{iteration}.pdf")
    else:
        fig.savefig(f"{OUTPUT_DIR}/{method_version}/{method}-policy-best.png")
        fig.savefig(f"{OUTPUT_DIR}/{method_version}/{method}-policy-best.pdf")

    plt.close(fig)


def plot_hyperparameters_tuning(method, method_version, data, data_large=None, value="Reward", sizes=[8], variable="Discount Rate", log=False):
    fig = plt.figure(figsize=(10.2, 6.4), dpi=200)
    # fig.tight_layout()

    if len(sizes) > 1:
        title = "Average {} on ({}x{} and {}x{}) Frozen Lake".format(value, sizes[0], sizes[0], sizes[1], sizes[1])
    else:
        title = "Average {} on {}x{} Frozen Lake".format(value, sizes[0], sizes[0])

    the_val = value
    value = "Average {}".format(the_val)
    # val_type = "Type of {}".format(the_val)
    val_type = "State size"
    the_df = pd.DataFrame(columns=[variable, value, val_type])

    # For VI and PI
    if isinstance(data, dict):
        for k, v in data.items():
            for val in v:
                if not log:
                    dic = {variable: float(k), value: float(val), val_type: "Small (8x8)"}
                else:
                    dic = {variable: np.log10(float(k)), value: float(val), val_type: "Small (8x8)"}
                the_df = the_df.append(dic, ignore_index=True)
            # DO NOT USE MAX
            # if not log:
            #     dic = {variable: float(k), value: float(max(v)), val_type: "Max"}
            # else:
            #     dic = {variable: np.log10(float(k)), value: float(max(v)), val_type: "Max"}
            # the_df = the_df.append(dic, ignore_index=True)

        for k, v in data_large.items():
            for val in v:
                if not log:
                    dic = {variable: float(k), value: float(val), val_type: "Large (64x64)"}
                else:
                    dic = {variable: np.log10(float(k)), value: float(val), val_type: "Large (64x64)"}
                the_df = the_df.append(dic, ignore_index=True)

        the_df = the_df.groupby([variable, val_type], as_index=False)[value].mean()

    # For Q-learning
    else:
        groupby_column = "Discount Rate"
        if variable == "Epsilon Value":
            groupby_column = "Epsilon"
        elif variable == "Decay Rate on Epsilon Value":
            groupby_column = "Decay Rate"

        df_mean = data.groupby([groupby_column], as_index=False)[the_val].mean()
        df_mean[val_type] = ["Small (8x8)"] * len(df_mean)

        df_large_mean = data_large.groupby([groupby_column], as_index=False)[the_val].mean()
        df_large_mean[val_type] = ["Large (25x25)"] * len(df_large_mean)

        # DO NOT USE MAX
        # df_max = data.groupby([groupby_column], as_index=False)[the_val].max()
        # df_max[val_type] = ["Max"] * len(df_max)

        the_df = pd.concat([df_mean, df_large_mean])
        the_df = the_df.rename(columns={the_val: value, 'Epsilon': variable, 'Decay Rate': variable})

        # Phat: Since Q-learning's epsilons are not in log scale
        # if log:
        #     the_df[variable] = the_df[variable].apply(lambda x: np.log10(float(x)))

    sns.lineplot(x=variable, y=value, hue=val_type, palette="Set1", markers=True, data=the_df).set(title=title)     # style=val_type,
    plt.title(title, fontsize=18)
    plt.xlabel(variable, fontsize=18)
    plt.ylabel(value, fontsize=18)
    plt.legend(fontsize=18)

    plt.xticks(fontsize=18)
    if the_val == "Reward":
        plt.yticks(ticks=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.8, 1.0],
                   labels=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.8, 1.0], fontsize=16)
    else:
        plt.yticks(fontsize=18)

    plt.grid()

    if not os.path.exists(f"{OUTPUT_DIR}/{method_version}"):
        makedirs(f"{OUTPUT_DIR}/{method_version}", exist_ok=True)

    fig.savefig(f"{OUTPUT_DIR}/{method_version}/{method}-{title}-{variable}.pdf")

    plt.close(fig)
def plot_tuning_frozen_lake_vi_pi(grid_sizes=[8], only_plot_best_policy=False):
    if len(grid_sizes) == 1:
        version = "small" if grid_sizes[0] < 16 else "large"

        vi_file_path = f"{INPUT_DIR}/frozen_lake_{version}_vi_policies.json"
        pi_file_path = f"{INPUT_DIR}/frozen_lake_{version}_pi_policies.json"

        with open(vi_file_path, "r") as input_file:
            vi_dict = json.load(input_file)
        with open(pi_file_path, "r") as input_file:
            pi_dict = json.load(input_file)

        if version == "small":
            pol = vi_dict["0.75"]["1e-06"]['policy']
            iteration = vi_dict["0.75"]["1e-06"]['iteration']
        else:
            pol = vi_dict["0.99"]["1e-06"]['policy']
            iteration = vi_dict["0.99"]["1e-06"]['iteration']
        plot_policy_frozen_lake("VI", version, grid_sizes[0], pol, iteration)

        if version == "small":
            pol = pi_dict["0.5"]["1e-06"]['policy']
            iteration = pi_dict["0.5"]["1e-06"]['iteration']
        else:
            pol = pi_dict["0.99"]["1e-06"]['policy']
            iteration = pi_dict["0.99"]["1e-06"]['iteration']
        plot_policy_frozen_lake("PI", version, grid_sizes[0], pol, iteration)

        if only_plot_best_policy:
            return

        vi = convert_dict_to_dicts(vi_dict)
        pi = convert_dict_to_dicts(pi_dict)

        vi_large, pi_large = None, None
    else:
        version = "merged"

        vi_file_path = f"{INPUT_DIR}/frozen_lake_small_vi_policies.json"
        with open(vi_file_path, "r") as input_file:
            vi_dict = json.load(input_file)
        with open(vi_file_path.replace("small", "large"), "r") as input_file:
            vi_dict_large = json.load(input_file)

        vi = convert_dict_to_dicts(vi_dict)
        vi_large = convert_dict_to_dicts(vi_dict_large)

        pi_file_path = f"{INPUT_DIR}/frozen_lake_small_pi_policies.json"
        with open(pi_file_path, "r") as input_file:
            pi_dict = json.load(input_file)
        with open(pi_file_path.replace("small", "large"), "r") as input_file:
            pi_dict_large = json.load(input_file)

        pi = convert_dict_to_dicts(pi_dict)
        pi_large = convert_dict_to_dicts(pi_dict_large)

    plot_hyperparameters_tuning(method="VI", method_version=version, data=vi[0], data_large=vi_large[0], value="Reward", sizes=grid_sizes)
    plot_hyperparameters_tuning(method="VI", method_version=version, data=vi[1], data_large=vi_large[1], value="Iteration", sizes=grid_sizes)
    plot_hyperparameters_tuning(method="VI", method_version=version, data=vi[2], data_large=vi_large[2], value="Time", sizes=grid_sizes)
    plot_hyperparameters_tuning(method="VI", method_version=version, data=vi[3], data_large=vi_large[3], value="Reward", sizes=grid_sizes, variable="Log Epsilon Value", log=True)
    plot_hyperparameters_tuning(method="VI", method_version=version, data=vi[4], data_large=vi_large[4], value="Iteration", sizes=grid_sizes, variable="Log Epsilon Value", log=True)
    plot_hyperparameters_tuning(method="VI", method_version=version, data=vi[5], data_large=vi_large[5], value="Time", sizes=grid_sizes, variable="Log Epsilon Value", log=True)

    plot_hyperparameters_tuning(method="PI", method_version=version, data=pi[0], data_large=pi_large[0], value="Reward", sizes=grid_sizes)
    plot_hyperparameters_tuning(method="PI", method_version=version, data=pi[1], data_large=pi_large[1], value="Iteration", sizes=grid_sizes)
    plot_hyperparameters_tuning(method="PI", method_version=version, data=pi[2], data_large=pi_large[2], value="Time", sizes=grid_sizes)
    plot_hyperparameters_tuning(method="PI", method_version=version, data=pi[3], data_large=pi_large[3], value="Reward", sizes=grid_sizes, variable="Log Epsilon Value", log=True)
    plot_hyperparameters_tuning(method="PI", method_version=version, data=pi[4], data_large=pi_large[4], value="Iteration", sizes=grid_sizes, variable="Log Epsilon Value", log=True)
    plot_hyperparameters_tuning(method="PI", method_version=version, data=pi[5], data_large=pi_large[5], value="Time", sizes=grid_sizes, variable="Log Epsilon Value", log=True)
def plot_tuning_frozen_lake_q_learning(grid_sizes=[8], only_plot_best_policy=False):
    if len(grid_sizes) == 1:
        version = "small" if grid_sizes[0] < 16 else "large"

        q_file_path = f"{INPUT_DIR}/frozen_lake_{version}_Qlearning_policies.json"

        with open(q_file_path, "r") as input_file:
            q_dict = json.load(input_file)

        if version == "small":
            pol = q_dict["0.9"]["10000.0"]["0.1"]['0.5']['0.9999']['policy']
            iteration = q_dict["0.9"]["10000.0"]["0.1"]['0.5']['0.9999']['iteration']
        else:
            pol = q_dict["0.9"]["10000.0"]["0.1"]['0.5']['0.9999']['policy']
            iteration = q_dict["0.9"]["10000.0"]["0.1"]['0.5']['0.9999']['iteration']

        plot_policy_frozen_lake("Q-learning", version, grid_sizes[0], pol, iteration)

        if only_plot_best_policy:
            return

        q_df = convert_dict_to_df(q_dict)
        q_large_df = None
    else:
        version = "merged"

        q_file_path = f"{INPUT_DIR}/frozen_lake_small_Qlearning_policies.json"
        with open(q_file_path, "r") as input_file:
            q_dict = json.load(input_file)
        with open(q_file_path.replace("small", "large"), "r") as input_file:
            q_dict_large = json.load(input_file)

        q_df = convert_dict_to_df(q_dict)
        q_large_df = convert_dict_to_df(q_dict_large)

    plot_hyperparameters_tuning(method="Q-learning", method_version=version, data=q_df, data_large=q_large_df, value="Reward", sizes=grid_sizes)
    plot_hyperparameters_tuning(method="Q-learning", method_version=version, data=q_df, data_large=q_large_df, value="Iteration", sizes=grid_sizes)
    plot_hyperparameters_tuning(method="Q-learning", method_version=version, data=q_df, data_large=q_large_df, value="Time", sizes=grid_sizes)

    plot_hyperparameters_tuning(method="Q-learning", method_version=version, data=q_df, data_large=q_large_df, value="Reward", sizes=grid_sizes, variable="Decay Rate on Epsilon Value")
    plot_hyperparameters_tuning(method="Q-learning", method_version=version, data=q_df, data_large=q_large_df, value="Iteration", sizes=grid_sizes, variable="Decay Rate on Epsilon Value")
    plot_hyperparameters_tuning(method="Q-learning", method_version=version, data=q_df, data_large=q_large_df, value="Time", sizes=grid_sizes, variable="Decay Rate on Epsilon Value")

    plot_hyperparameters_tuning(method="Q-learning", method_version=version, data=q_df, data_large=q_large_df, value="Reward", sizes=grid_sizes, variable="Epsilon Value", log=True)
    plot_hyperparameters_tuning(method="Q-learning", method_version=version, data=q_df, data_large=q_large_df, value="Iteration", sizes=grid_sizes, variable="Epsilon Value", log=True)
    plot_hyperparameters_tuning(method="Q-learning", method_version=version, data=q_df, data_large=q_large_df, value="Time", sizes=grid_sizes, variable="Epsilon Value", log=True)
def plot_tuning_forest_management_vi_pi(map_size=8):
    version = "small" if map_size < 256 else "large"
    vi_file_path = f"{INPUT_DIR}/forest_{version}_vi_policies.csv"
    pi_file_path = f"{INPUT_DIR}/forest_{version}_pi_policies.csv"

    vi_df = pd.read_csv(vi_file_path)
    pi_df = pd.read_csv(pi_file_path)

    for method, df, in zip(["VI", "PI"], [vi_df, pi_df]):
        best_policy = df[df["Reward"] == df["Reward"].max()]
        print(best_policy)

        best_policy = df[df["Reward"] == df["Reward"].max()]
        pol = eval(best_policy["Policy"].item())
        iteration = int(df[df["Reward"] == df["Reward"].max()]["Iteration"].item())

        best_policies_groupby_df = df.loc[df.groupby(["Discount"], as_index=False)["Reward"].idxmax()["Reward"]]
        policies_on_discount = [eval(x) for x in best_policies_groupby_df["Policy"].tolist()]
        discounts = best_policies_groupby_df["Discount"].tolist()
        dis_rewards = best_policies_groupby_df["Reward"].tolist()

        best_policies_groupby_df = df.loc[df.groupby(["Epsilon"], as_index=False)["Reward"].idxmax()["Reward"]]
        policies_on_epsilon = [eval(x) for x in best_policies_groupby_df["Policy"].tolist()]
        epsilons = best_policies_groupby_df["Epsilon"].tolist()
        eps_rewards = best_policies_groupby_df["Reward"].tolist()

        plot_policy_forest_management(method, version, map_size, pol, iteration, policies_on_discount, discounts, dis_rewards, ylabel="Discount (Reward)")
        plot_policy_forest_management(method, version, map_size, pol, iteration, policies_on_epsilon, epsilons, eps_rewards, ylabel="Epsilon (Reward)")
def plot_tuning_forest_management_q_learning(map_size=8):
    version = "small" if map_size < 256 else "large"
    q_file_path = f"{INPUT_DIR}/forest_{version}_Qlearning_policies.csv"
    q_df = pd.read_csv(q_file_path)

    # SMALL: Iteration / Alpha / Epsilon / Epsilon Decay = 1000000 / 0.1 / 0.75 / 0.9999
    best_policy = q_df[q_df["Reward"] == q_df["Reward"].max()]
    pol = eval(best_policy["Policy"].item())
    iteration = int(q_df[q_df["Reward"] == q_df["Reward"].max()]["Iteration"].item())

    best_policies_groupby_df = q_df.loc[q_df.groupby(["Epsilon"], as_index=False)["Reward"].idxmax()["Reward"]]
    policies_on_epsilon = [list(eval(x)) for x in best_policies_groupby_df["Policy"].tolist()]
    epsilons = best_policies_groupby_df["Epsilon"].tolist()
    eps_rewards = best_policies_groupby_df["Reward"].tolist()

    best_policies_groupby_df = q_df.loc[q_df.groupby(["Epsilon Decay"], as_index=False)["Reward"].idxmax()["Reward"]]
    policies_on_epsilon_decay = [list(eval(x)) for x in best_policies_groupby_df["Policy"].tolist()]
    epsilon_decays = best_policies_groupby_df["Epsilon Decay"].tolist()
    eps_decay_rewards = best_policies_groupby_df["Reward"].tolist()

    plot_policy_forest_management("Q-learning", version, map_size, pol, iteration, policies_on_epsilon, epsilons, eps_rewards, ylabel="Epsilon (Reward)")
    plot_policy_forest_management("Q-learning", version, map_size, pol, iteration, policies_on_epsilon_decay, epsilon_decays, eps_decay_rewards, ylabel="Epsilon Decay (Reward)")

    plot_best_policy_forest_management("Q-learning", version, map_size, pol, iteration)


def plot_convergence_frozen_lake_reward_time_policies_vs_iterations(grid_size, method, discount, epsilon, episode=10000, alpha=0.1, epsilon_dr=0.9999, min_epsilon=0.01):

    env = FrozenLakeEnv(desc=MAPS[f"{grid_size}x{grid_size}"])
    version = "small" if grid_size < 16 else "large"

    if method == "VI":
        policy, solve_iter, solve_time, plot_data = value_iteration(env, discount=discount, epsilon=epsilon, best_setting=True)
    elif method == "PI":
        policy, solve_iter, solve_time, plot_data = policy_iteration(env, discount=discount, epsilon=epsilon, best_setting=True)
    # Q-learning
    else:
        policy, solve_iter, solve_time, q_table, rewards, plot_data = q_learning(env, discount, episode, alpha, epsilon, epsilon_dr, min_epsilon, best_setting=True)

    fig, ax = plt.subplots(figsize=(12.8, 6.4), dpi=200)

    iterations = list(plot_data.keys())
    rewards = [plot_data[i]["reward"] for i in iterations]      # Clarification: This is mean value
    policies = [plot_data[i]["policy"] for i in iterations]
    times = [plot_data[i]["time"] for i in iterations]
    max_deltas = [plot_data[i]["max_delta"] for i in iterations]

    ax.plot(iterations, max_deltas, color="red")
    ax.set_ylabel("Max Delta", color="red", fontsize=18)
    ax.set_xlabel("Iterations", fontsize=18)

    ax2 = ax.twinx()
    ax2.plot(iterations, rewards, color="green")
    ax2.set_ylabel("Mean Value", color="green", fontsize=18)

    plt.title(f"Convergence (Reward vs Iteration) - {method} - Frozen Lake ({version.capitalize()})", fontsize=18)
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Mean Value", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.grid()

    fig.savefig(f"{OUTPUT_DIR}/{version}/{method}-FrozenLake-{version.capitalize()}-Convergence.png")
    fig.savefig(f"{OUTPUT_DIR}/{version}/{method}-FrozenLake-{version.capitalize()}-Convergence.pdf")
    plt.close(fig)

    fig = plt.figure(figsize=(12.8, 6.4), dpi=200)
    plt.plot(iterations, times)
    plt.ylabel("Time (seconds)", fontsize=18)
    plt.xlabel("Iterations", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(f"Training time - {method} - Frozen Lake ({version.capitalize()})", fontsize=18)
    plt.savefig(f"{OUTPUT_DIR}/{version}/{method}-FrozenLake-{version.capitalize()}-Runtime.png")
    plt.savefig(f"{OUTPUT_DIR}/{version}/{method}-FrozenLake-{version.capitalize()}-Runtime.pdf")
    plt.close(fig)
def plot_convergence_forest_management_reward_time_policies_vs_iterations(map_size, method, discount, epsilon, episode, alpha=0.1, epsilon_dr=0.9999):
    if map_size < 256:
        version = "small"
        P, R = forest(S=8, r1=12, r2=6, p=0.1)  # Small states
    else:
        version = "large"
        P, R = forest(S=625, r1=100, r2=15, p=0.01)  # Large states

    if method == "VI":
        vi = ValueIteration(P, R, gamma=discount, epsilon=epsilon, max_iter=episode, run_stat_frequency=1)
        vi.run()
        print(f"Best policy: {vi.policy}")
        plot_data = vi.run_stats
    elif method == "PI":
        pi = PolicyIteration(P, R, gamma=discount, max_iter=episode, run_stat_frequency=1)
        pi.run()
        print(f"Best policy: {pi.policy}")
        plot_data = pi.run_stats
    # Q-learning
    else:
        q = QLearning(P, R, gamma=discount, alpha=alpha, epsilon=epsilon, epsilon_decay=epsilon_dr, n_iter=episode, run_stat_frequency=1)
        q.run()
        print(f"Best policy: {q.policy}")
        plot_data = q.run_stats

    fig, ax = plt.subplots(figsize=(12.8, 6.4), dpi=200)

    mean_val = [x["Mean V"] for x in plot_data]
    times = [x["Time"] for x in plot_data]
    max_deltas = [x["Error"] for x in plot_data]
    iterations = np.arange(1, len(mean_val) + 1, 1)

    ax.plot(iterations, max_deltas, color="red")
    ax.set_ylabel("Max Delta", color="red", fontsize=18)
    ax.set_xlabel("Iterations", fontsize=18)

    ax2 = ax.twinx()
    ax2.plot(iterations, mean_val, color="green")
    ax2.set_ylabel("Mean Value", color="green", fontsize=18)

    plt.title(f"Convergence (Reward vs Iteration) - {method} - Forest Management ({version.capitalize()})", fontsize=18)
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Mean Value", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.grid()

    fig.savefig(f"{OUTPUT_DIR}/{version}/{method}-ForestManagement-{version.capitalize()}-Convergence.png")
    fig.savefig(f"{OUTPUT_DIR}/{version}/{method}-ForestManagement-{version.capitalize()}-Convergence.pdf")
    plt.close(fig)

    fig = plt.figure(figsize=(12.8, 6.4), dpi=200)
    plt.plot(iterations, times)
    plt.ylabel("Time (seconds)", fontsize=18)
    plt.xlabel("Iterations", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(f"Training time - {method} - Forest Management ({version.capitalize()})", fontsize=18)
    plt.savefig(f"{OUTPUT_DIR}/{version}/{method}-ForestManagement-{version.capitalize()}-Runtime.png")
    plt.savefig(f"{OUTPUT_DIR}/{version}/{method}-ForestManagement-{version.capitalize()}-Runtime.pdf")
    plt.close(fig)


if __name__ == '__main__':
    INPUT_DIR = "../outputs/trained_policies"

    if not os.path.exists("../outputs/maps.json"):
        print("maps.json not found")
        exit()

    with open("../outputs/maps.json", "r") as input_file:
        MAPS = json.load(input_file)

    # ------- FROZEN LAKE: PLOT TUNING DATA -------
    OUTPUT_DIR = "../outputs/plots/FrozenLake"
    makedirs(OUTPUT_DIR, exist_ok=True)

    # ONLY FOR BEST POLICY PLOTS
    plot_tuning_frozen_lake_vi_pi(grid_sizes=[8], only_plot_best_policy=True)
    plot_tuning_frozen_lake_vi_pi(grid_sizes=[25], only_plot_best_policy=True)
    plot_tuning_frozen_lake_q_learning(grid_sizes=[8], only_plot_best_policy=True)
    plot_tuning_frozen_lake_q_learning(grid_sizes=[25], only_plot_best_policy=True)

    # TUNING PLOTS
    plot_tuning_frozen_lake_vi_pi(grid_sizes=[8, 25])
    plot_tuning_frozen_lake_q_learning(grid_sizes=[8, 25])

    # ------- FROZEN LAKE: PLOT CONVERGENCE DATA (REWARD/STATE VALUE, TIME VS ITERATIONS) ------- IN PROGRESS
    plot_convergence_frozen_lake_reward_time_policies_vs_iterations(grid_size=8, method="VI", discount=0.75, epsilon=1e-6)
    plot_convergence_frozen_lake_reward_time_policies_vs_iterations(grid_size=8, method="PI", discount=0.5, epsilon=1e-6)
    plot_convergence_frozen_lake_reward_time_policies_vs_iterations(grid_size=8, method="Q-learning", discount=0.9, epsilon=0.5, episode=1000000, alpha=0.1, epsilon_dr=0.9999)     # ~45 minutes

    plot_convergence_frozen_lake_reward_time_policies_vs_iterations(grid_size=25, method="VI", discount=0.99, epsilon=1e-6)
    plot_convergence_frozen_lake_reward_time_policies_vs_iterations(grid_size=25, method="PI", discount=0.99, epsilon=1e-6)
    plot_convergence_frozen_lake_reward_time_policies_vs_iterations(grid_size=25, method="Q-learning", discount=0.9, epsilon=0.5, episode=10000, alpha=0.1, epsilon_dr=0.9999)

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # ------- FOREST: PLOT TUNING DATA -------
    OUTPUT_DIR = "../outputs/plots/ForestManagement"
    makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------ SMALL

    plot_tuning_forest_management_vi_pi(map_size=8)
    plot_tuning_forest_management_q_learning(map_size=8)

    plot_convergence_forest_management_reward_time_policies_vs_iterations(map_size=8, method="VI", discount=0.9, epsilon=1e-12, episode=100)  # Discount: 0.5 --> 0.9, 11 iter
    plot_convergence_forest_management_reward_time_policies_vs_iterations(map_size=8, method="PI", discount=0.9, epsilon=1e-15, episode=100)  # Discount: 0.5 --> 0.9, 5 iter
    plot_convergence_forest_management_reward_time_policies_vs_iterations(map_size=8, method="Q-learning", discount=0.9, epsilon=0.5, episode=10000000, alpha=0.01, epsilon_dr=0.999)

    # ------------------------------------------------------------------ LARGE

    plot_tuning_forest_management_vi_pi(map_size=625)

    # NO TUNING --> DO NOT HAVE DATA TO PLOT
    # plot_tuning_forest_management_q_learning(map_size=625)

    plot_convergence_forest_management_reward_time_policies_vs_iterations(map_size=625, method="VI", discount=0.9, epsilon=1e-6, episode=100)    # Discount: 0.5 --> 0.9, 27 iter
    plot_convergence_forest_management_reward_time_policies_vs_iterations(map_size=625, method="PI", discount=0.9, epsilon=0.001, episode=100)    # Discount: 0.5 --> 0.9, 9 iter

    # NO TUNING --> USE HYPERPARAMETERS FROM SMALL FOREST
    plot_convergence_forest_management_reward_time_policies_vs_iterations(map_size=625, method="Q-learning", discount=0.9, epsilon=0.75, episode=10000000, alpha=0.1, epsilon_dr=0.9999)


