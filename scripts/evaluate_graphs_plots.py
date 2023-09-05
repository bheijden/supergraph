import os
import json
import hashlib
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import supergraph.open_colors as oc


def export_legend(fig, legend, expand=None):
    expand = [-5, -5, 5, 5] if expand is None else expand
    # fig = legend.figure
    # fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    return bbox
    # fig.savefig(filename, dpi="figure", bbox_inches=bbox)


if __name__ == "__main__":
    # Setup sns plotting
    import seaborn as sns
    sns.set(style="whitegrid", font_scale=1.5)
    # cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # cmap = cmap[1:] + cmap[:1]
    # for key, value in plt.rcParams.items():
        # if "font.size" not in key:
            # continue
        # print(key, value)
    scaling = 5
    MUST_BREAK = False
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 6 * scaling
    plt.rcParams['legend.fontsize'] = 5 * scaling
    plt.rcParams['font.size'] = 7 * scaling
    plt.rcParams['xtick.labelsize'] = 5 * scaling
    plt.rcParams['ytick.labelsize'] = 5 * scaling
    plt.rcParams['xtick.major.pad'] = -0.0 * scaling
    plt.rcParams['ytick.major.pad'] = -0.0 * scaling
    plt.rcParams['lines.linewidth'] = 0.65 * scaling
    plt.rcParams['lines.markersize'] = 4.0 * scaling
    plt.rcParams['axes.xmargin'] = 0.0
    plt.rcParams['axes.ymargin'] = 0.0
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # Rescale figures
    width_points = 245.71811 * scaling
    width_inches = width_points / plt.rcParams['figure.dpi']
    default_figsize = plt.rcParams['figure.figsize']
    rescaled_width = width_inches
    rescaled_height = width_inches * default_figsize[1] / default_figsize[0]
    rescaled_figsize = [rescaled_width, rescaled_height]
    fullwidth_figsize = [c*s for c, s in zip([1, 0.52], rescaled_figsize)]
    thirdwidth_figsize = [c * s for c, s in zip([1 / 3, 0.5], rescaled_figsize)]
    sixthwidth_figsize = [c * s for c, s in zip([1 / 6, 0.5], rescaled_figsize)]
    print("Default figsize:", default_figsize)
    print("Rescaled figsize:", rescaled_figsize)
    print("Fullwidth figsize:", fullwidth_figsize)
    print("Thirdwidth figsize:", thirdwidth_figsize)
    print("Sixthwidth figsize:", sixthwidth_figsize)
    labels = {
        # Supergraph type
        "mcs": "mcs",
        "generational": "gen",
        "topological": "top",
        # Topology type
        "bidirectional-ring": "bi",
        "unidirectional-ring": "uni",
        "unirandom-ring": "rand",
        # Combination mode
        "linear": "linear",
        "power": "power",
        # Sort mode
        "optimal": "optimal",
        "arbitrary": "arbitrary",
        # Backtrack
        0: "0",
        5: "5",
        10: "10",
        15: "15",
        20: "20",
        # Pendulum
        "async": "mcs",
        "deterministic": "seq",
    }
    cscheme = {
        # Supergraph type
        "mcs": "indigo",
        "generational": "grape",
        "topological": "red",
        # Topology type
        "bidirectional-ring": "blue",
        "unidirectional-ring": "orange",
        "unirandom-ring": "green",
        # Combination mode
        "linear": "indigo",
        "power": "red",
        # Sort mode
        "arbitrary": "indigo",
        "optimal": "red",
        # Backtrack
        0: "red",
        5: "pink",
        10: "grape",
        15: "violet",
        20: "indigo",
        # Sigma
        0.0: "red",
        0.1: "pink",
        0.2: "grape",
        0.3: "violet",
        # Pendulum
        "async": "indigo",
        "deterministic": "red",
        # Environment
        "real": "indigo",
        "sim": "red",
        }
    cscheme.update({labels[k]: v for k, v in cscheme.items() if k in labels.keys()})
    ecolor, fcolor = oc.cscheme_fn(cscheme)

    import wandb
    api = wandb.Api()

    # Pendulum plots
    filters = {
        "group": {"$in": ["train-real-evaluate-2023-08-16-1520"]},
        "config.seed": {"$in": [0, 1, 2, 3, 4]},
        # "JobType": {"$in": ["deterministic", "async"]},
        "State": "finished",
    }

    # Generate a unique identifier from the filters
    filter_str = json.dumps(filters, sort_keys=True)  # Convert dict to string in a consistent manner
    hash_object = hashlib.md5(filter_str.encode())  # Use MD5 or another hashing algorithm
    filter_hash = hash_object.hexdigest()  # Get the hash as a string

    # Use the hash as the filename
    filename = "df_" + filter_hash + ".pkl"

    if all([os.path.exists(filename.replace("df_", f"df_{f}_")) for f in ["speed", "train", "eval"]]):
        # If the file already exists, simply load the dataframe from it
        print("Loading dataframe [pendulum] from file:", filename)
        df_speed = pd.read_pickle(filename.replace("df_", f"df_speed_"))
        df_train = pd.read_pickle(filename.replace("df_", f"df_train_"))
        df_eval = pd.read_pickle(filename.replace("df_", f"df_eval_"))
    else:
        # If the file doesn't exist, generate the dataframe as usual
        runs = api.runs(path="supergraph", filters=filters, per_page=1000)

        df_list_train = []
        df_list_eval = []
        df_list_speed = []
        for r in tqdm.tqdm(runs, desc="Loading data [pendulum]"):
            # Train curve
            h = r.history(samples=10000, keys=["train/ep_rew_mean", "train/total_timesteps"], x_axis="_step", pandas=True, stream="default")
            new_columns = {"job_type": r.job_type, "seed": r.config["seed"]}
            h = h.assign(**new_columns)
            df_list_train.append(h)

            # Speed curve
            h1 = r.history(samples=10000, keys=["speed/supergraph_type", "speed/fps"], x_axis="_step", pandas=True, stream="default")
            h2 = r.history(samples=10000, keys=["supergraph/supergraph_type", "supergraph/efficiency_percentage"], x_axis="_step", pandas=True, stream="default")

            # join h1 and h2 on "speed/supergraph_type" and "supergraph/supergraph_type"
            h = h1.merge(h2, left_on="speed/supergraph_type", right_on="supergraph/supergraph_type", suffixes=("_speed", "_supergraph"))
            h.drop(columns=["supergraph/supergraph_type"], inplace=True)

            new_columns = {"job_type": r.job_type, "seed": r.config["seed"]}
            h = h.assign(**new_columns)
            df_list_speed.append(h)

            # Eval curve
            new_columns = {"job_type": r.job_type, "seed": r.config["seed"], "ep_rew_mean_real": r.summary["final/real/ep_rew_mean"], "ep_rew_mean_sim": r.summary["speed/ep_rew_mean"]}
            # Create a dataframe from new_columns
            h = pd.DataFrame(new_columns, index=[0])
            df_list_eval.append(h)
        df_speed = pd.concat(df_list_speed)
        df_train = pd.concat(df_list_train)
        df_eval = pd.concat(df_list_eval)

        # Save the dataframe to a file for future use
        df_speed.to_pickle(filename.replace("df_", "df_speed_"))
        df_train.to_pickle(filename.replace("df_", "df_train_"))
        df_eval.to_pickle(filename.replace("df_", "df_eval_"))

    # Pendulum plots
    plots_pendulum = {"speed": None, "train": None, "eval": None}

    # Speed plot
    d = df_speed.copy()
    d["supergraph_type"] = d["speed/supergraph_type"].map(labels)
    d = d[d["job_type"] != "deterministic"]
    d["job_type"] = d["job_type"].map(labels)

    # remove speed/supergraph_type column
    d = d.drop(columns=["speed/supergraph_type"])

    # Group by relevant columns and compute the mean
    d = d.groupby(["supergraph_type", "seed", "job_type"]).agg({
        "speed/fps": 'mean',
        "supergraph/efficiency_percentage": 'mean'
    }).reset_index()

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
    plots_pendulum["speed"] = (fig, ax)
    sns.scatterplot(x="supergraph/efficiency_percentage", y="speed/fps", hue="supergraph_type", palette=fcolor, ax=ax, data=d, legend=True)

    # Fit a linear regression model using NumPy
    X = d["supergraph/efficiency_percentage"].values
    y = d["speed/fps"].values
    coefficients = np.polyfit(X, y, 1)
    Xh = np.linspace(0, 100, 100)  # X
    y_line = coefficients[0] * Xh + coefficients[1]

    # Plot the linear line
    ax.plot(Xh, y_line, color='black', linestyle='--', linewidth=plt.rcParams['lines.linewidth']*0.8)
    # ax.plot(Xh, y_line, label='lin', color='black', linestyle='--', linewidth=plt.rcParams['lines.linewidth']*0.8)

    # Set legend & axis labels
    ax.set_ylim(1e6, 4e6)
    ax.set_yticks([1e6, 2e6, 3e6, 4e6])
    ax.set_xlim(0, 100)
    ax.set_xlabel('efficiency (%)')
    ax.set_ylabel("speed (fps)")
    # Place the legend on the right outside of the figure
    ax.legend(handles=ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1],
              # ncol=1, loc='best', bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
              ncol=1, loc='lower right', fancybox=True, shadow=True)

    # Eval plot
    d = df_eval.copy()
    d["job_type"] = d["job_type"].map(labels)

    # Melting the DataFrame
    df_melted = d.melt(id_vars=['job_type', 'seed'], value_vars=['ep_rew_mean_real', 'ep_rew_mean_sim'], var_name='measure',
                       value_name='cost')
    df_melted["cost"] = df_melted["cost"]*-1

    # Adding a new column 'platform'
    df_melted["environment"] = df_melted['measure'].map({'ep_rew_mean_real': 'real', 'ep_rew_mean_sim': 'sim'})

    # Dropping the 'measure' column, if not needed
    df_melted.drop('measure', axis=1, inplace=True)

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
    plots_pendulum["eval"] = (fig, ax)
    sns.barplot(x="job_type", y="cost", hue="environment", data=df_melted, palette=fcolor, ax=ax)

    # Set legend & axis labels
    ax.set_ylim(0, 1400)
    ax.set_xlabel('environment')
    ax.set_ylabel("cost")
    ax.legend(handles=None, ncol=1, loc='best', fancybox=True, shadow=True)

    # Train plot
    d = df_train.copy()
    d["job_type"] = d["job_type"].map(labels)

    # Step 1: Group by 'transient/matched_percentage' and 'sigma', then interpolate 'transient/t_elapsed'
    d_grouped = d.groupby(['job_type', 'seed'])

    # Define a common 'transient/matched_percentage' range for interpolation
    common_steps = np.linspace(0, 50_000, num=101)

    # Function to interpolate 'transient/t_elapsed' for each group
    def interpolate_group(group):
        return np.interp(common_steps, group['train/total_timesteps'], group['train/ep_rew_mean'])


    # Apply the interpolation
    d_interpolated = d_grouped.apply(interpolate_group)

    # Convert the interpolated data back to a DataFrame
    d_interpolated = pd.DataFrame(d_interpolated.tolist(), index=d_interpolated.index)
    d_interpolated_reset = d_interpolated.reset_index()
    d_interpolated_reset_melted = d_interpolated_reset.melt(id_vars=['job_type', 'seed'], var_name='total_timesteps_interp',
                                                            value_name='ep_rew_mean_interp')
    d_interpolated_reset_melted["total_timesteps_interp"] = d_interpolated_reset_melted["total_timesteps_interp"].map(lambda x: common_steps[x])
    d_interpolated_reset_melted["cost"] = d_interpolated_reset_melted["ep_rew_mean_interp"]*-1

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
    plots_pendulum["train"] = (fig, ax)
    sns.lineplot(data=d_interpolated_reset_melted, x="total_timesteps_interp", y="cost", hue="job_type", ax=ax,
                 palette=fcolor, errorbar="sd")

    # Set legend & axis labels
    ax.set_ylim(0, 1400)
    ax.set_xlim(0, 50_000)
    ax.set_xlabel("steps")
    ax.set_ylabel("cost")
    ax.legend(handles=None, ncol=1, loc="best", fancybox=True, shadow=True)

    # Computational complexity plots
    filters = {
        # "group": {"$in": ["test-evaluation-2023-07-01-1706", "test-evaluation-2023-06-30-2117"]},
        "group": {"$in": ["default-32-evaluation-2023-08-08-1631", "default-64-evaluation-2023-08-08-1731"]},
        "config.topology_type": {"$in": ["bidirectional-ring", "unidirectional-ring", "unirandom-ring"]},
        "config.supergraph_type": {"$in": ["mcs"]},
        "config.sigma": {"$in": [0.0, 0.1, 0.2, 0.3]},
        "config.num_nodes": {"$in": [2, 4, 8, 16, 32, 64]},
        "config.sort_mode": {"$in": ["arbitrary", None, "null"]},
        "config.combination_mode": {"$in": ["linear", "null", None]},
        "config.backtrack": 5,  # 20
        "State": "finished",
    }
    runs = api.runs(path="supergraph", filters=filters, per_page=1000)

    # Load data
    rows_list = []
    for r in tqdm.tqdm(runs, desc="Loading data [computational complexity]"):
        if len(r.summary.keys()) == 0:
            print("skipping: ", r.config, r.summary)
            continue
        v = r.summary["final/efficiency_percentage"]
        if "final/t_elapsed" in r.summary.keys():
            t = r.summary["final/t_elapsed"]
        elif "transient/t_elapsed" in r.summary.keys():
            t = r.summary["transient/t_elapsed"]
        else:
            print("skipping: ", r.config, r.summary)
            continue
        new_row = {"topology_type": r.config["topology_type"], "num_nodes": r.config["num_nodes"], "sigma": r.config["sigma"],
                   "group": r.group, "seed": r.config["seed"], "supergraph_type": r.config["supergraph_type"],
                   "efficiency_percentage": v, "t_elapsed": t}
        rows_list.append(new_row)
    df = pd.DataFrame(rows_list)

    # Apply mapping labels on columns
    # d = df[(df["topology_type"] == "bidirectional-ring")].copy()
    d = df.copy()
    d["supergraph_type"] = d["supergraph_type"].map(labels)
    d["topology_type"] = d["topology_type"].map(labels)
    d["topology"] = d["topology_type"]
    d["nodes"] = d["num_nodes"]

    # Group by relevant columns and compute the mean
    d = d.groupby(['supergraph_type', 'topology_type', 'topology', 'nodes', 'sigma']).agg({
        't_elapsed': 'mean',
        'efficiency_percentage': 'mean'
    }).reset_index()

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fullwidth_figsize)
    plot_computational_complexity = (fig, ax)
    sns.scatterplot(x="t_elapsed", y="efficiency_percentage", hue="sigma", size="nodes",
                    style="topology", palette=fcolor, ax=ax, data=d, legend=True)

    # Set legend & axis labels
    ax.set_ylim(0, 100)
    ax.set_xlim(5e0, 1.5e3)
    ax.set_xscale("log")
    ax.set_xlabel('elapsed time (s)')
    ax.set_ylabel("efficiency (%)")
    # Place the legend on the right outside of the figure
    ax.legend(handles=ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1],
              ncol=2, loc='upper left', bbox_to_anchor=(1, 1), fancybox=True, shadow=True)

    # Transient plots
    filters = {
        # "group": {"$in": ["test-evaluation-2023-07-01-1706", "test-evaluation-2023-06-30-2117"]},
        "group": {"$in": ["default-32-evaluation-2023-08-08-1631", "default-64-evaluation-2023-08-08-1731"]},
        "config.topology_type": {"$in": ["bidirectional-ring", "unidirectional-ring", "unirandom-ring"]},
        "config.supergraph_type": {"$in": ["mcs"]},
        "config.sigma": {"$in": [0, 0.1, 0.2, 0.3]},
        # "config.sigma": {"$in": [0, 0.1]},
        "config.num_nodes": 32,
        "config.sort_mode": {"$in": ["null", "arbitrary", None]},
        "State": "finished",
    }

    # Generate a unique identifier from the filters
    filter_str = json.dumps(filters, sort_keys=True)  # Convert dict to string in a consistent manner
    hash_object = hashlib.md5(filter_str.encode())  # Use MD5 or another hashing algorithm
    filter_hash = hash_object.hexdigest()  # Get the hash as a string

    # Use the hash as the filename
    filename = "df_" + filter_hash + ".pkl"

    if os.path.exists(filename):
        # If the file already exists, simply load the dataframe from it
        print("Loading dataframe [transient] from file:", filename)
        df = pd.read_pickle(filename)
    else:
        # If the file doesn't exist, generate the dataframe as usual
        runs = api.runs(path="supergraph", filters=filters, per_page=1000)

        df_list = []
        for r in tqdm.tqdm(runs, desc="Loading data [transient]"):
            h = r.history(samples=10000,
                          keys=["transient/t_elapsed", "transient/matched_percentage", "transient/efficiency_percentage"],
                          x_axis="_step", pandas=True, stream="default")
            new_columns = {"topology_type": r.config["topology_type"], "num_nodes": r.config["num_nodes"],
                           "supergraph_type": r.config["supergraph_type"], "sigma": r.config["sigma"], "seed": r.config["seed"],
                           "group": r.group, "sort_mode": r.config["sort_mode"], "combination_mode": r.config["combination_mode"],
                           "backtrack": r.config["backtrack"]}
            h = h.assign(**new_columns)
            df_list.append(h)
        df = pd.concat(df_list)

        # Save the dataframe to a file for future use
        df.to_pickle(filename)

    # Plot
    plots_transient = {"bidirectional-ring": None, "unidirectional-ring": None, "unirandom-ring": None}
    for topology_type in tqdm.tqdm(plots_transient.keys(), desc="Plotting [transient]"):
        # Only select data for this topology type and remove "topology_type" column from the dataframe
        # d = df[(df["topology_type"] == topology_type) & (df["seed"] == 1)]
        d = df[(df["topology_type"] == topology_type)]
        d = d.drop(columns=["topology_type"])
        # Apply mapping labels on "supergraph_type" column
        d["supergraph_type"] = d["supergraph_type"].map(labels)

        # Step 1: Group by 'transient/matched_percentage' and 'sigma', then interpolate 'transient/t_elapsed'
        d_grouped = d.groupby(['sigma', 'seed'])

        # Define a common 'transient/matched_percentage' range for interpolation
        common_matched_percentage = np.linspace(0, 100, num=101)

        # Function to interpolate 'transient/t_elapsed' for each group
        def interpolate_group(group):
            return np.interp(common_matched_percentage, group['transient/matched_percentage'], group['transient/t_elapsed'])

        # Apply the interpolation
        d_interpolated = d_grouped.apply(interpolate_group)

        # Convert the interpolated data back to a DataFrame
        d_interpolated = pd.DataFrame(d_interpolated.tolist(), index=d_interpolated.index)
        d_interpolated_reset = d_interpolated.reset_index()
        d_interpolated_reset_melted = d_interpolated_reset.melt(id_vars=['sigma', 'seed'], var_name='matched_percentage_interp', value_name='t_elapsed_interp')

        # Plot
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
        plots_transient[topology_type] = (fig, ax)
        sns.lineplot(data=d_interpolated_reset_melted, x="matched_percentage_interp", y="t_elapsed_interp", hue="sigma", ax=ax, palette=fcolor, errorbar="sd")

        # Set legend & axis labels
        ax.set_ylim(0, 140)
        ax.set_xlim(0, 100)
        ax.set_xlabel("matched (%)")
        ax.set_ylabel("elapsed time (s)")
        ax.legend(handles=None, ncol=4, loc='upper left', fancybox=True, shadow=True, bbox_to_anchor=(1.0, 1.0))

    # Ablation plots
    filters = {
        "group": {"$in": ["default-32-evaluation-2023-08-08-1631", "default-64-evaluation-2023-08-08-1731",
                          "sort-32-0.1-evaluation-2023-08-10-0826", "backtrack-32-0.1-evaluation-2023-08-10-0829",
                          "combination-32-0.1-evaluation-2023-08-09-2313"]},
        # "group": {"$in": ["test-evaluation-2023-07-01-1706", "test-evaluation-2023-06-30-2117",
        #                   "combination-evaluation-2023-07-03-2204", "backtrack-evaluation-2023-07-03-1622"]},
        "config.topology_type": {"$in": ["bidirectional-ring", "unidirectional-ring", "unirandom-ring"]},
        "config.supergraph_type": {"$in": ["mcs"]},
        "config.sigma": 0.1,
        "config.num_nodes": 32,
        "config.sort_mode": {"$in": ["optimal", "arbitrary", None, "null"]},
        "config.combination_mode": {"$in": ["power", "linear", "null", None]},
        "config.backtrack": {"$in": [0, 5, 10, 15, 20]},
        "State": "finished",
    }
    runs = api.runs(path="supergraph", filters=filters, per_page=1000)

    # Load data
    rows_list = []
    for r in tqdm.tqdm(runs, desc="Loading data [ablation]"):
        if len(r.summary.keys()) == 0:
            print("skipping: ", r.config, r.summary)
            continue
        v = r.summary["final/efficiency_percentage"]
        if "final/t_elapsed" in r.summary.keys():
            t = r.summary["final/t_elapsed"]
        elif "transient/t_elapsed" in r.summary.keys():
            t = r.summary["transient/t_elapsed"]
        else:
            print("skipping: ", r.config, r.summary)
            continue
        new_row = {"topology_type": r.config["topology_type"], "num_nodes": r.config["num_nodes"], "sigma": r.config["sigma"],
                   "sort_mode": r.config["sort_mode"], "combination_mode": r.config["combination_mode"], "backtrack": r.config["backtrack"],
                   "group": r.group, "seed": r.config["seed"], "supergraph_type": r.config["supergraph_type"],
                   "efficiency_percentage": v, "t_elapsed": t}
        rows_list.append(new_row)
    df = pd.DataFrame(rows_list)

    # Plot ablation[combination_mode]
    run_mask = {
        # "combination_mode": ((df["group"] == "test-evaluation-2023-06-30-2117") | (df["group"] == "combination-evaluation-2023-07-03-2204")) &
        "combination_mode": ((df["group"] == "default-32-evaluation-2023-08-08-1631") | (df["group"] == "combination-32-0.1-evaluation-2023-08-09-2313")) &
                            (df["sort_mode"] != "optimal") &
                            (df["backtrack"] == 5),
        # "sort_mode": ((df["group"] == "test-evaluation-2023-06-30-2117") | (df["group"] == "test-evaluation-2023-07-01-1706")) &
        "sort_mode": ((df["group"] == "default-32-evaluation-2023-08-08-1631") | (df["group"] == "sort-32-0.1-evaluation-2023-08-10-0826")) &
                     (df["combination_mode"] == "linear") &
                     (df["backtrack"] == 5) &
                     (df["topology_type"] == "unidirectional-ring"),
        # "backtrack": ((df["group"] == "backtrack-evaluation-2023-07-03-1622")) &
        "backtrack": ((df["group"] == "default-32-evaluation-2023-08-08-1631") | (df["group"] == "backtrack-32-0.1-evaluation-2023-08-10-0829")) &
                      (df["combination_mode"] == "linear") &
                      (df["sort_mode"] != "optimal")
    }
    plots_ablation_efficiency = {"combination_mode": None, "sort_mode": None, "backtrack": None}
    plots_ablation_time = {"combination_mode": None, "sort_mode": None, "backtrack": None}
    ncol = {"combination_mode": 1, "sort_mode": 1, "backtrack": 1}
    for ablation_type in tqdm.tqdm(plots_ablation_efficiency.keys(), desc="Plotting ablation"):
        d = df[run_mask[ablation_type]].copy()
        # Only select data for this topology type and remove "topology_type" column from the dataframe
        # Apply mapping labels on columns
        d["supergraph_type"] = d["supergraph_type"].map(labels)
        d["topology_type"] = d["topology_type"].map(labels)

        # Plot
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
        plots_ablation_efficiency[ablation_type] = (fig, ax)
        sns.barplot(x="topology_type", y="efficiency_percentage", hue=ablation_type, data=d, palette=fcolor, ax=ax,
                    order=[labels[k] for k in ["unidirectional-ring", "bidirectional-ring", "unirandom-ring"]])

        # Set legend & axis labels
        ax.set_ylim(0, 100)
        ax.set_xlabel('topology')
        ax.set_ylabel("efficiency (%)")
        if ablation_type == "backtrack":
            from matplotlib.colors import LinearSegmentedColormap

            # Create your custom colormap
            cmap_colors = [(i, fcolor[i]) for i in [0, 5, 10, 15, 20]]
            norm_vals = np.linspace(0, 20, len(cmap_colors))
            colors = [(norm / 20, color) for norm, color in cmap_colors]
            cmap = LinearSegmentedColormap.from_list('backtrack_cmap', colors)

            _fig, _ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
            # cbar_ax = _fig.add_axes([0.2, 0.85, thirdwidth_figsize[0], thirdwidth_figsize[1]])
            cbar_ax = fig.add_axes([0.16, 0.775, 0.67, 0.65])

            # Hide grid lines
            cbar_ax.grid(False)

            # Hide axes ticks
            cbar_ax.set_xticks([])
            cbar_ax.set_yticks([])

            # Hide axes
            cbar_ax.set_frame_on(False)
            cbar_ax.xaxis.set_visible(False)
            cbar_ax.yaxis.set_visible(False)

            # Generate some example heatmap data
            data = np.random.randint(0, 21, (10, 10))
            cax = _ax.matshow(data, cmap=cmap)

            # Remove existing colorbar if any
            if len(_fig.axes) > 1:
                _fig.delaxes(_fig.axes[-1])

            # Backtrack
            cbar = plt.colorbar(cax, ax=cbar_ax, ticks=[0, 5, 10, 15, 20], orientation='horizontal', ticklocation='top')
            ax.get_legend().remove()
        else:
            ax.legend(handles=None, ncol=ncol[ablation_type], loc='best', fancybox=True, shadow=True)

        # Plot
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
        plots_ablation_time[ablation_type] = (fig, ax)
        sns.barplot(x="topology_type", y="t_elapsed", hue=ablation_type, data=d, palette=fcolor, ax=ax,
                    order=[labels[k] for k in ["unidirectional-ring", "bidirectional-ring", "unirandom-ring"]])

        # Set legend & axis labels
        if ablation_type == "combination_mode":
            ax.set_yscale("log")
        ax.set_xlabel("topology")
        ax.set_ylabel("elapsed time (s)")
        if ablation_type == "backtrack":
            from matplotlib.colors import LinearSegmentedColormap

            # Create your custom colormap
            cmap_colors = [(i, fcolor[i]) for i in [0, 5, 10, 15, 20]]
            norm_vals = np.linspace(0, 20, len(cmap_colors))
            colors = [(norm / 20, color) for norm, color in cmap_colors]
            cmap = LinearSegmentedColormap.from_list('backtrack_cmap', colors)

            _fig, _ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
            # cbar_ax = _fig.add_axes([0.2, 0.85, thirdwidth_figsize[0], thirdwidth_figsize[1]])
            cbar_ax = fig.add_axes([0.16, 0.775, 0.67, 0.65])

            # Hide grid lines
            cbar_ax.grid(False)

            # Hide axes ticks
            cbar_ax.set_xticks([])
            cbar_ax.set_yticks([])

            # Hide axes
            cbar_ax.set_frame_on(False)
            cbar_ax.xaxis.set_visible(False)
            cbar_ax.yaxis.set_visible(False)

            # Generate some example heatmap data
            data = np.random.randint(0, 21, (10, 10))
            cax = _ax.matshow(data, cmap=cmap)

            # Remove existing colorbar if any
            if len(_fig.axes) > 1:
                _fig.delaxes(_fig.axes[-1])

            # Backtrack
            cbar = plt.colorbar(cax, ax=cbar_ax, ticks=[0, 5, 10, 15, 20], orientation='horizontal', ticklocation='top')
            ax.set_ylim([0, 140])
            ax.get_legend().remove()
        else:
            ax.legend(handles=None, ncol=ncol[ablation_type], loc='best', fancybox=True, shadow=True)

    # Performance plots
    filters = {
        # "group": {"$in": ["test-evaluation-2023-07-01-1706", "test-evaluation-2023-06-30-2117"]},
        "group": {"$in": ["default-32-evaluation-2023-08-08-1631", "default-64-evaluation-2023-08-08-1731"]},
        "config.topology_type": {"$in": ["bidirectional-ring", "unidirectional-ring", "unirandom-ring"]},
        "config.supergraph_type": {"$in": ["generational", "topological", "mcs"]},
        "config.sigma": {"$in": [0, 0.1, 0.2, 0.3]},
        "config.num_nodes": {"$in": [2, 4, 8, 16, 32, 64]},
        "config.sort_mode": {"$in": ["null", "arbitrary", None]},
        "State": "finished",
    }
    runs = api.runs(path="supergraph", filters=filters, per_page=1000)

    # Load data
    rows_list = []
    for r in tqdm.tqdm(runs, desc="Loading data [performance]"):
        v = r.summary["final/efficiency_percentage"]
        new_row = {"topology_type": r.config["topology_type"], "num_nodes": r.config["num_nodes"], "sigma": r.config["sigma"],
                   "supergraph_type": r.config["supergraph_type"], "efficiency_percentage": v}
        rows_list.append(new_row)
    df = pd.DataFrame(rows_list)

    # Plot performance_size
    plots_perf_size = {"bidirectional-ring": None, "unidirectional-ring": None, "unirandom-ring": None}
    for topology_type in tqdm.tqdm(plots_perf_size.keys(), desc="Plotting performance_size"):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
        plots_perf_size[topology_type] = (fig, ax)

        # Only select data for this topology type and remove "topology_type" column from the dataframe
        d = df[(df["topology_type"] == topology_type) & (df["sigma"] == 0.0)]
        d = d.drop(columns=["topology_type"])
        # Apply mapping labels on "supergraph_type" column
        d["supergraph_type"] = d["supergraph_type"].map(labels)

        # sns.boxplot(x="num_nodes", y="efficiency_percentage", hue="supergraph_type", data=d, ax=ax,)
        sns.barplot(x="num_nodes", y="efficiency_percentage", hue="supergraph_type", data=d, palette=fcolor, ax=ax,
                    hue_order=[labels[k] for k in ["mcs", "generational", "topological"]])

        # Set legend & axis labels
        ax.set_ylim(0, 100)
        ax.set_xlabel('nodes')
        ax.set_ylabel("efficiency (%)")
        ax.legend(handles=None, ncol=3, loc='upper right', fancybox=True, shadow=True, bbox_to_anchor=(2, 2))

    # Plot performance_sigma
    plots_perf_sigma = {"bidirectional-ring": None, "unidirectional-ring": None, "unirandom-ring": None}
    for topology_type in tqdm.tqdm(plots_perf_sigma.keys(), desc="Plotting performance_sigma"):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
        plots_perf_sigma[topology_type] = (fig, ax)

        # Only select data for this topology type and remove "topology_type" column from the dataframe
        d = df[(df["topology_type"] == topology_type) & (df["num_nodes"] == 32)]
        d = d.drop(columns=["topology_type"])
        # Apply mapping labels on "supergraph_type" column
        d["supergraph_type"] = d["supergraph_type"].map(labels)

        # sns.boxplot(x="num_nodes", y="efficiency_percentage", hue="supergraph_type", data=d, ax=ax,)
        sns.barplot(x="sigma", y="efficiency_percentage", hue="supergraph_type", data=d, palette=fcolor, ax=ax,
                    hue_order=[labels[k] for k in ["mcs", "generational", "topological"]])

        # Set legend & axis labels
        ax.set_ylim(0, 100)
        ax.set_xlabel('sigma')
        ax.set_ylabel("efficiency (%)")
        ax.legend(handles=None, ncol=3, loc='upper right', fancybox=True, shadow=True, bbox_to_anchor=(2, 2))

    # Save
    PAPER_DIR = "/home/r2ci/Documents/project/MCS/MCS_RA-L/figures/python"
    VERSION_ID = ""

    # Prepend _ for readability
    VERSION_ID = "_" + VERSION_ID if len(VERSION_ID) else ""

    # Save plots
    fig, ax = plot_computational_complexity
    fig.savefig(f"{PAPER_DIR}/computational_complexity{VERSION_ID}.pdf", bbox_inches='tight')
    for pendulum_type, (fig, ax) in plots_pendulum.items():
        fig.savefig(f"{PAPER_DIR}/pendulum_{pendulum_type}{VERSION_ID}.pdf", bbox_inches='tight')
        if MUST_BREAK:
            break
    for topology_type, (fig, ax) in plots_transient.items():
        fig.savefig(f"{PAPER_DIR}/transient_time_{topology_type}_legend{VERSION_ID}.pdf", bbox_inches=export_legend(fig, ax.get_legend()))
        ax.get_legend().remove()
        fig.savefig(f"{PAPER_DIR}/transient_time_{topology_type}{VERSION_ID}.pdf", bbox_inches='tight')
        if MUST_BREAK:
            break
    for ablation_type, (fig, ax) in plots_ablation_efficiency.items():
        from matplotlib.transforms import Bbox, TransformedBbox
        fig_width, fig_height = fig.get_size_inches()
        width, height = thirdwidth_figsize  # The dimensions you want for the lower-right corner in inches
        # bbox = Bbox.from_bounds(0, 0, width, height)
        bbox = ax.get_tightbbox(fig.canvas.get_renderer())
        # Convert the bounding box to inches
        bbox_inches = TransformedBbox(bbox, fig.dpi_scale_trans.inverted())
        # bbox_inches = BBo
        fig.savefig(
            f"{PAPER_DIR}/ablation_efficiency_{ablation_type}{VERSION_ID}.pdf",
            bbox_inches=bbox_inches
        )
        # fig.savefig(f"{PAPER_DIR}/ablation_efficiency_{ablation_type}{VERSION_ID}.pdf", bbox_inches='tight')
        if MUST_BREAK:
            break
    for ablation_type, (fig, ax) in plots_ablation_time.items():
        from matplotlib.transforms import Bbox, TransformedBbox
        fig_width, fig_height = fig.get_size_inches()
        width, height = thirdwidth_figsize  # The dimensions you want for the lower-right corner in inches
        # bbox = Bbox.from_bounds(0, 0, width, height)
        bbox = ax.get_tightbbox(fig.canvas.get_renderer())
        # Convert the bounding box to inches
        bbox_inches = TransformedBbox(bbox, fig.dpi_scale_trans.inverted())
        # bbox_inches = BBo
        fig.savefig(
            f"{PAPER_DIR}/ablation_time_{ablation_type}{VERSION_ID}.pdf",
            bbox_inches=bbox_inches
        )
        fig.savefig(f"{PAPER_DIR}/ablation_time_{ablation_type}{VERSION_ID}.pdf", bbox_inches='tight')
        if MUST_BREAK:
            break
    for topology_type, (fig, ax) in plots_perf_size.items():
        fig.savefig(f"{PAPER_DIR}/performance_size_{topology_type}_legend{VERSION_ID}.pdf", bbox_inches=export_legend(fig, ax.get_legend()))
        ax.get_legend().remove()
        fig.savefig(f"{PAPER_DIR}/performance_size_{topology_type}{VERSION_ID}.pdf", bbox_inches='tight')
        if MUST_BREAK:
            break
    for topology_type, (fig, ax) in plots_perf_sigma.items():
        fig.savefig(f"{PAPER_DIR}/performance_sigma_{topology_type}_legend{VERSION_ID}.pdf", bbox_inches=export_legend(fig, ax.get_legend()))
        ax.get_legend().remove()
        fig.savefig(f"{PAPER_DIR}/performance_sigma_{topology_type}{VERSION_ID}.pdf", bbox_inches='tight')
        if MUST_BREAK:
            break