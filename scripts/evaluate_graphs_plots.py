import os
import json
import hashlib
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import supergraph.open_colors as oc


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
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['font.size'] = 12
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['xtick.major.pad'] = -2.0
    plt.rcParams['ytick.major.pad'] = -2.0
    plt.rcParams['lines.linewidth'] = 1.3
    plt.rcParams['axes.xmargin'] = 0.0
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # Rescale figures
    width_points = 245.71811*2
    width_inches = width_points / plt.rcParams['figure.dpi']
    default_figsize = plt.rcParams['figure.figsize']
    rescaled_width = width_inches
    rescaled_height = width_inches * default_figsize[1] / default_figsize[0]
    rescaled_figsize = [rescaled_width, rescaled_height]
    half_figsize = [0.5*s for s in rescaled_figsize]
    halfwidth_figsize = [c*s for c, s in zip([0.5, 0.6], rescaled_figsize)]
    fullwidth_figsize = [c*s for c, s in zip([1, 0.6], rescaled_figsize)]
    thirdwidth_figsize = [c * s for c, s in zip([1 / 3, 0.6], rescaled_figsize)]
    print("Default figsize:", default_figsize)
    print("Rescaled figsize:", rescaled_figsize)
    print("Half figsize:", half_figsize)
    print("Fullwidth figsize:", fullwidth_figsize)
    print("Halfwidth figsize:", halfwidth_figsize)
    print("Thirdwidth figsize:", thirdwidth_figsize)
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
        }
    cscheme.update({labels[k]: v for k, v in cscheme.items() if k in labels.keys()})
    ecolor, fcolor = oc.cscheme_fn(cscheme)

    import wandb
    api = wandb.Api()

    # Computational complexity plots
    filters = {
        # "group": {"$in": ["test-evaluation-2023-07-01-1706", "test-evaluation-2023-06-30-2117"]},
        "group": {"$in": ["default-32-evaluation-2023-08-08-1631", "default-64-evaluation-2023-08-08-1731"]},
        "config.topology_type": {"$in": ["bidirectional-ring", "unidirectional-ring", "unirandom-ring"]},
        "config.supergraph_type": {"$in": ["mcs"]},
        "config.sigma": {"$in": [0.0, 0.1, 0.2, 0.3]},
        # "config.num_nodes": {"$in": [2, 4, 8, 16, 32, 64]},
        "config.num_nodes": {"$in": [8, 16, 32, 64]},
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

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fullwidth_figsize)
    plot_computational_complexity = (fig, ax)
    sns.scatterplot(x="t_elapsed", y="efficiency_percentage", hue="sigma", size="nodes",
                    style="topology", palette=fcolor, ax=ax, data=d, legend=True)

    # Set legend & axis labels
    ax.set_ylim(0, 100)
    ax.set_xscale("log")
    ax.set_xlabel('elapsed time (s)')
    ax.set_ylabel("efficiency (%)")
    ax.legend(handles=None, ncol=3, loc='upper right', fancybox=True, shadow=True, prop={'size': 7})

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
        ax.legend(handles=None, ncol=1, loc='upper left', fancybox=True, shadow=True, prop={'size': 7})

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
        ax.legend(handles=None, ncol=1, loc='best', fancybox=True, shadow=True, prop={'size': 7})

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
        ax.legend(handles=None, ncol=1, loc='best', fancybox=True, shadow=True, prop={'size': 7})

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
        ax.set_xlabel('number of nodes')
        ax.set_ylabel("efficiency (%)")
        ax.legend(handles=None, ncol=1, loc='upper right', fancybox=True, shadow=True, prop={'size': 7})

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
        ax.legend(handles=None, ncol=1, loc='upper right', fancybox=True, shadow=True, prop={'size': 7})

    # Save
    PAPER_DIR = "/home/r2ci/Documents/project/MCS/MCS_RA-L/figures/python"
    VERSION_ID = ""

    # Prepend _ for readability
    VERSION_ID = "_" + VERSION_ID if len(VERSION_ID) else ""

    # Save plots
    fig, ax = plot_computational_complexity
    fig.savefig(f"{PAPER_DIR}/computational_complexity{VERSION_ID}.pdf", bbox_inches='tight')
    for topology_type, (fig, ax) in plots_transient.items():
        fig.savefig(f"{PAPER_DIR}/transient_time_{topology_type}{VERSION_ID}.pdf", bbox_inches='tight')
    for ablation_type, (fig, ax) in plots_ablation_efficiency.items():
        fig.savefig(f"{PAPER_DIR}/ablation_efficiency_{ablation_type}{VERSION_ID}.pdf", bbox_inches='tight')
    for ablation_type, (fig, ax) in plots_ablation_time.items():
        fig.savefig(f"{PAPER_DIR}/ablation_time_{ablation_type}{VERSION_ID}.pdf", bbox_inches='tight')
    for topology_type, (fig, ax) in plots_perf_size.items():
        fig.savefig(f"{PAPER_DIR}/performance_size_{topology_type}{VERSION_ID}.pdf", bbox_inches='tight')
    for topology_type, (fig, ax) in plots_perf_sigma.items():
        fig.savefig(f"{PAPER_DIR}/performance_sigma_{topology_type}{VERSION_ID}.pdf", bbox_inches='tight')
