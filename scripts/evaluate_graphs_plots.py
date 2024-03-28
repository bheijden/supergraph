import dill as pickle
import os
import json
import hashlib
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import supergraph.open_colors as oc
import supergraph
import supergraph.evaluate as eval


def export_legend(fig, legend, expand=None):
    expand = [-5, -5, 5, 5] if expand is None else expand
    # fig = legend.figure
    # fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    return bbox
    # fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def make_cps_plots():
    # ###############################################################
    # # Ablation plots
    # ###############################################################
    # filters = {
    #     "group": {"$in": ["default-32-evaluation-2023-08-08-1631", "default-64-evaluation-2023-08-08-1731",
    #                       "sort-32-0.1-evaluation-2023-08-10-0826", "backtrack-32-0.1-evaluation-2023-08-10-0829",
    #                       "combination-32-0.1-evaluation-2023-08-09-2313"]},
    #     # "group": {"$in": ["test-evaluation-2023-07-01-1706", "test-evaluation-2023-06-30-2117",
    #     #                   "combination-evaluation-2023-07-03-2204", "backtrack-evaluation-2023-07-03-1622"]},
    #     "config.topology_type": {"$in": ["bidirectional-ring", "unidirectional-ring", "unirandom-ring"]},
    #     "config.supergraph_type": {"$in": ["mcs"]},
    #     "config.sigma": 0.1,
    #     "config.num_nodes": 32,
    #     "config.sort_mode": {"$in": ["optimal", "arbitrary", None, "null"]},
    #     "config.combination_mode": {"$in": ["power", "linear", "null", None]},
    #     "config.backtrack": {"$in": [0, 5, 10, 15, 20]},
    #     "State": "finished",
    # }
    # runs = api.runs(path="supergraph", filters=filters, per_page=1000)
    #
    # # Load data
    # rows_list = []
    # for r in tqdm.tqdm(runs, desc="Loading data [ablation]"):
    #     if len(r.summary.keys()) == 0:
    #         print("skipping: ", r.config, r.summary)
    #         continue
    #     v = r.summary["final/efficiency_percentage"]
    #     if "final/t_elapsed" in r.summary.keys():
    #         t = r.summary["final/t_elapsed"]
    #     elif "transient/t_elapsed" in r.summary.keys():
    #         t = r.summary["transient/t_elapsed"]
    #     else:
    #         print("skipping: ", r.config, r.summary)
    #         continue
    #     new_row = {"topology_type": r.config["topology_type"], "num_nodes": r.config["num_nodes"], "sigma": r.config["sigma"],
    #                "sort_mode": r.config["sort_mode"], "combination_mode": r.config["combination_mode"], "backtrack": r.config["backtrack"],
    #                "group": r.group, "seed": r.config["seed"], "supergraph_type": r.config["supergraph_type"],
    #                "efficiency_percentage": v, "t_elapsed": t}
    #     rows_list.append(new_row)
    # df = pd.DataFrame(rows_list)
    #
    # # Plot ablation[combination_mode]
    # run_mask = {
    #     # "combination_mode": ((df["group"] == "test-evaluation-2023-06-30-2117") | (df["group"] == "combination-evaluation-2023-07-03-2204")) &
    #     "combination_mode": ((df["group"] == "default-32-evaluation-2023-08-08-1631") | (df["group"] == "combination-32-0.1-evaluation-2023-08-09-2313")) &
    #                         (df["sort_mode"] != "optimal") &
    #                         (df["backtrack"] == 5),
    #     # "sort_mode": ((df["group"] == "test-evaluation-2023-06-30-2117") | (df["group"] == "test-evaluation-2023-07-01-1706")) &
    #     "sort_mode": ((df["group"] == "default-32-evaluation-2023-08-08-1631") | (df["group"] == "sort-32-0.1-evaluation-2023-08-10-0826")) &
    #                  (df["combination_mode"] == "linear") &
    #                  (df["backtrack"] == 5) &
    #                  (df["topology_type"] == "unidirectional-ring"),
    #     # "backtrack": ((df["group"] == "backtrack-evaluation-2023-07-03-1622")) &
    #     "backtrack": ((df["group"] == "default-32-evaluation-2023-08-08-1631") | (df["group"] == "backtrack-32-0.1-evaluation-2023-08-10-0829")) &
    #                  (df["combination_mode"] == "linear") &
    #                  (df["sort_mode"] != "optimal")
    # }
    plots_ablation_efficiency = {"combination_mode": (None, None), "sort_mode": (None, None), "backtrack": (None, None)}
    plots_ablation_time = {"combination_mode": (None, None), "sort_mode": (None, None), "backtrack": (None, None)}
    # ncol = {"combination_mode": 1, "sort_mode": 1, "backtrack": 1}
    # for ablation_type in tqdm.tqdm(plots_ablation_efficiency.keys(), desc="Plotting ablation"):
    #
    #     d = df[run_mask[ablation_type]].copy()
    #     # Only select data for this topology type and remove "topology_type" column from the dataframe
    #     # Apply mapping labels on columns
    #     d["supergraph_type"] = d["supergraph_type"].map(labels)
    #     d["topology_type"] = d["topology_type"].map(labels)
    #     if ablation_type == "sort_mode":
    #         d["sort_mode"] = d["sort_mode"].map(labels)
    #
    #     # Plot
    #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
    #     plots_ablation_efficiency[ablation_type] = (fig, ax)
    #     sns.barplot(x="topology_type", y="efficiency_percentage", hue=ablation_type, data=d, palette=fcolor, ax=ax,
    #                 order=[labels[k] for k in ["unidirectional-ring", "bidirectional-ring", "unirandom-ring"]])
    #
    #     # Set legend & axis labels
    #     ax.set_ylim(0, 100)
    #     ax.set_xlabel('topology')
    #     ax.set_ylabel("efficiency (%)")
    #     if ablation_type == "backtrack":
    #         from matplotlib.colors import LinearSegmentedColormap
    #
    #         # Create your custom colormap
    #         cmap_colors = [(i, fcolor[i]) for i in [0, 5, 10, 15, 20]]
    #         norm_vals = np.linspace(0, 20, len(cmap_colors))
    #         colors = [(norm / 20, color) for norm, color in cmap_colors]
    #         cmap = LinearSegmentedColormap.from_list('backtrack_cmap', colors)
    #
    #         _fig, _ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
    #         # cbar_ax = _fig.add_axes([0.2, 0.85, thirdwidth_figsize[0], thirdwidth_figsize[1]])
    #         cbar_ax = fig.add_axes([0.16, 0.75, 0.67, 0.65])
    #
    #         # Hide grid lines
    #         cbar_ax.grid(False)
    #
    #         # Hide axes ticks
    #         cbar_ax.set_xticks([])
    #         cbar_ax.set_yticks([])
    #
    #         # Hide axes
    #         cbar_ax.set_frame_on(False)
    #         cbar_ax.xaxis.set_visible(False)
    #         cbar_ax.yaxis.set_visible(False)
    #
    #         # Generate some example heatmap data
    #         data = np.random.randint(0, 21, (10, 10))
    #         cax = _ax.matshow(data, cmap=cmap)
    #
    #         # Remove existing colorbar if any
    #         if len(_fig.axes) > 1:
    #             _fig.delaxes(_fig.axes[-1])
    #
    #         # Backtrack
    #         cbar = plt.colorbar(cax, ax=cbar_ax, ticks=[0, 5, 10, 15, 20], orientation='horizontal', ticklocation='top')
    #         ax.get_legend().remove()
    #     else:
    #         ax.legend(handles=None, ncol=ncol[ablation_type], loc='best', fancybox=True, shadow=True)
    #
    #     # Plot
    #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
    #     plots_ablation_time[ablation_type] = (fig, ax)
    #     sns.barplot(x="topology_type", y="t_elapsed", hue=ablation_type, data=d, palette=fcolor, ax=ax,
    #                 order=[labels[k] for k in ["unidirectional-ring", "bidirectional-ring", "unirandom-ring"]])
    #
    #     # Set legend & axis labels
    #     if ablation_type == "combination_mode":
    #         ax.set_yscale("log")
    #     ax.set_xlabel("topology")
    #     ax.set_ylabel("elapsed time (s)")
    #     if ablation_type == "backtrack":
    #         from matplotlib.colors import LinearSegmentedColormap
    #
    #         # Create your custom colormap
    #         cmap_colors = [(i, fcolor[i]) for i in [0, 5, 10, 15, 20]]
    #         norm_vals = np.linspace(0, 20, len(cmap_colors))
    #         colors = [(norm / 20, color) for norm, color in cmap_colors]
    #         cmap = LinearSegmentedColormap.from_list('backtrack_cmap', colors)
    #
    #         _fig, _ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
    #         # cbar_ax = _fig.add_axes([0.2, 0.85, thirdwidth_figsize[0], thirdwidth_figsize[1]])
    #         cbar_ax = fig.add_axes([0.16, 0.75, 0.67, 0.65])
    #
    #         # Hide grid lines
    #         cbar_ax.grid(False)
    #
    #         # Hide axes ticks
    #         cbar_ax.set_xticks([])
    #         cbar_ax.set_yticks([])
    #
    #         # Hide axes
    #         cbar_ax.set_frame_on(False)
    #         cbar_ax.xaxis.set_visible(False)
    #         cbar_ax.yaxis.set_visible(False)
    #
    #         # Generate some example heatmap data
    #         data = np.random.randint(0, 21, (10, 10))
    #         cax = _ax.matshow(data, cmap=cmap)
    #
    #         # Remove existing colorbar if any
    #         if len(_fig.axes) > 1:
    #             _fig.delaxes(_fig.axes[-1])
    #
    #         # Backtrack
    #         cbar = plt.colorbar(cax, ax=cbar_ax, ticks=[0, 5, 10, 15, 20], orientation='horizontal', ticklocation='top')
    #         ax.set_ylim([0, 140])
    #         ax.get_legend().remove()
    #     else:
    #         ax.legend(handles=None, ncol=ncol[ablation_type], loc='best', fancybox=True, shadow=True)

    ###############################################################
    # Computational complexity plots
    ###############################################################
    filters = {
        "group": {"$in": ["cps-all-noablation-evaluation-2024-03-17-1248"]},
        "config.topology_type": {"$in": ["v2v-platooning", "uav-swarm-control"]},
        "config.supergraph_type": {"$in": ["mcs"]},
        "config.sigma": {"$in": [0.0, 0.1, 0.2, 0.3]},
        "config.num_nodes": {"$in": [2, 4, 8, 16, 32, 64]},
        "config.sort_mode": {"$in": ["arbitrary"]},
        "config.combination_mode": {"$in": ["linear"]},
        "config.backtrack": 20,  # 20
        "State": "finished",
    }
    runs = api.runs(path="supergraph", filters=filters, per_page=1000)

    # Load data
    rows_list = []
    for r in tqdm.tqdm(runs, desc="Loading data [cps][computational complexity]"):
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
    scaling = 0.73
    cps_fullwidth_figsize = [c * s*scaling for c, s in zip([1, 0.75], rescaled_figsize)]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=cps_fullwidth_figsize)
    plot_computational_complexity = (fig, ax)
    sns.scatterplot(x="t_elapsed", y="efficiency_percentage", hue="sigma", size="nodes",
                    style="topology", palette=fcolor, ax=ax, data=d, legend=True)

    # Set legend & axis labels
    ax.set_ylim(0, 100)
    ax.set_xlim(0.8e-1, 1.5e1)
    ax.set_xscale("log")
    ax.set_xlabel('elapsed time (s)')
    ax.set_ylabel("efficiency (%)")
    # Place the legend on the right outside of the figure
    ax.legend(handles=ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1],
              ncol=2, loc='upper left', bbox_to_anchor=(1, 1), fancybox=True, shadow=True)

    ###############################################################
    # Transient plots
    ###############################################################
    filters = {
        # "group": {"$in": ["test-evaluation-2023-07-01-1706", "test-evaluation-2023-06-30-2117"]},
        "group": {"$in": ["cps-all-noablation-evaluation-2024-03-17-1248"]},
        "config.topology_type": {"$in": ["v2v-platooning", "uav-swarm-control"]},
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
        for r in tqdm.tqdm(runs, desc="Loading data [cps][transient]"):
            h = r.history(samples=10000,
                          keys=["transient/t_elapsed", "transient/matched_percentage", "transient/efficiency_percentage"],
                          x_axis="_step", pandas=True, stream="default")
            new_columns = {"topology_type": r.config["topology_type"], "num_nodes": r.config["num_nodes"],
                           "supergraph_type": r.config["supergraph_type"], "sigma": r.config["sigma"],
                           "seed": r.config["seed"],
                           "group": r.group, "sort_mode": r.config["sort_mode"],
                           "combination_mode": r.config["combination_mode"],
                           "backtrack": r.config["backtrack"]}
            h = h.assign(**new_columns)
            df_list.append(h)
        df = pd.concat(df_list)

        # Save the dataframe to a file for future use
        df.to_pickle(filename)

    # Plot
    plots_transient = {"v2v-platooning": None, "uav-swarm-control": None}
    for topology_type in tqdm.tqdm(plots_transient.keys(), desc="Plotting [cps][transient]"):
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
        d_interpolated_reset_melted = d_interpolated_reset.melt(id_vars=['sigma', 'seed'],
                                                                var_name='matched_percentage_interp',
                                                                value_name='t_elapsed_interp')

        # Plot
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
        plots_transient[topology_type] = (fig, ax)
        sns.lineplot(data=d_interpolated_reset_melted, x="matched_percentage_interp", y="t_elapsed_interp", hue="sigma", ax=ax,
                     palette=fcolor, errorbar="sd")

        # Set legend & axis labels
        ax.set_ylim(0, 6)
        ax.set_xlim(0, 100)
        ax.set_xlabel("matched (%)")
        ax.set_ylabel("elapsed time (s)")
        ax.legend(handles=None, ncol=4, loc='upper left', fancybox=True, shadow=True, bbox_to_anchor=(1.0, 1.0))

    ###############################################################
    # Performance plots
    ###############################################################
    filters = {
        # "group": {"$in": ["test-evaluation-2023-07-01-1706", "test-evaluation-2023-06-30-2117"]},
        "group": {"$in": ["cps-all-noablation-evaluation-2024-03-17-1248"]},
        "config.topology_type": {"$in": ["v2v-platooning", "uav-swarm-control"]},
        "config.supergraph_type": {"$in": ["generational", "topological", "mcs"]},
        "config.sigma": {"$in": [0, 0.1, 0.2, 0.3]},
        "config.num_nodes": {"$in": [2, 4, 8, 16, 32, 64]},
        "config.sort_mode": {"$in": ["null", "arbitrary", None]},
        "State": "finished",
    }
    runs = api.runs(path="supergraph", filters=filters, per_page=1000)

    # Load data
    rows_list = []
    for r in tqdm.tqdm(runs, desc="Loading data [cps][performance]"):
        v = r.summary["final/efficiency_percentage"]
        new_row = {"topology_type": r.config["topology_type"], "num_nodes": r.config["num_nodes"], "sigma": r.config["sigma"],
                   "supergraph_type": r.config["supergraph_type"], "efficiency_percentage": v}
        rows_list.append(new_row)
    df = pd.DataFrame(rows_list)

    ###############################################################
    # Plot performance_size
    ###############################################################
    plots_perf_size = {"v2v-platooning": None, "uav-swarm-control": None}
    for topology_type in tqdm.tqdm(plots_perf_size.keys(), desc="Plotting [cps][performance_size]"):
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

    ###############################################################
    # Plot performance_sigma
    ###############################################################
    plots_perf_sigma = {"v2v-platooning": None, "uav-swarm-control": None}
    for topology_type in tqdm.tqdm(plots_perf_sigma.keys(), desc="Plotting [cps][performance_sigma]"):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
        plots_perf_sigma[topology_type] = (fig, ax)

        # Only select data for this topology type and remove "topology_type" column from the dataframe
        d = df[(df["topology_type"] == topology_type) & (df["num_nodes"] == 8)]
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

    ###############################################################
    # SAVING
    ###############################################################
    version_id = f"_cps{VERSION_ID}"
    fig, ax = plot_computational_complexity
    fig.savefig(f"{PAPER_DIR}/computational_complexity{version_id}.pdf", bbox_inches='tight')
    for topology_type, (fig, ax) in plots_transient.items():
        fig.savefig(f"{PAPER_DIR}/transient_time_{topology_type}_legend{version_id}.pdf", bbox_inches=export_legend(fig, ax.get_legend()))
        ax.get_legend().remove()
        fig.savefig(f"{PAPER_DIR}/transient_time_{topology_type}{version_id}.pdf", bbox_inches='tight')
        if MUST_BREAK:
            break
    for ablation_type, (fig, ax) in plots_ablation_efficiency.items():
        if fig is None:
            print(f"Skipping ablation_efficiency_{ablation_type}{version_id}.pdf")
            continue
        from matplotlib.transforms import Bbox, TransformedBbox
        fig_width, fig_height = fig.get_size_inches()
        width, height = thirdwidth_figsize  # The dimensions you want for the lower-right corner in inches
        # bbox = Bbox.from_bounds(0, 0, width, height)
        bbox = ax.get_tightbbox(fig.canvas.get_renderer())
        # Convert the bounding box to inches
        bbox_inches = TransformedBbox(bbox, fig.dpi_scale_trans.inverted())
        # bbox_inches = BBo
        fig.savefig(
            f"{PAPER_DIR}/ablation_efficiency_{ablation_type}{version_id}.pdf",
            bbox_inches=bbox_inches
        )
        # fig.savefig(f"{PAPER_DIR}/ablation_efficiency_{ablation_type}{version_id}.pdf", bbox_inches='tight')
        if MUST_BREAK:
            break
    for ablation_type, (fig, ax) in plots_ablation_time.items():
        if fig is None:
            print(f"Skipping ablation_time_{ablation_type}{version_id}.pdf")
            continue
        from matplotlib.transforms import Bbox, TransformedBbox
        fig_width, fig_height = fig.get_size_inches()
        # width, height = thirdwidth_figsize  # The dimensions you want for the lower-right corner in inches
        # bbox = Bbox.from_bounds(0, 0, width, height)
        bbox = ax.get_tightbbox(fig.canvas.get_renderer())
        # Convert the bounding box to inches
        bbox_inches = TransformedBbox(bbox, fig.dpi_scale_trans.inverted())
        # Increase the height of the bounding box by 0.1 inch
        x0, y0, x1, y1 = bbox_inches.bounds
        bbox_inches_add = Bbox.from_bounds(x0, y0, x1 - x0, y1 - y0 + 0.01)
        # bbox_inches = BBo
        fig.savefig(
            f"{PAPER_DIR}/ablation_time_{ablation_type}{version_id}.pdf",
            bbox_inches=bbox_inches_add
        )
        if MUST_BREAK:
            break
    for topology_type, (fig, ax) in plots_perf_size.items():
        fig.savefig(f"{PAPER_DIR}/performance_size_{topology_type}_legend{version_id}.pdf", bbox_inches=export_legend(fig, ax.get_legend()))
        ax.get_legend().remove()
        fig.savefig(f"{PAPER_DIR}/performance_size_{topology_type}{version_id}.pdf", bbox_inches='tight')
        if MUST_BREAK:
            break
    for topology_type, (fig, ax) in plots_perf_sigma.items():
        fig.savefig(f"{PAPER_DIR}/performance_sigma_{topology_type}_legend{version_id}.pdf", bbox_inches=export_legend(fig, ax.get_legend()))
        ax.get_legend().remove()
        fig.savefig(f"{PAPER_DIR}/performance_sigma_{topology_type}{version_id}.pdf", bbox_inches='tight')
        if MUST_BREAK:
            break
    return plots_ablation_efficiency, plots_ablation_time, plot_computational_complexity, plots_transient, plots_perf_size, plots_perf_sigma


def make_abstract_topology_plots():
    ###############################################################
    # Ablation plots
    ###############################################################
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
        if ablation_type == "sort_mode":
            d["sort_mode"] = d["sort_mode"].map(labels)

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
            cbar_ax = fig.add_axes([0.16, 0.75, 0.67, 0.65])

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
            cbar_ax = fig.add_axes([0.16, 0.75, 0.67, 0.65])

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
    ###############################################################
    # Computational complexity plots
    ###############################################################
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

    ###############################################################
    # Transient plots
    ###############################################################
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
                           "supergraph_type": r.config["supergraph_type"], "sigma": r.config["sigma"],
                           "seed": r.config["seed"],
                           "group": r.group, "sort_mode": r.config["sort_mode"],
                           "combination_mode": r.config["combination_mode"],
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
        d_interpolated_reset_melted = d_interpolated_reset.melt(id_vars=['sigma', 'seed'],
                                                                var_name='matched_percentage_interp',
                                                                value_name='t_elapsed_interp')

        # Plot
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
        plots_transient[topology_type] = (fig, ax)
        sns.lineplot(data=d_interpolated_reset_melted, x="matched_percentage_interp", y="t_elapsed_interp", hue="sigma", ax=ax,
                     palette=fcolor, errorbar="sd")

        # Set legend & axis labels
        ax.set_ylim(0, 140)
        ax.set_xlim(0, 100)
        ax.set_xlabel("matched (%)")
        ax.set_ylabel("elapsed time (s)")
        ax.legend(handles=None, ncol=4, loc='upper left', fancybox=True, shadow=True, bbox_to_anchor=(1.0, 1.0))

    ###############################################################
    # Performance plots
    ###############################################################
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

    ###############################################################
    # Plot performance_size
    ###############################################################
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

    ###############################################################
    # Plot performance_sigma
    ###############################################################
    plots_perf_sigma = {"bidirectional-ring": None, "unidirectional-ring": None, "unirandom-ring": None}
    for topology_type in tqdm.tqdm(plots_perf_sigma.keys(), desc="Plotting performance_sigma"):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
        plots_perf_sigma[topology_type] = (fig, ax)

        # Only select data for this topology type and remove "topology_type" column from the dataframe
        d = df[(df["topology_type"] == topology_type) & (df["num_nodes"] == 8)]
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

    ###############################################################
    # SAVING
    ###############################################################
    fig, ax = plot_computational_complexity
    fig.savefig(f"{PAPER_DIR}/computational_complexity{VERSION_ID}.pdf", bbox_inches='tight')
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
        # width, height = thirdwidth_figsize  # The dimensions you want for the lower-right corner in inches
        # bbox = Bbox.from_bounds(0, 0, width, height)
        bbox = ax.get_tightbbox(fig.canvas.get_renderer())
        # Convert the bounding box to inches
        bbox_inches = TransformedBbox(bbox, fig.dpi_scale_trans.inverted())
        # Increase the height of the bounding box by 0.1 inch
        x0, y0, x1, y1 = bbox_inches.bounds
        bbox_inches_add = Bbox.from_bounds(x0, y0, x1 - x0, y1 - y0 + 0.01)
        # bbox_inches = BBo
        fig.savefig(
            f"{PAPER_DIR}/ablation_time_{ablation_type}{VERSION_ID}.pdf",
            bbox_inches=bbox_inches_add
        )
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
    return plots_ablation_efficiency, plots_ablation_time, plot_computational_complexity, plots_transient, plots_perf_size, plots_perf_sigma


def make_plot_pendulum():
    ###############################################################
    # Get delay sim data
    ###############################################################
    filters = {
        "group": {"$in": ["train-delaysim-evaluate-2024-03-18-0759"]},
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

    if all([os.path.exists(filename.replace("df_", f"df_{f}_")) for f in ["eval_delay_sim"]]):
        # If the file already exists, simply load the dataframe from it
        print("Loading dataframe [pendulum] from file:", filename)
        df_eval_delay_sim = pd.read_pickle(filename.replace("df_", f"df_eval_delay_sim_"))
    else:
        # If the file doesn't exist, generate the dataframe as usual
        runs = api.runs(path="supergraph", filters=filters, per_page=1000)

        df_list_eval_delay_sim = []
        for r in tqdm.tqdm(runs, desc="Loading data [pendulum]"):
            # Train curve
            h = r.history(samples=10000, keys=["train/ep_rew_mean", "train/total_timesteps"], x_axis="_step", pandas=True,
                          stream="default")
            # new_columns = {"job_type": r.job_type, "seed": r.config["seed"]}
            # h = h.assign(**new_columns)
            # df_list_train.append(h)
            #
            # # Speed curve
            # h1 = r.history(samples=10000, keys=["speed/supergraph_type", "speed/fps"], x_axis="_step", pandas=True,
            #                stream="default")
            # h2 = r.history(samples=10000, keys=["supergraph/supergraph_type", "supergraph/efficiency_percentage"],
            #                x_axis="_step", pandas=True, stream="default")
            #
            # # join h1 and h2 on "speed/supergraph_type" and "supergraph/supergraph_type"
            # h = h1.merge(h2, left_on="speed/supergraph_type", right_on="supergraph/supergraph_type",
            #              suffixes=("_speed", "_supergraph"))
            # h.drop(columns=["supergraph/supergraph_type"], inplace=True)
            #
            # new_columns = {"job_type": r.job_type, "seed": r.config["seed"]}
            # h = h.assign(**new_columns)
            # df_list_speed.append(h)

            # Eval curve
            new_columns = {"job_type": r.job_type, "seed": r.config["seed"],
                           "ep_rew_mean_deterministic": r.summary["final/deterministic/ep_rew_mean"],
                           "ep_rew_mean_async": r.summary["final/async/ep_rew_mean"]}
            # Create a dataframe from new_columns
            h = pd.DataFrame(new_columns, index=[0])
            df_list_eval_delay_sim.append(h)
        df_eval_delay_sim = pd.concat(df_list_eval_delay_sim)

        # Save the dataframe to a file for future use
        df_eval_delay_sim.to_pickle(filename.replace("df_", "df_eval_delay_sim_"))

    ###############################################################
    # Pendulum plots
    ###############################################################
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

    # Join df_eval and df_eval_delay_sim based on job_type and seed
    df_eval = df_eval.merge(df_eval_delay_sim[['job_type', 'seed', 'ep_rew_mean_deterministic', 'ep_rew_mean_async']], on=['job_type', 'seed'])

    plots_pendulum = {"speed": None, "train": None, "eval": None, "eval_delay_sim": None}

    # Speed plot
    d = df_speed.copy()
    d["supergraph_type"] = d["speed/supergraph_type"].map(labels)
    # Applying the condition to change 'supergraph_type' where 'job_type' is 'deterministic'
    d = d[(d["job_type"] != "deterministic") | ((d["job_type"] == "deterministic") & (d["supergraph_type"] == "mcs"))]
    d.loc[d["job_type"] == "deterministic", "supergraph_type"] = 'deterministic'
    d["job_type"] = d["job_type"].map(labels)

    # remove speed/supergraph_type column
    d = d.drop(columns=["speed/supergraph_type"])

    # Group by relevant columns and compute the mean
    d = d.groupby(["supergraph_type", "job_type"]).agg({
        "speed/fps": 'mean',
        "supergraph/efficiency_percentage": 'mean'
    }).reset_index()

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
    plots_pendulum["speed"] = (fig, ax)
    sns.scatterplot(x="supergraph/efficiency_percentage", y="speed/fps", hue="supergraph_type", palette=fcolor, ax=ax, data=d, legend=True,
                    hue_order=[labels[k] for k in ["deterministic", "mcs", "generational", "topological"]])

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
    ax.set_ylim(0e6, 6e6)
    ax.set_yticks([2e6, 4e6, 6e6])
    ax.set_xlim(0, 100)
    ax.set_xlabel('efficiency (%)')
    ax.set_ylabel("speed (fps)")
    # Place the legend on the right outside of the figure
    ax.legend(handles=ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1],
              # ncol=1, loc='best', bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
              ncol=4, loc='lower right', fancybox=True, shadow=True, bbox_to_anchor=(2, 2))

    # Eval plot
    d = df_eval.copy()
    d["job_type"] = d["job_type"].map(labels)

    # Melting the DataFrame
    df_melted = d.melt(id_vars=['job_type', 'seed'], value_vars=['ep_rew_mean_real', 'ep_rew_mean_sim', "ep_rew_mean_deterministic", "ep_rew_mean_async"], var_name='measure',
                       value_name='cost')
    df_melted["cost"] = df_melted["cost"]*-1

    # Adding a new column 'platform'
    df_melted["environment"] = df_melted['measure'].map({'ep_rew_mean_real': 'real', 'ep_rew_mean_sim': 'sim',
                                                         'ep_rew_mean_deterministic': labels['deterministic'], 'ep_rew_mean_async': labels['async']})

    # Dropping the 'measure' column, if not needed
    df_melted.drop('measure', axis=1, inplace=True)

    # # Selecting and duplicating "real" entries with a new environment label
    # real_entries = df_melted[df_melted['environment'] == 'real'].copy()
    # real_entries['environment'] = 'sim_delay'
    # # Appending duplicated entries to the original DataFrame
    # df_concat = pd.concat([df_melted, real_entries], ignore_index=True)

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
    plots_pendulum["eval"] = (fig, ax)
    # sns.barplot(x="job_type", y="cost", hue="environment", data=df_melted, palette=fcolor, ax=ax)
    sns.barplot(x="environment", y="cost", hue="job_type", data=df_melted, palette=fcolor, ax=ax,
                hue_order=[labels[k] for k in ["async", "deterministic"]],
                order=[k for k in ["sim", "real"]])

    # Set legend & axis labels
    ax.set_ylim(0, 1400)
    ax.set_xlabel('environment')
    ax.set_ylabel("cost")
    ax.legend(handles=None, ncol=1, loc='best', fancybox=True, shadow=True)

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
    plots_pendulum["eval_delay_sim"] = (fig, ax)
    # sns.barplot(x="job_type", y="cost", hue="environment", data=df_melted, palette=fcolor, ax=ax)
    sns.barplot(x="environment", y="cost", hue="job_type", data=df_melted, palette=fcolor, ax=ax,
                hue_order=[labels[k] for k in ["async", "deterministic"]],
                order=[k for k in ["seq", "mcs", "real"]])

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

    ###############################################################
    # SAVING
    ###############################################################
    for pendulum_type, (fig, ax) in plots_pendulum.items():
        fig.savefig(f"{PAPER_DIR}/pendulum_{pendulum_type}{VERSION_ID}_legend{VERSION_ID}.pdf", bbox_inches=export_legend(fig, ax.get_legend()))
        ax.get_legend().remove()
        fig.savefig(f"{PAPER_DIR}/pendulum_{pendulum_type}{VERSION_ID}.pdf", bbox_inches='tight')
        if MUST_BREAK:
            break
    return plots_pendulum


def make_box_pushing_plots():
    ###############################################################
    # Box pushing plots
    ###############################################################
    all_plots_box = dict(fixed={"speed": None, "traj": None, "final": None, "zoom": None},
                         variable={"speed": None, "traj": None, "final": None, "zoom": None})

    # Prepare data frames
    SKIP_BOX_PUSHING_PLOTS = True  # todo: UNCOMMENT
    ERRORBAR = ("ci", 95)  # "sd"  #
    LOG_PATH = "/home/r2ci/phd-work/paper/logs"
    ALL_CONFIGS = {"variable": {"t_zoom": 10,
                                "t_final": 16,
                                "yticks_speed": [2, 4, 6, 8],
                                "ylim_traj": [0, 52],
                                "ylim_zoom": [0, 6],
                                "ylim_final": [0, 10],
                                "xticks_traj": [0, 4, 8, 12, 16],
                                "xticks_zoom": [10, 12, 14, 16],
                                },
                   "fixed": {"t_zoom": 19,
                             "t_final": 25,
                             "yticks_speed": [2, 4, 6, 8],
                             "ylim_traj": [0, 52],
                             "ylim_zoom": [0, 6],
                             "ylim_final": [0, 10],
                             "xticks_traj": [0, 5, 10, 15, 20, 25],
                             "xticks_zoom": [19, 21, 23, 25],
                             }
                   }
    ALL_EXP_DIRS = {
        "fixed": [
            f"{LOG_PATH}/2023-12-12-1746_real_2ndcalib_rex_randomeps_MCS_recorded_3Hz_3iter_vx300s",
            f"{LOG_PATH}/2023-12-12-1824_real_2ndcalib_rex_randomeps_topological_recorded_3Hz_3iter_vx300s",
            f"{LOG_PATH}/2023-12-12-1814_real_2ndcalib_rex_randomeps_generational_recorded_3Hz_3iter_vx300s",
            f"{LOG_PATH}/2023-12-12-1834_real_2ndcalib_brax_3Hz_3iter_vx300s",
        ],
        "variable": [
            # Variable + video
            f"{LOG_PATH}/2023-12-14-1058_real_2ndcalib_rex_randomeps_MCS_recorded_VarHz_3iter_record_imagevx300s",
            f"{LOG_PATH}/2023-12-14-1141_real_2ndcalib_rex_randomeps_topological_recorded_VarHz_3iter_record_imagevx300s",
            f"{LOG_PATH}/2023-12-14-1131_real_2ndcalib_rex_randomeps_generational_recorded_VarHz_3iter_record_imagevx300s",
            f"{LOG_PATH}/2023-12-14-1159_real_2ndcalib_brax_VarHz_3iter_record_image_vx300s",
            # Variable
            # f"{LOG_PATH}/2023-12-12-1636_real_2ndcalib_rex_randomeps_MCS_recorded_VarHz_3iter_vx300s",
            # f"{LOG_PATH}/2023-12-12-1718_real_2ndcalib_rex_randomeps_topological_recorded_VarHz_3iter_vx300s",
            # f"{LOG_PATH}/2023-12-12-1708_real_2ndcalib_rex_randomeps_generational_recorded_VarHz_3iter_vx300s",
            # f"{LOG_PATH}/2023-12-12-1734_real_2ndcalib_brax_VarHz_3iter_vx300s",
        ]
    }

    for name in tqdm.tqdm(all_plots_box.keys(), desc="Plotting box pushing"):
        if SKIP_BOX_PUSHING_PLOTS:
            continue
        plots_box = all_plots_box[name]
        EXP_DIRS = ALL_EXP_DIRS[name]
        CONFIGS = ALL_CONFIGS[name]
        df_perf = []
        df_speed = []
        for LOG_DIR in EXP_DIRS:
            with open(f"{LOG_DIR}/eval_data.pkl", "rb") as f:
                data = pickle.load(f)

            # Add performance data to the data frame
            timestamps_tiled = np.tile(data["timestamps"], data["cm"].shape[0])
            for t, cm in zip(timestamps_tiled, data["cm"].flatten()):
                # For each cost, create a dictionary with 'key' and 'cost'
                df_perf.append({"supergraph_type": data["supergraph_type"], "time": t, "cm": cm})

            # Add speed data to the data frame
            if data["supergraph_type"] in ["mcs", "generational", "topological", "deterministic"]:
                df_speed.append({"supergraph_type": data["supergraph_type"], "supergraph/efficiency_percentage": data["efficiency"], "delay": data["delay"],
                                 "speed/fps": 1/data["delay"], "rate": data["rate"]})
        # Convert the list of dictionaries to a DataFrame
        df_perf = pd.DataFrame(df_perf)
        df_speed = pd.DataFrame(df_speed)

        # Relabel
        df_perf["supergraph_type"] = df_perf["supergraph_type"].map(labels)

        # Plot speed
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
        plots_box["speed"] = (fig, ax)
        sns.scatterplot(x="supergraph/efficiency_percentage", y="speed/fps", hue="supergraph_type", palette=fcolor, ax=ax,
                        data=df_speed, legend=True,
                        hue_order=[labels[k] for k in ["deterministic", "mcs", "generational", "topological"]])

        # Filter out rows where 'supergraph_type' is 'deterministic'
        filtered_df = df_speed[df_speed["supergraph_type"] != "deterministic"]

        # Fit a linear regression model using NumPy
        X = filtered_df["supergraph/efficiency_percentage"].values
        y = filtered_df["speed/fps"].values
        coefficients = np.polyfit(X, y, 1)
        Xh = np.linspace(0, 100, 100)  # X
        y_line = coefficients[0] * Xh + coefficients[1]

        # Plot the linear line
        ax.plot(Xh, y_line, color='black', linestyle='--', linewidth=plt.rcParams['lines.linewidth'] * 0.8)

        # Set legend & axis labels
        ax.set_ylim(2, 3.8)
        ax.set_yticks(CONFIGS["yticks_speed"])
        ax.set_xlim(0, 100)
        ax.set_xlabel('efficiency (%)')
        ax.set_ylabel("speed (Hz)")
        # Place the legend on the right outside of the figure
        ax.legend(handles=ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1],
                  # ncol=1, loc='best', bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
                  ncol=4, loc='upper left', fancybox=True, shadow=True, bbox_to_anchor=(2, 2))

        # Plot trajectory
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
        plots_box["traj"] = (fig, ax)
        sns.lineplot(data=df_perf, x="time", y="cm", hue="supergraph_type", ax=ax, legend=True,
                     palette=fcolor, errorbar=ERRORBAR, hue_order=[labels[k] for k in ["mcs", "generational", "topological", "deterministic"]])

        # Set legend & axis labels
        t_final = CONFIGS["t_final"]
        ax.set_ylim(*CONFIGS["ylim_traj"])
        ax.set_xlim(0, t_final)
        ax.set_xticks(CONFIGS["xticks_traj"])
        ax.set_xlabel("time (s)")
        ax.set_ylabel("distance (cm)")
        # ax.legend(handles=None, ncol=1, loc="best", fancybox=True, shadow=True)

        # Plot trajectory
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
        plots_box["zoom"] = (fig, ax)
        sns.lineplot(data=df_perf, x="time", y="cm", hue="supergraph_type", ax=ax, legend=True,
                     palette=fcolor, errorbar=ERRORBAR, hue_order=[labels[k] for k in ["mcs", "generational", "topological", "deterministic"]])

        # Set legend & axis labels
        t_zoom = CONFIGS["t_zoom"]
        ax.set_xlim(t_zoom, t_final)
        ax.set_xticks(CONFIGS["xticks_zoom"])
        ax.set_ylim(*CONFIGS["ylim_zoom"])
        ax.set_xlabel("time (s)")
        ax.set_ylabel("distance (cm)")
        # ax.legend(handles=None, ncol=1, loc="best", fancybox=True, shadow=True)

        # Plot zoomed frame from [t_zoom, t_final] and y-axis from [0, 10]
        # Create a rectangle patch
        import matplotlib.patches as patches
        rect = patches.Rectangle((t_zoom, 0), t_final-t_zoom, 6, linewidth=1, edgecolor=ecolor["zoomed_frame"], facecolor='none')

        # Add the rectangle to the Axes
        plots_box["traj"][1].add_patch(rect)

        # Assign experiment identifiers
        df_perf['experiment_id'] = df_perf.groupby(['supergraph_type', 'time']).cumcount()

        # Function to interpolate for a given group using NumPy
        def interpolate_group(group, t_eval):
            return np.interp(t_eval, group['time'], group['cm'])

        # Apply interpolation for each supergraph_type and experiment
        interpolated_values = df_perf.groupby(['supergraph_type', 'experiment_id']).apply(interpolate_group, t_eval=t_final)

        # Reset index to convert MultiIndex to columns
        df_final = interpolated_values.reset_index()
        df_final.columns = ['supergraph_type', 'experiment_id', 'cm_at_t_final']

        # Plot
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=thirdwidth_figsize)
        plots_box["final"] = (fig, ax)
        sns.barplot(x="supergraph_type", y="cm_at_t_final", hue="supergraph_type", data=df_final, palette=fcolor, ax=ax,
                    legend=True,
                    hue_order=[labels[k] for k in ["mcs", "deterministic", "generational", "topological"]],
                    order=[labels[k] for k in ["mcs", "deterministic", "generational", "topological"]],
                    # order=[k for k in ["sim", "real"]]
                    )

        # Set legend & axis labels
        ax.set_ylim(*CONFIGS["ylim_final"])
        ax.set_xlabel('method')
        # ax.set_yscale("log")
        ax.set_ylabel("distance (cm)")
        # ax.legend(handles=None, ncol=1, loc='best', fancybox=True, shadow=True)

    ###############################################################
    # SAVING
    ###############################################################
    for exp_type, plots_box in all_plots_box.items():
        if any([v is None for v in plots_box.values()]):
            print(f"Skipped: all_plots_box[{exp_type}]")
            continue
        for box_type, (fig, ax) in plots_box.items():
            fig.savefig(f"{PAPER_DIR}/box_{exp_type}_{box_type}{VERSION_ID}_legend{VERSION_ID}.pdf", bbox_inches=export_legend(fig, ax.get_legend()))
            ax.get_legend().remove()
            fig.savefig(f"{PAPER_DIR}/box_{exp_type}_{box_type}{VERSION_ID}.pdf", bbox_inches='tight')
            if MUST_BREAK:
                break
    return all_plots_box


def make_graph_plots():
    # todo: plot pendulum graphs (maybe not necessary?)
    # todo: plot cps graphs
    # todo: define edge and face colors, consistent with the paper.
    # todo: plot mcs, gen and top graphs
    ###############################################################
    # Plot cps plots
    ###############################################################
    GRAPH_ATTRS = dict(node_size=500,
                       node_fontsize=10,
                       edge_linewidth=2.0,
                       node_linewidth=2.5,
                       arrowsize=20,
                       arrowstyle="->",
                       connectionstyle="arc3,rad=0.2", )
    DATA_DIR = "/home/r2ci/supergraph/data"
    ECOLOR = {"v2v-platooning": {"sim": "#6741d9", "main": "#c2255c", "other": "#f08c00"},
              "uav-swarm-control": {"sim": "#6741d9", "main": "#c2255c", "other": "#495057"}}
    FCOLOR = {"v2v-platooning": {"sim": "#d0bfff", "main": "#fcc2d7", "other": "#ffec99"},
              "uav-swarm-control": {"sim": "#d0bfff", "main": "#fcc2d7", "other": "#e9ecef"}}
    NODE_LABELS = {"v2v-platooning": {"sim": "SIM", "main": "L", "other": "F"},
                   "uav-swarm-control": {"sim": "SIM", "main": "C", "other": "UAV"}}

    topology_types = ["uav-swarm-control", "v2v-platooning"]
    plots_computation_graphs = {topology_type: None for topology_type in topology_types}
    plots_mcs_graphs = {topology_type: None for topology_type in topology_types}
    plots_gen_graphs = {topology_type: None for topology_type in topology_types}
    plots_top_graphs = {topology_type: None for topology_type in topology_types}
    for topology_type in topology_types:

        # Load metadata from file and verify that it matches the params_graph
        eps = 0
        metadata = {"seed": 0,
                    "frequency_type": 20,
                    "topology_type": topology_type,
                    "theta": 0.07,
                    "sigma": 0.2,
                    "scaling_mode": "after_generation",
                    "window": 1,
                    "num_nodes": 4,
                    "max_freq": 200,
                    "episodes": 10,
                    "length": 10,
                    "leaf_kind": 0}
        # with open(f"{DATA_DIR}/{name}/metadata.yaml", "r") as f:
        #     metadata_check = yaml.load(f, Loader=yaml.FullLoader)
        # assert all([metadata_check[k] == v for k, v in metadata.items()])

        # Load graph from file
        name = supergraph.evaluate.to_graph_name(**metadata)
        RUN_DIR = f"{DATA_DIR}/{name}"
        EPS_DIR = f"{RUN_DIR}/{eps}"
        assert os.path.exists(EPS_DIR), f"Episode directory does not exist: {EPS_DIR}"
        G_edges = np.load(f"{EPS_DIR}/G_edges.npy")
        G_ts = np.load(f"{EPS_DIR}/G_ts.npy")
        G = supergraph.evaluate.from_numpy(G_edges, G_ts)

        leaf_kind = metadata["leaf_kind"]
        num_nodes = metadata["num_nodes"]

        # Set colors & labels
        labels_map = NODE_LABELS[topology_type]
        labels = {k: f"${labels_map['other']}_{k}$" for k in range(num_nodes)}
        labels[0] = labels_map["main"]
        labels[num_nodes-1] = labels_map["sim"]

        # Get maps
        ecolor_map = ECOLOR[topology_type]
        ecolor = {k: ecolor_map["other"] for k in range(num_nodes)}
        ecolor[0] = ecolor_map["main"]
        ecolor[num_nodes-1] = ecolor_map["sim"]
        fcolor_map = FCOLOR[topology_type]
        fcolor = {k: fcolor_map["other"] for k in range(num_nodes)}
        fcolor[0] = fcolor_map["main"]
        fcolor[num_nodes - 1] = fcolor_map["sim"]

        # Set order
        order = {k: num_nodes-k-1 for k in range(num_nodes)}
        supergraph.set_node_order(G, order)
        supergraph.set_node_colors(G, ecolor=ecolor, fcolor=fcolor)
        # Plot
        S_top, S_gen = supergraph.evaluate.baselines_S(G, leaf_kind)
        S_sup, _S_init_to_S, m_sup = supergraph.grow_supergraph(G, leaf_kind,
                                                        combination_mode="linear",
                                                        backtrack=20,
                                                        sort_fn=None,
                                                        progress_fn=None,
                                                        progress_bar=True,
                                                        validate=False)

        GRAPH_ATTRS = dict(node_size=300,
                           node_fontsize=10,
                           edge_linewidth=2.0,
                           node_linewidth=2.5,
                           arrowsize=10,
                           arrowstyle="-|>",
                           connectionstyle="arc3,rad=0.2", )

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fullwidth_figsize)
        plots_computation_graphs[topology_type] = (fig, ax)
        supergraph.plot_graph(G, max_x=0.5, label_map=labels, ax=ax, draw_labels=False, **GRAPH_ATTRS)
        ax.set_xlim([-0.01, 0.25])
        ax.set_xlabel("time (s)")

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fullwidth_figsize)
        plots_mcs_graphs[topology_type] = (fig, ax)
        supergraph.plot_graph(S_sup, label_map=labels, ax=ax, draw_labels=False, **GRAPH_ATTRS)
        ax.set_xlim([-0.5, 25])
        ax.set_ylim([-0.3, num_nodes-1+0.3])
        ax.set_xlabel("topological generation")

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fullwidth_figsize)
        supergraph.plot_graph(S_top, label_map=labels, ax=ax, draw_labels=False, **GRAPH_ATTRS)
        plots_top_graphs[topology_type] = (fig, ax)
        ax.set_xlim([-0.5, 25])
        ax.set_ylim([-0.3, num_nodes-1+0.3])
        ax.set_xlabel("topological generation")

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fullwidth_figsize)
        supergraph.plot_graph(S_gen, label_map=labels, ax=ax, draw_labels=False, **GRAPH_ATTRS)
        plots_gen_graphs[topology_type] = (fig, ax)
        ax.set_xlim([-0.5, 25])
        ax.set_ylim([-0.3, num_nodes - 1 + 0.3])
        ax.set_xlabel("topological generation")
        print(f"S_sup: {len(S_sup)} nodes | {S_gen.number_of_nodes()} nodes | {S_top.number_of_nodes()} nodes")
        # plt.show()
    ###############################################################
    # PENDULUM PLOTS
    ###############################################################
    # LOG_PATH = "/home/r2ci/phd-work/paper/logs"

    ###############################################################
    # SAVING
    ###############################################################
    for topology_type, (fig, ax) in plots_computation_graphs.items():
        fig.savefig(f"{PAPER_DIR}/computation_graphs_{topology_type}{VERSION_ID}.pdf", bbox_inches='tight')
        if MUST_BREAK:
            break
    for topology_type, (fig, ax) in plots_mcs_graphs.items():
        fig.savefig(f"{PAPER_DIR}/mcs_{topology_type}{VERSION_ID}.pdf", bbox_inches='tight')
        if MUST_BREAK:
            break
    for topology_type, (fig, ax) in plots_top_graphs.items():
        fig.savefig(f"{PAPER_DIR}/top_{topology_type}{VERSION_ID}.pdf", bbox_inches='tight')
        if MUST_BREAK:
            break
    for topology_type, (fig, ax) in plots_gen_graphs.items():
        fig.savefig(f"{PAPER_DIR}/gen_{topology_type}{VERSION_ID}.pdf", bbox_inches='tight')
        if MUST_BREAK:
            break
    return plots_computation_graphs, plots_mcs_graphs, plots_top_graphs, plots_gen_graphs


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
    twothirdwidth_figsize = [c * s for c, s in zip([2 / 3, 0.5], rescaled_figsize)]
    sixthwidth_figsize = [c * s for c, s in zip([1 / 6, 0.5], rescaled_figsize)]
    print("Default figsize:", default_figsize)
    print("Rescaled figsize:", rescaled_figsize)
    print("Fullwidth figsize:", fullwidth_figsize)
    print("Thirdwidth figsize:", thirdwidth_figsize)
    print("Twothirdwidth figsize:", twothirdwidth_figsize)
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
        "v2v-platooning": "v2v",
        "uav-swarm-control": "uav",
        # Combination mode
        "linear": "linear",
        "power": "power",
        # Sort mode
        "optimal": "max-edge",
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
        # "v2v-platooning": "v2v",
        # "uav-swarm-control": "uav",
        # Combination mode
        "linear": "indigo",
        "power": "red",
        # Sort mode
        "arbitrary": "indigo",
        "optimal": "red",
        # line
        "zoomed_frame": "gray",
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
        "deterministic": "orange",
        # Environment
        "real": "indigo",
        "sim": "red",
    }
    cscheme.update({labels[k]: v for k, v in cscheme.items() if k in labels.keys()})
    ecolor, fcolor = oc.cscheme_fn(cscheme)

    import wandb
    api = wandb.Api()

    # Version
    PAPER_DIR = "/home/r2ci/Documents/project/MCS/MCS_RA-L/figures/python"
    VERSION_ID = ""
    VERSION_ID = "_" + VERSION_ID if len(VERSION_ID) else ""  # Prepend _ for readability

    # Make plots
    make_graph_plots()
    plots_pendulum = make_plot_pendulum()
    plots_cps_ablation_efficiency, plots_cps_ablation_time, plot_cps_computational_complexity, plots_cps_transient, plots_cps_perf_size, plots_cps_perf_sigma = make_cps_plots()
    all_plots_box = make_box_pushing_plots()
    plots_ablation_efficiency, plots_ablation_time, plot_computational_complexity, plots_transient, plots_perf_size, plots_perf_sigma = make_abstract_topology_plots()



