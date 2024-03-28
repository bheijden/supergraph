import numpy as np
import supergraph as sg
import supergraph.evaluate
import wandb
from tqdm import tqdm
import multiprocessing as mp
from itertools import product
from functools import reduce, partial
from operator import mul
import os
import yaml
import datetime

# TOPOLOGY_TYPE = ["unirandom-ring", "bidirectional-ring", "unidirectional-ring"]
TOPOLOGY_TYPE = ["v2v-platooning", "uav-swarm-control"]
FREQUENCY_TYPE = [20]  # "linear", "proportional", or a fixed number
MAX_FREQ = [200]
WINDOW = [1]
LEAF_KIND = [0]
EPISODES = [10]
LENGTH = [10]
NUM_NODES = [2, 4, 8, 16, 32, 64]  # todo: generate graphs for 2 nodes # [2, 4, 8, 16, 32, 64]  # , 64]
SIGMA = [0., 0.1, 0.2, 0.3]  # [0, 0.1, 0.2, 0.3]  # [0, 0.1, 0.2, 0.3]
THETA = [0.07]
SCALING_MODE = ["after_generation"]
SEED = [0, 1, 2, 3, 4]

# Algorithm inputs
SUPERGRAPH_TYPE = ["mcs", "topological", "generational"]#, "sequential"]
COMBINATION_MODE = ["linear"] #, power]
SORT_MODE = ["arbitrary"]#, "optimal"]
BACKTRACK = [20]  # [0, 10, 15, 20]

# Logging inputs
MUST_LOG = True  # todo:
MULTIPROCESSING = True  # todo:
WORKERS = 12
os.environ["WANDB_SILENT"] = "true"
DATA_DIR = "/home/r2ci/supergraph/data"
PROJECT = "supergraph"
SYNC_MODE = "offline"
GROUP = f"cps-all-noablation-evaluation-{datetime.datetime.today().strftime('%Y-%m-%d-%H%M')}"  # todo: adjust name


def progress_fn(run, Gs_num_nodes, t_final, t_elapsed, Gs_num_partitions, Gs_matched, i_partition, G_monomorphism, G, S):
    supergraph_size = len(S)
    supergraph_nodes = supergraph_size * (i_partition + 1 + sum(Gs_num_partitions))
    matched_nodes = len(G_monomorphism) + sum(Gs_matched)
    efficiency_ratio = matched_nodes / supergraph_nodes
    efficiency_percentage = efficiency_ratio * 100
    matched_ratio = matched_nodes / Gs_num_nodes
    matched_percentage = matched_ratio * 100
    t_final[0] = t_elapsed
    metrics = {"transient/supergraph_size": supergraph_size,
               "transient/supergraph_nodes": supergraph_nodes,
               "transient/matched_nodes": matched_nodes,
               "transient/efficiency_ratio": efficiency_ratio,
               "transient/efficiency_percentage": efficiency_percentage,
               "transient/matched_ratio": matched_ratio,
               "transient/matched_percentage": matched_percentage,
               "transient/t_elapsed": t_elapsed,
               }
    if run is not None:
        run.log(metrics)


def log_final(Gs_num_nodes, S, Gs_monomorphism, t_elapsed):
    Gs_num_partitions = []

    for G_mono in Gs_monomorphism:
        G_num_partitions = 0
        for n_motif, (i, n_host) in G_mono.items():
            G_num_partitions = max(G_num_partitions, i + 1)
        Gs_num_partitions.append(G_num_partitions)
    num_partitions = sum(Gs_num_partitions)
    supergraph_size = len(S)
    supergraph_nodes = supergraph_size * num_partitions
    matched_nodes = sum([len(G_mono) for G_mono in Gs_monomorphism])
    efficiency_ratio = matched_nodes / supergraph_nodes
    efficiency_percentage = efficiency_ratio * 100
    matched_ratio = matched_nodes / Gs_num_nodes
    matched_percentage = matched_ratio * 100
    metrics = {"final/supergraph_size": supergraph_size,
               "final/supergraph_nodes": supergraph_nodes,
               "final/matched_nodes": matched_nodes,
               "final/efficiency_ratio": efficiency_ratio,
               "final/efficiency_percentage": efficiency_percentage,
               "final/matched_ratio": matched_ratio,
               "final/matched_percentage": matched_percentage,
               "final/complete_match": matched_nodes == Gs_num_nodes,
               }
    if t_elapsed is not None:
        metrics["final/t_elapsed"] = t_elapsed
    return metrics


# Define a function to generate all episodes for the given parameters
def evaluate_graph(seed, frequency_type, topology_type, theta, sigma, scaling_mode, window, num_nodes, max_freq, episodes, length, leaf_kind):

    # Create a unique temporary directory
    name = supergraph.evaluate.to_graph_name(seed, frequency_type, topology_type, theta, sigma, scaling_mode, window, num_nodes, max_freq, episodes, length, leaf_kind)
    RUN_DIR = f"{DATA_DIR}/{name}"

    # Load metadata from file and verify that it matches the params_graph
    metadata = {"seed": seed,
                "frequency_type": frequency_type,
                "topology_type": topology_type,
                "theta": theta,
                "sigma": sigma,
                "scaling_mode": scaling_mode,
                "window": window,
                "num_nodes": num_nodes,
                "max_freq": max_freq,
                "episodes": episodes,
                "length": length,
                "leaf_kind": leaf_kind}
    with open(f"{DATA_DIR}/{name}/metadata.yaml", "r") as f:
        metadata_check = yaml.load(f, Loader=yaml.FullLoader)
    assert all([metadata_check[k] == v for k, v in metadata.items()])

    # Load graph from file
    Gs = []
    for i in range(episodes):
        EPS_DIR = f"{RUN_DIR}/{i}"
        assert os.path.exists(EPS_DIR), f"Episode directory does not exist: {EPS_DIR}"
        G_edges = np.load(f"{EPS_DIR}/G_edges.npy")
        G_ts = None # np.load(f"{EPS_DIR}/G_ts.npy")
        G = supergraph.evaluate.from_numpy(G_edges, G_ts)
        Gs.append(G)
    Gs_num_nodes = sum([G.number_of_nodes() for G in Gs])

    # Run evaluation for each supergraph type. Logged as separate runs to wandb.
    for supergraph_type in SUPERGRAPH_TYPE:
        S_top, S_gen = None, None
        if supergraph_type == "mcs":
            # Generate combinations
            all_algorithms = [COMBINATION_MODE, SORT_MODE, BACKTRACK]
            algorithm_combinations = list(product(*all_algorithms))

            for combination_mode, sort_mode, backtrack in algorithm_combinations:
                # Define config
                config = metadata.copy()
                config["supergraph_type"] = supergraph_type
                config["combination_mode"] = combination_mode
                config["sort_mode"] = sort_mode
                config["backtrack"] = backtrack

                # Set sort mode
                if sort_mode == "optimal":
                    # Only run optimal sort for unidirectional-ring (otherwise it is not optimal)
                    if not topology_type == "unidirectional-ring":
                        continue
                    sort_fn = supergraph.evaluate.perfect_sort
                elif sort_mode == "arbitrary":
                    sort_fn = None
                else:
                    raise ValueError(f"Invalid sort mode: {sort_mode}")

                # Setup wandb
                t_final = [0.]
                if MUST_LOG:
                    run = wandb.init(project=PROJECT,
                                     job_type=f"supergraph-{supergraph_type}",
                                     mode=SYNC_MODE,
                                     group=GROUP,
                                     config=config)
                    mcs_progress_fn = partial(progress_fn, run, Gs_num_nodes, t_final)
                else:
                    mcs_progress_fn = partial(progress_fn, None, Gs_num_nodes, t_final)
                    run = None

                # Run evaluation
                S_sup, _S_init_to_S, m_sup = sg.grow_supergraph(Gs, leaf_kind,
                                                                combination_mode=combination_mode,
                                                                backtrack=backtrack,
                                                                sort_fn=sort_fn,
                                                                progress_fn=mcs_progress_fn,
                                                                progress_bar=False,
                                                                validate=False)

                # Get metrics
                metrics = log_final(Gs_num_nodes, S_sup, m_sup, t_final[0])

                # Finish wandb run
                if MUST_LOG:
                    run.log(metrics)
                    run.finish()
                del m_sup, S_sup, _S_init_to_S
        elif supergraph_type == "topological":
            # Get supergraph
            if S_top is None:
                S_top, S_gen = supergraph.evaluate.baselines_S(Gs, leaf_kind)

            # Evaluate supergraph
            m_top = sg.evaluate_supergraph(Gs, S_top, progress_bar=False, name="S_top")

            # Get metrics
            metrics = log_final(Gs_num_nodes, S_top, m_top, None)

            # Log wandb run
            if MUST_LOG:
                # Define config
                config = metadata.copy()
                config["supergraph_type"] = supergraph_type

                run = wandb.init(project=PROJECT,
                                 job_type=f"supergraph-{supergraph_type}",
                                 mode=SYNC_MODE,
                                 group=GROUP,
                                 config=config)
                run.log(metrics)
                run.finish()
            del m_top
        elif supergraph_type == "generational":
            # Get supergraph
            if S_gen is None:
                S_top, S_gen = supergraph.evaluate.baselines_S(Gs, leaf_kind)

            # Evaluate supergraph
            m_gen = sg.evaluate_supergraph(Gs, S_gen, progress_bar=False, name="S_gen")

            # Get metrics
            metrics = log_final(Gs_num_nodes, S_gen, m_gen, None)

            # Log wandb run
            if MUST_LOG:
                # Define config
                config = metadata.copy()
                config["supergraph_type"] = supergraph_type

                run = wandb.init(project=PROJECT,
                                 job_type=f"supergraph-{supergraph_type}",
                                 mode=SYNC_MODE,
                                 group=GROUP,
                                 config=config)
                run.log(metrics)
                run.finish()
            del m_gen
        elif supergraph_type == "sequential":
            # Get supergraph
            linear_iter = supergraph.evaluate.linear_S_iter(Gs)
            S_seq, _ = next(linear_iter)

            # Evaluate supergraph
            m_seq = sg.evaluate_supergraph(Gs, S_seq, progress_bar=False, name="S_seq")

            # Get metrics
            metrics = log_final(Gs_num_nodes, S_seq, m_seq, None)

            # Log wandb run
            if MUST_LOG:
                # Define config
                config = metadata.copy()
                config["supergraph_type"] = supergraph_type

                run = wandb.init(project=PROJECT,
                                 job_type=f"supergraph-{supergraph_type}",
                                 mode=SYNC_MODE,
                                 group=GROUP,
                                 config=config)
                run.log(metrics)
                run.finish()
            del m_seq, S_seq


if __name__ == '__main__':
    if MUST_LOG:
        wandb.setup()

    # Generate combinations of all the necessary graphs
    all_graphs = [SEED, FREQUENCY_TYPE, TOPOLOGY_TYPE, THETA, SIGMA, SCALING_MODE, WINDOW, NUM_NODES, MAX_FREQ, EPISODES, LENGTH, LEAF_KIND]
    graph_combinations = list(product(*all_graphs))
    all_exist = True
    for params in graph_combinations:  # Verify that all graphs are generated in DATA_DIR
        seed, freq_type, topology_type, theta, sigma, scaling_mode, window, num_nodes, max_freq, episodes, length, leaf_kind = params
        name = supergraph.evaluate.to_graph_name(seed, freq_type, topology_type, theta, sigma, scaling_mode, window, num_nodes, max_freq, episodes, length, leaf_kind)
        if not os.path.exists(f"{DATA_DIR}/{name}"):
            print(f"Missing graph: {name}")
            all_exist = False
    assert all_exist, "Not all graphs exist"

    # Create a multiprocessing pool
    pool = mp.Pool(WORKERS)

    # Create a progress bar
    total = reduce(mul, [len(p) for p in all_graphs], 1)
    pbar = tqdm(total=total, desc="Graphs")
    update = lambda *a: pbar.update()

    # Call the function for each combination of parameters using multiprocessing
    for params_graph in graph_combinations:
        if MULTIPROCESSING:
            pool.apply_async(evaluate_graph, args=params_graph, callback=update)
        else:
            evaluate_graph(*params_graph)
            update()

    # Close and join the pool
    pool.close()
    pool.join()

    # # Get the results
    # for params, res in results.items():
    #     results[params] = res.get()
    #     print(results[params][0][0])




