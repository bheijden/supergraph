import numpy as np
import supergraph as sg
import supergraph.evaluate
import networkx as nx
import wandb
from typing import List, Set, Tuple
from tqdm import tqdm
import multiprocessing as mp
from itertools import product
from functools import reduce
from operator import mul
import tempfile
import shutil
import os
import yaml

# os.environ["WANDB_SILENT"] = "false"


# DatasetType = Tuple[List[nx.DiGraph], List[float], Set[int]]

MUST_LOG = True
DATA_DIR = "/home/r2ci/supergraph/data"


# Define a function to generate all episodes for the given parameters
def generate_episodes(seed, frequency_type, topology_type, theta, sigma, window, num_nodes, max_freq, episodes, length, leaf_kind):
    # todo: add communication graph plot to the table?
    # todo: store communication graph edges separately?

    # Create a unique temporary directory
    # temp_dir = tempfile.mkdtemp()  # Create a unique temporary directory
    # artifact_type = f"graph-{topology_type}"
    name = supergraph.evaluate.to_graph_name(seed, frequency_type, topology_type, theta, sigma, window, num_nodes, max_freq, episodes, length, leaf_kind)
    metadata = {"seed": seed,
                "frequency_type": frequency_type,
                "topology_type": topology_type,
                "theta": theta,
                "sigma": sigma,
                "window": window,
                "num_nodes": num_nodes,
                "max_freq": max_freq,
                "episodes": episodes,
                "length": length,
                "leaf_kind": leaf_kind}

    # Write metadata as yaml to file
    RUN_DIR = f"{DATA_DIR}/{name}"
    if MUST_LOG:
        os.mkdir(RUN_DIR)
        with open(f"{RUN_DIR}/metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

    # # Setup wandb
    # run = wandb.init(project="supergraph",
    #                  job_type="generate-graphs",
    #                  mode="offline")
    # artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata)

    # Split seed into seeds for frequency, topology, episode
    rng_freq, rng_topology, rng_episode = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]

    # Determine frequencies
    if frequency_type == "linear":
        fs = [float(i) for i in range(1, num_nodes)] + [float(max_freq)]
    elif frequency_type == "proportional":
        fs = [float(1 + i * (max_freq - 1) / (num_nodes - 1)) for i in range(0, num_nodes)]
    else:
        raise ValueError("Frequency type not supported")

    # Determine edges
    edges = {(i, i) for i in range(len(fs))}  # Stateful edges
    if topology_type == "unidirectional-ring":
        edges.update({(i, (i + 1) % len(fs)) for i in range(len(fs) - 1)})  # Add forward ring edges
        edges.add((len(fs) - 1, 0))  # Add reverse ring edges
    elif topology_type == "bidirectional-ring":
        edges.update({(i, (i + 1) % len(fs)) for i in range(len(fs) - 1)})  # Add forward ring edges
        edges.add((len(fs) - 1, 0))  # Add forward ring edges
        edges.update({(j, i) for i, j in edges})  # Add reverse ring edges
    elif topology_type == "unirandom-ring":
        edges.update({(i, (i + 1) % len(fs)) for i in range(len(fs) - 1)})  # Add forward ring edges
        edges.add((len(fs) - 1, 0))  # Add reverse ring edges
        if len(fs) > 2:
            for i in range(len(fs)):
                j = rng_topology.integers(len(fs)-1)
                c = 0
                for k in range(len(fs)):
                    if (i, k) in edges:
                        continue
                    elif c == j:
                        edges.add((i, k))
                        break
                    else:
                        c += 1
    else:
        raise ValueError("Topology type not supported")

    # Generate episodes
    seeds = rng_episode.integers(low=0, high=np.iinfo(np.int32).max, size=episodes)
    # Gs = []
    for eps, s in tqdm(enumerate(seeds), desc="Generating episodes", disable=True):
        G = sg.evaluate.create_graph(fs, edges, length, seed=s, theta=theta, sigma=sigma, progress_bar=False, return_ts=False, with_attributes=False)
        G = sg.evaluate.prune_by_window(G, window)
        G = sg.evaluate.prune_by_leaf(G, leaf_kind)
        # Gs.append(G)

        # Write edges, ts to file
        G_edges, G_ts = supergraph.evaluate.to_numpy(G)

        # Create directory for episode in temp_dir
        EPS_DIR = f"{RUN_DIR}/{eps}"
        if MUST_LOG:
            os.mkdir(EPS_DIR)
            np.save(f"{EPS_DIR}/G_edges.npy", G_edges)
            np.save(f"{EPS_DIR}/G_ts.npy", G_ts)

        # # Load edges, ts from file
        # G_edges = np.load(f"{EPS_DIR}/G_edges.npy")
        # G_ts = np.load(f"{EPS_DIR}/G_ts.npy")
        # G_numpy = supergraph.evaluate.from_numpy(G_edges, G_ts)
        #
        # # Verify that the graphs are the same
        # assert G.number_of_nodes() == G_numpy.number_of_nodes()
        # assert G.number_of_edges() == G_numpy.number_of_edges()
        #
        # # Check that nodes & edges are the same
        # for u in G.nodes:
        #     if not G.nodes[u]["ts"] == G_numpy.nodes[u]["ts"]:
        #         print(u, G.nodes[u]["ts"], G_numpy.nodes[u]["ts"])
        #         raise ValueError("Node data not the same")
        #     assert G.nodes[u] == G_numpy.nodes[u]
        # for (u, v) in G.edges:
        #     assert G_numpy.has_edge(u, v)

    # # Save artifact
    # artifact.add_dir(local_path=temp_dir)
    # run.log_artifact(artifact)
    #
    # # Close wandb
    # run.finish()

    # # Delete the temporary directories
    # if os.path.exists(temp_dir):
    #     try:
    #         shutil.rmtree(temp_dir)
    #         # print("Temporary directory deleted.")
    #     except OSError as e:
    #         print(f"Failed to delete temporary directory: {str(e)}")
    # return Gs, fs, edges


if __name__ == '__main__':
    wandb.setup()

    # Function inputs
    TOPOLOGY_TYPE = ["unirandom-ring", "bidirectional-ring", "unidirectional-ring"]
    FREQUENCY_TYPE = ["linear"]
    MAX_FREQ = [200]
    WINDOW = [0]
    LEAF_KIND = [0]
    EPISODES = [20]
    LENGTH = [100]
    NUM_NODES = [2, 4, 8, 16, 32, 64]
    SIGMA = [0, 0.1, 0.2, 0.3]
    THETA = [0.07]
    SEED = [0, 1, 2, 3, 4]

    # Generate combinations of all the parameters
    all_params = [SEED, FREQUENCY_TYPE, TOPOLOGY_TYPE, THETA, SIGMA, WINDOW, NUM_NODES, MAX_FREQ, EPISODES, LENGTH, LEAF_KIND]
    param_combinations = product(*all_params)

    # Create a multiprocessing pool
    pool = mp.Pool(mp.cpu_count())

    # Create a progress bar
    total = reduce(mul, [len(p) for p in all_params], 1)
    pbar = tqdm(total=total, desc="Parameter combinations")
    update = lambda *a: pbar.update()

    # Call the function for each combination of parameters using multiprocessing
    for params in param_combinations:
        # tst = generate_episodes(*params)
        # update()
        pool.apply_async(generate_episodes, args=params, callback=update)

    # Close and join the pool
    pool.close()
    pool.join()

    # # Get the results
    # for params, res in results.items():
    #     results[params] = res.get()
    #     print(results[params][0][0])




