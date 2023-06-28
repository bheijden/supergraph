from tqdm import tqdm
import itertools
from functools import partial
from typing import List, Set, Tuple, Union

import numpy as np
from math import floor, ceil

import networkx as nx

from supergraph import open_colors as oc, as_supergraph

edge_data = {"color": oc.ecolor.used, "linestyle": "-", "alpha": 1.0}
pruned_edge_data = {"color": oc.ecolor.pruned, "linestyle": "--", "alpha": 0.5}
delayed_edge_data = {"color": oc.ecolor.pruned, "linestyle": "-", "alpha": 1.0}


def plot_graph(
    ax,
    _G,
    node_size: int = 300,
    node_fontsize=10,
    edge_linewidth=2.0,
    node_linewidth=1.5,
    arrowsize=10,
    arrowstyle="->",
    connectionstyle="arc3,rad=0.1",
):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(nrows=1)
        fig.set_size_inches(12, 5)

    edges = _G.edges(data=True)
    nodes = _G.nodes(data=True)
    # edge_color = [data['color'] for u, v, data in edges]
    # edge_style = [data['linestyle'] for u, v, data in edges]
    edge_color = [data.get("color", edge_data["color"]) for u, v, data in edges]
    edge_alpha = [data.get("alpha", edge_data["alpha"]) for u, v, data in edges]
    edge_style = [data.get("linestyle", edge_data["linestyle"]) for u, v, data in edges]
    node_alpha = [data["alpha"] for n, data in nodes]
    node_ecolor = [data["edgecolor"] for n, data in nodes]
    node_fcolor = [data["facecolor"] for n, data in nodes]
    node_labels = {n: data["seq"] for n, data in nodes}

    # Get positions
    pos = {n: data["position"] for n, data in nodes}

    # Draw graph
    nx.draw_networkx_nodes(
        _G,
        ax=ax,
        pos=pos,
        node_color=node_fcolor,
        alpha=node_alpha,
        edgecolors=node_ecolor,
        node_size=node_size,
        linewidths=node_linewidth,
    )
    nx.draw_networkx_edges(
        _G,
        ax=ax,
        pos=pos,
        edge_color=edge_color,
        alpha=edge_alpha,
        style=edge_style,
        arrowsize=arrowsize,
        arrowstyle=arrowstyle,
        connectionstyle=connectionstyle,
        width=edge_linewidth,
        node_size=node_size,
    )
    nx.draw_networkx_labels(_G, pos, node_labels, ax=ax, font_size=node_fontsize)

    # Set ticks
    # node_order = {data["kind"]: data["position"][1] for n, data in nodes}
    # yticks = list(node_order.values())
    # ylabels = list(node_order.keys())
    # ax.set_yticks(yticks, labels=ylabels)
    # ax.tick_params(left=False, bottom=True, labelleft=True, labelbottom=True)
    ax.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)


def get_excalidraw_graph():
    def _get_ts(_f, _T):
        # Get the number of time steps
        _n = floor(_T * _f) + 1

        # Get the time steps
        _ts = np.linspace(0, (_n - 1) / _f, _n)

        return _ts

    order = ["world", "sensor", "actuator", "agent"]
    order = {k: i for i, k in enumerate(order)}
    cscheme = {"world": "grape", "sensor": "pink", "agent": "teal", "actuator": "orange"}
    ecolor, fcolor = oc.cscheme_fn(cscheme)

    # Graph definition
    T = 0.3
    f = dict(world=20, sensor=20, agent=10, actuator=10)
    phase = dict(world=0.0, sensor=0.2 / 20, agent=0.4 / 20, actuator=1.2 / 20)
    ts = {k: _get_ts(fn, T) + phase[k] for k, fn in f.items()}

    # Initialize a Directed Graph
    G0 = nx.DiGraph()

    # Add nodes
    node_kinds = dict()
    for n in ts.keys():
        node_kinds[n] = []
        for i, _ts in enumerate(ts[n]):
            data = dict(
                kind=n,
                seq=i,
                ts=_ts,
                order=order[n],
                edgecolor=ecolor[n],
                facecolor=fcolor[n],
                position=(_ts, order[n]),
                alpha=1.0,
            )
            id = f"{n}_{i}"
            G0.add_node(id, **data)
            node_kinds[n].append(id)

    # Add edges
    edges = [("world", "sensor"), ("sensor", "agent"), ("agent", "actuator"), ("actuator", "world")]
    for (i, o) in tqdm(edges, desc="Generate graph"):
        for idx, id_seq in enumerate(node_kinds[i]):
            data_source = G0.nodes[id_seq]
            # Add stateful edge
            if idx > 0:
                data = {"delay": 0.0, "pruned": False}
                data.update(**edge_data)
                G0.add_edge(node_kinds[i][idx - 1], id_seq, **data)
            for id_tar in node_kinds[o]:
                data_target = G0.nodes[id_tar]
                if data_target["ts"] >= data_source["ts"]:
                    data = {"delay": 0.0, "pruned": False}
                    data.update(**edge_data)
                    G0.add_edge(id_seq, id_tar, **data)
                    break

    def get_delayed_graph(_G, _delay, _node_kinds):
        _Gd = _G.copy(as_view=False)

        for (source, target, d) in _delay:
            ts_recv = _Gd.nodes[source]["ts"] + d
            undelayed_data = _Gd.edges[(source, target)]
            target_data = _Gd.nodes[target]
            target_kind = target_data["kind"]
            _Gd.remove_edge(source, target)
            for id_tar in _node_kinds[target_kind]:
                data_target = _Gd.nodes[id_tar]
                if data_target["ts"] >= ts_recv:
                    delayed_data = undelayed_data.copy()
                    delayed_data.update(**{"delay": d, "pruned": False})
                    delayed_data.update(**delayed_edge_data)
                    _Gd.add_edge(source, id_tar, **delayed_data)
                    break
        return _Gd

    # Apply delays
    delays_1 = [("actuator_0", "world_2", 0.9 / 20), ("world_1", "sensor_1", 0.9 / 20), ("world_4", "sensor_4", 0.9 / 20)]
    G1 = get_delayed_graph(G0, delays_1, node_kinds)

    # Apply delays
    delays_2 = [("sensor_2", "agent_1", 0.9 / 20), ("actuator_1", "world_4", 0.9 / 20)]
    G2 = get_delayed_graph(G0, delays_2, node_kinds)

    return G0, G1, G2


def run_excalidraw_example():
    # Get excalidraw graph
    G0, G1, G2 = get_excalidraw_graph()

    # Split
    from deprecated import balanced_partition

    root_kind = "agent"
    G0_partition = balanced_partition(G0, root_kind)
    G1_partition = balanced_partition(G1, root_kind)
    G2_partition = balanced_partition(G2, root_kind)

    # Get all topological sorts
    G0_partition_topo = {k: list(nx.all_topological_sorts(G0_partition[k])) for k in G0_partition.keys()}
    G1_partition_topo = {k: list(nx.all_topological_sorts(G1_partition[k])) for k in G1_partition.keys()}
    G2_partition_topo = {k: list(nx.all_topological_sorts(G2_partition[k])) for k in G2_partition.keys()}

    # kind_to_index, unique_kinds = get_unique_kinds(G0)

    # G0_partition_onehot = {k: [topo_to_onehot([G0.nodes[n] for n in g], kind_to_index) for g in G0_partition_topo[k]] for k in G0_partition_topo.keys()}

    # Create new plot
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=3)
    fig.set_size_inches(12, 15)
    plot_graph(axes[0], G0)
    plot_graph(axes[1], G1)
    plot_graph(axes[2], G2)

    fig, axes = plt.subplots(nrows=3)
    fig.set_size_inches(12, 15)
    for idx, G_partition in enumerate([G0_partition, G1_partition, G2_partition]):
        for k, p in G_partition.items():
            # Create new plot
            print(f"Partition {k}")
            plot_graph(axes[idx], p)
    plt.show()

    # Given
    for P in [G0_partition_topo, G1_partition_topo, G2_partition_topo]:
        s = 0
        for k, v in P.items():
            s += len(v)
            print(k, len(v))
        print(s)


def ornstein_uhlenbeck_samples(rng, theta, mu, sigma, dt, x0, n):
    X = np.zeros(n)
    X[0] = x0
    for i in range(1, n):
        dW = rng.normal(0, np.sqrt(dt))
        X[i] = X[i - 1] + theta * (mu - X[i - 1]) * dt + sigma * dW
    return X


def create_graph(
    fs: List[float],
    edges: Set[Tuple[int, int]],
    T: float,
    seed: int = 0,
    theta: float = 0.07,
    sigma: float = 0.1,
    progress_bar: bool = True,
):
    assert all([T > 1 / _f for _f in fs]), "The largest sampling time must be smaller than the simulated time T."

    _all_colors = [
        "red",
        "pink",
        "grape",
        "violet",
        "indigo",
        "blue",
        "cyan",
        "teal",
        "green",
        "lime",
        "yellow",
        "orange",
        "gray",
    ]
    cscheme = {i: _all_colors[i % len(_all_colors)] for i in range(len(fs))}
    ecolor, fcolor = oc.cscheme_fn(cscheme)

    # Function body
    rng = np.random.default_rng(seed=seed)
    fn_dt = [
        partial(ornstein_uhlenbeck_samples, rng=rng, theta=theta, mu=1 / f, sigma=sigma / f, dt=1 / f, x0=1 / f) for f in fs
    ]
    dt = [
        np.stack((np.linspace(1 / f, T, ceil(T * f)), np.clip(fn(n=ceil(T * f)), 1 / f, np.inf))) for f, fn in zip(fs, fn_dt)
    ]
    ts = [np.stack((_dt[0], np.cumsum(_dt[1]))) for _dt in dt]
    ts = [np.concatenate((np.array([[0.0], [0.0]]), _ts), axis=1) for _ts in ts]

    # Find where entries ts are ts < min_ts
    min_ts = np.min([_ts[1][-1] for _ts in ts])
    idx = [np.where(_ts[1] < min_ts)[0] for _ts in ts]
    ts = [_ts[:, _idx] for _idx, _ts in zip(idx, ts)]

    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # [ax.plot(*_dt) for _dt in dt]
    #
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # [ax.plot(_ts[0], _ts[1]) for _ts in ts]
    #
    # plt.show()

    # Initialize a Directed Graph
    G = nx.DiGraph()

    # Add nodes
    node_kinds = dict()
    for n, (_f, _ts) in enumerate(zip(fs, ts)):
        node_kinds[n] = []
        for i, __ts in enumerate(_ts[1]):
            data = dict(
                kind=n, seq=i, ts=__ts, order=n, edgecolor=ecolor[n], facecolor=fcolor[n], position=(__ts, n), alpha=1.0
            )
            id = f"{n}_{i}"
            G.add_node(id, **data)
            node_kinds[n].append(id)

    # Add edges
    iters = 0
    for (o, i) in tqdm(edges, desc="Generate episode", disable=not progress_bar):
        _seq = 0
        for idx, id_seq in enumerate(node_kinds[o]):
            data_source = G.nodes[id_seq]
            if idx > 0 and o == i:  # Add stateful edge
                data = {"delay": 0.0, "pruned": False}
                data.update(**edge_data)
                G.add_edge(node_kinds[i][idx - 1], id_seq, **data)
            else:
                for id_tar in node_kinds[i][_seq:]:
                    iters += 1
                    data_target = G.nodes[id_tar]
                    _seq = data_target["seq"]
                    if o < i:
                        must_connect = data_target["ts"] >= data_source["ts"]
                    else:  # To break cycles
                        must_connect = data_target["ts"] > data_source["ts"]
                    if must_connect:
                        data = {"delay": 0.0, "pruned": False}
                        data.update(**edge_data)
                        G.add_edge(id_seq, id_tar, **data)
                        break
    # Check that G is a DAG
    assert nx.is_directed_acyclic_graph(G), "The graph is not a DAG."
    return G


def prune_by_window(G: nx.DiGraph, window: int = 1) -> nx.DiGraph:
    G = G.copy(as_view=False)
    if window == 0:
        return G
    edges_remove = []
    for n in G.nodes():
        pred = list(G.predecessors(n))
        # Sort by kind
        inputs = {}
        for n_out in pred:
            if G.nodes[n_out]["kind"] not in inputs:
                inputs[G.nodes[n_out]["kind"]] = [n_out]
            else:
                inputs[G.nodes[n_out]["kind"]].append(n_out)
        # Sort by seq number
        for k, v in inputs.items():
            inputs[k] = sorted(v, key=lambda x: G.nodes[x]["seq"], reverse=True)[1:]
        # Remove edges
        for v in inputs.values():
            edges_remove.extend([(n_out, n) for n_out in v])
    G.remove_edges_from(edges_remove)

    return G


def prune_by_leaf(G: nx.DiGraph, leaf_kind) -> nx.DiGraph:
    # Define leafs
    leafs_G = {n: data for n, data in G.nodes(data=True) if data["kind"] == leaf_kind}
    leafs_G = [k for k in sorted(leafs_G.keys(), key=lambda k: leafs_G[k]["seq"])]

    # Remove non-ancestor leafs
    anc = nx.ancestors(G, leafs_G[-1])
    anc.add(leafs_G[-1])
    G.remove_nodes_from([n for n in G.nodes if n not in anc])
    return G


def linear_S_iter(Gs: Union[List[nx.DiGraph], nx.DiGraph]):
    Gs = [Gs] if isinstance(Gs, nx.DiGraph) else Gs
    attribute_set = {"kind", "order", "edgecolor", "facecolor", "position", "alpha"}
    kinds = {}
    for G in Gs:
        kinds.update({G.nodes[n]["kind"]: data for n, data in G.nodes(data=True)})
    kinds = {k: {a: d for a, d in data.items() if a in attribute_set} for k, data in kinds.items()}

    perm_iter = itertools.permutations(kinds.keys(), len(kinds))
    for perm in perm_iter:
        P = nx.DiGraph()
        slots = {k: 0 for k in kinds}
        # monomorphism = dict()
        sort = []
        for n in perm:
            k = n
            s = slots[k]
            # Add monomorphism map
            name = f"s{k}_{s}"
            sort.append(name)
            # monomorphism[n] = name
            # Add node and data
            data = kinds[k].copy()
            data.update({"seq": s})
            P.add_node(name, **data)
            # Increase slot count
            slots[k] += 1
        S, monomorphism = as_supergraph(P, sort=sort)

        yield S, monomorphism


def baselines_S(Gs: Union[nx.DiGraph, List[nx.DiGraph]], leaf_kind, toposorts: List[List] = None):
    Gs = Gs if isinstance(Gs, list) else [Gs]

    if toposorts is None:
        toposorts = []
        for G in Gs:
            toposorts.append(list(nx.topological_sort(G)))

    # Get kinds
    attribute_set = {"kind", "order", "edgecolor", "facecolor", "position", "alpha"}
    kinds = {Gs[0].nodes[n]["kind"]: data for n, data in Gs[0].nodes(data=True)}
    kinds = {k: {a: d for a, d in data.items() if a in attribute_set} for k, data in kinds.items()}

    # Determine partitions
    largest_nodes = 0
    largest_depth = 0
    partitions: List[List[List]] = []
    for i_G, topo in enumerate(toposorts):
        partitions.append([])
        last_leaf = 0
        for i_n, n in enumerate(topo):
            if Gs[i_G].nodes[n]["kind"] == leaf_kind:
                # Determine depth of partition
                P = Gs[i_G].subgraph(topo[last_leaf:i_n])  # Excludes leaf node
                depth = len(list(nx.topological_generations(P)))

                # Store largest depth and number of nodes
                largest_depth = max(largest_depth, depth)
                largest_nodes = max(largest_nodes, len(P))

                # Add partition
                partitions[-1].append(topo[last_leaf : i_n + 1])
                last_leaf = i_n + 1

    # Create two supergraphs
    S_all = nx.DiGraph()
    slots = {k: 0 for k in kinds}
    sort = []
    for _ in range(largest_nodes):
        sort.append([])
        for k, data in kinds.items():
            if k == leaf_kind:
                continue
            s = slots[k]
            # Add monomorphism map
            name = f"s{k}_{s}"
            sort[-1].append(name)
            # Add node and data
            data = kinds[k].copy()
            data.update({"seq": s})
            S_all.add_node(name, **data)
            # Increase slot count
            slots[k] += 1

            # We add feasible edges to previous nodes
            # if i == 0:
            #     continue
            # for k_prev in kinds.keys():
            #     if k_prev == leaf_kind:
            #         continue
            #     if (k_prev, k) in E_val:
            #         for i_prev in range(i):
            #             name_prev = f"s{k_prev}_{i_prev}"
            #             S_all.add_edge(name_prev, name, **edge_data)

    # Add leaf node
    data = kinds[leaf_kind].copy()
    data.update({"seq": 0})
    slots[leaf_kind] += 1
    S_all.add_node(f"s{leaf_kind}_0", **data)
    sort.append([f"s{leaf_kind}_0"])
    # for k_prev in kinds.keys():
    #     if k_prev == leaf_kind:
    #         continue
    #     if (k_prev, leaf_kind) in E_val:
    #         for i_prev in range(largest_nodes):
    #             S_top.add_edge(f"s{k_prev}_{i_prev}", f"s{leaf_kind}_0", **edge_data)
    S_top, _ = as_supergraph(S_all, sort=sort)

    # Create supergraph with only the largest depth
    sort_gen = sort[-(largest_depth + 1) :]
    S_gen, _ = as_supergraph(S_all.subgraph([n for gen in sort_gen for n in gen]).copy(), sort=sort_gen)
    # S_gen = S_top.subgraph([n for gen in generations[:largest_depth] for n in gen] + generations[-1]).copy()
    return S_top, S_gen

    # # Set positions of nodes
    # generations = list(nx.topological_generations(S_top))
    # for i_gen, gen in enumerate(generations):
    #     for i_node, n in enumerate(gen):
    #         S_top.nodes[n]["position"] = (i_gen, i_node)
    #         S_top.nodes[n]["generation"] = i_gen
    #
    # # Set positions of nodes
    # generations = list(nx.topological_generations(S_gen))
    # for i_gen, gen in enumerate(generations):
    #     for i_node, n in enumerate(gen):
    #         S_gen.nodes[n]["position"] = (i_gen, i_node)
    #         S_gen.nodes[n]["generation"] = i_gen
    #
    #
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(nrows=2)
    # fig.set_size_inches(12, 15)
    # plot_graph(axes[0], S_top)
    # plot_graph(axes[1], S_gen)
    # plt.show()


def perfect_sort(P):
    """Sorts a partition into the optimal topological order.

    Only valid if there is never a cycle between kinds within every possible partition.

    E.g. A->B->C->A is allowed if A is the leaf kind, but not if B is the leaf kind (because then the cycle A->C->A may exist).

    NOTE: The sort does not take into account what the leaf node is
    """
    kinds_super = {}
    kinds_new = {}
    for node in P.nodes:
        kind = P.nodes[node]["kind"]
        kinds_dict = kinds_super if node[0] == "s" else kinds_new
        if kind not in kinds_dict:
            kinds_dict[kind] = []
        kinds_dict[kind].append(node)
    # Use a regex that sorts super kind
    kinds_super = {k: sorted(v, key=lambda x: int(x.split("_")[-1])) for k, v in kinds_super.items()}
    kinds_new = {k: sorted(v, key=lambda x: int(x.split("_")[-1])) for k, v in kinds_new.items()}
    kinds = sorted(set(kinds_super.keys()).union(set(kinds_new.keys())), reverse=False)
    sort = []
    for kind in kinds:
        if kind in kinds_super:
            sort += kinds_super[kind]
        if kind in kinds_new:
            sort += kinds_new[kind]
    return sort
