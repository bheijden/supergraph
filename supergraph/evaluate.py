import itertools
from functools import partial
from typing import List, Set, Tuple

import numpy as np
from math import floor, ceil

import networkx as nx

from supergraph import open_colors as oc

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
    for (i, o) in edges:
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
    fs: List[float], edges: Set[Tuple[int, int]], T: float, seed: int = 0, theta: float = 0.07, sigma: float = 0.1
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
    for (o, i) in edges:
        for idx, id_seq in enumerate(node_kinds[o]):
            _seq = 0
            data_source = G.nodes[id_seq]
            if idx > 0 and o == i:  # Add stateful edge
                data = {"delay": 0.0, "pruned": False}
                data.update(**edge_data)
                G.add_edge(node_kinds[i][idx - 1], id_seq, **data)
            else:
                for id_tar in node_kinds[i][_seq:]:
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


def linear_S_iter(G: nx.DiGraph, E_val):
    attribute_set = {"kind", "order", "edgecolor", "facecolor", "position", "alpha"}
    kinds = {G.nodes[n]["kind"]: data for n, data in G.nodes(data=True)}
    kinds = {k: {a: d for a, d in data.items() if a in attribute_set} for k, data in kinds.items()}

    perm_iter = itertools.permutations(kinds.keys(), len(kinds))
    for sort in perm_iter:
        S = nx.DiGraph()
        slots = {k: 0 for k in kinds}
        monomorphism = dict()
        for n in sort:
            k = n
            s = slots[k]
            # Add monomorphism map
            name = f"s{k}_{s}"
            monomorphism[n] = name
            # Add node and data
            data = kinds[k].copy()
            data.update({"seq": s})
            S.add_node(name, **data)
            # Increase slot count
            slots[k] += 1

        # Add feasible edges
        for i, n_out in enumerate(sort):
            name_out = monomorphism[n_out]
            for _j, n_in in enumerate(sort[i + 1 :]):
                name_in = monomorphism[n_in]
                e_S = (name_out, name_in)  # Corresponding edge in S
                e_kind = (S.nodes[name_out]["kind"], S.nodes[name_in]["kind"])  # Kind of edge

                # Add all feasible edges
                if e_kind in E_val:
                    S.add_edge(*e_S, **edge_data)

        # Set positions of nodes
        generations = list(nx.topological_generations(S))
        for i_gen, gen in enumerate(generations):
            for i_node, n in enumerate(gen):
                S.nodes[n]["position"] = (i_gen, i_node)
                S.nodes[n]["generation"] = i_gen
        yield S, monomorphism
