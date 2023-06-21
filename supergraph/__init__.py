__version__ = "0.0.0"

from functools import partial
import itertools
import tqdm
from typing import List, Dict, Tuple, Union, Callable, Set, Any
from math import ceil, floor
from collections import deque
import numpy as np
import supergraph.open_colors as oc
import networkx as nx


edge_data = {"color": oc.ecolor.used, "linestyle": "-", "alpha": 1.}
pruned_edge_data = {"color": oc.ecolor.pruned, "linestyle": "--", "alpha": 0.5}
delayed_edge_data = {"color": oc.ecolor.pruned, "linestyle": "-", "alpha": 1.0}


def plot_graph(ax, _G,
               node_size: int = 300,
               node_fontsize=10,
               edge_linewidth=2.0,
               node_linewidth=1.5,
               arrowsize=10,
               arrowstyle="->",
               connectionstyle="arc3,rad=0.1"):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(nrows=1)
        fig.set_size_inches(12, 5)

    edges = _G.edges(data=True)
    nodes = _G.nodes(data=True)
    # edge_color = [data['color'] for u, v, data in edges]
    # edge_style = [data['linestyle'] for u, v, data in edges]
    edge_color = [data.get('color', edge_data['color']) for u, v, data in edges]
    edge_alpha = [data.get('alpha', edge_data['alpha']) for u, v, data in edges]
    edge_style = [data.get('linestyle', edge_data['linestyle']) for u, v, data in edges]
    node_alpha = [data['alpha'] for n, data in nodes]
    node_ecolor = [data['edgecolor'] for n, data in nodes]
    node_fcolor = [data['facecolor'] for n, data in nodes]
    node_labels = {n: data["seq"] for n, data in nodes}

    # Get positions
    pos = {n: data["position"] for n, data in nodes}

    # Draw graph
    nx.draw_networkx_nodes(_G, ax=ax, pos=pos, node_color=node_fcolor, alpha=node_alpha, edgecolors=node_ecolor,
                           node_size=node_size, linewidths=node_linewidth)
    nx.draw_networkx_edges(_G, ax=ax, pos=pos, edge_color=edge_color, alpha=edge_alpha, style=edge_style,
                           arrowsize=arrowsize, arrowstyle=arrowstyle, connectionstyle=connectionstyle,
                           width=edge_linewidth, node_size=node_size)
    nx.draw_networkx_labels(_G, pos, node_labels, ax=ax, font_size=node_fontsize)

    # Set ticks
    # node_order = {data["kind"]: data["position"][1] for n, data in nodes}
    # yticks = list(node_order.values())
    # ylabels = list(node_order.keys())
    # ax.set_yticks(yticks, labels=ylabels)
    # ax.tick_params(left=False, bottom=True, labelleft=True, labelbottom=True)
    ax.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)


def ancestral_partition(G, root_kind):
    # Copy Graph
    G = G.copy(as_view=False)

    # Get & sort root nodes
    roots = {n: data for n, data in G.nodes(data=True) if data["kind"] == root_kind}
    roots = {k: roots[k] for k in sorted(roots.keys(), key=lambda k: roots[k]["seq"])}

    P = {}
    G_partition = {}
    for k, (root, data) in enumerate(roots.items()):
        ancestors = list(nx.ancestors(G, root))
        ancestors_root = ancestors + [root]

        # Get subgraph
        g = nx.DiGraph()
        g.add_node(root, **G.nodes[root])
        for n in ancestors_root:
            g.add_node(n, **G.nodes[n])
        for (o, i, data) in G.edges(ancestors_root, data=True):
            if o in ancestors_root and i in ancestors_root:
                g.add_edge(o, i, **data)
        # Add generation information to nodes
        generations = nx.topological_generations(g)
        for i_gen, gen in enumerate(generations):
            for n in gen:
                g.nodes[n]["generation"] = i_gen
                # g.nodes[n]["ancestors"] = list(nx.ancestors(g, n))
                # g.nodes[n]["descendants"] = list(nx.descendants(g, n))
        P[k] = g

        # Get subgraph
        G_partition[k] = G.subgraph(ancestors_root).copy(as_view=False)

        # Remove nodes
        G.remove_nodes_from(ancestors_root)

    # for k in P.keys():
    #     assert P[k].edges() == G_partition[k].edges()
    #     for n, data in P[k].nodes(data=True):
    #         for key, val in data.items():
    #             if key == "generation":
    #                 continue
    #             print(key, val==G_partition[k].nodes[n][key])

    return P


def balanced_partition(G, root_kind=None):
    # Copy Graph
    G = G.copy(as_view=False)
    nodes = G.nodes(data=True)

    # Check that graph is DAG
    assert nx.is_directed_acyclic_graph(G), "The graph must be a connected DAG"

    # Check that graph is connected
    assert nx.is_weakly_connected(G), "The graph must be a connected DAG (no separate components)"

    # Count number of nodes of each type
    node_types: Dict[str, Dict[str, Dict]] = {}
    for n, data in G.nodes(data=True):
        if data["kind"] not in node_types:
            node_types[data["kind"]] = dict()
        node_types[data["kind"]][n] = data

    # Sort nodes of each type by sequence number
    for k, v in node_types.items():
        node_types[k] = {k: v[k] for k in sorted(v.keys(), key=lambda k: v[k]["seq"])}

    # Define ideal partition size (either number of root nodes or number of nodes of least common)
    num_partitions = min([len(v) for v in node_types.values()]) if root_kind is None else len(node_types[root_kind])
    size_ideal = {k: -(-len(v) // num_partitions) for k, v in node_types.items()}
    assert min([len(v) for v in node_types.values()]) > 0, "The minimum number of nodes of every type per partition must be > 0."

    # Prepare partitioning
    candidates = []
    for (n, data) in nodes:
        # Initialize every node with its in_degree, p_max=None, and p=None, p_edge=[]
        data_partition = {"in_degree": G.in_degree(n), "p_max": None, "p_edge": {}}
        data.update(**data_partition)

        # Start traversing graph from nodes with in_degree=0 (i.e. add them to the candidate list)
        if data["in_degree"] == 0:
            candidates.append(n)

    # [OPTIONAL] set max_partition to the partition of e.g. an ancestral partition.
    if root_kind is not None:
        G_ancestral = ancestral_partition(G, root_kind)
        for i, _G in G_ancestral.items():
            for n in _G.nodes:
                G.nodes[n]["p_max"] = i

    # Start traversing graph from nodes with in_degree=0.
    G_partition = {}
    sizes = [size_ideal.copy()]
    while len(candidates) > 0:
        # Get next candidate
        c = candidates.pop(0)
        kind = nodes[c]["kind"]

        # Determine partition of incoming edges
        assert len(nodes[c]["p_edge"]) == nodes[c]["in_degree"], "The number of edge partitions must be equal to the in_degree of the node."
        p_edge = max(list(nodes[c]["p_edge"].values()) + [0])  # Add 0 to p_edge to account for nodes with in_degree=0
        if p_edge >= len(sizes):  # If p is larger than the number of partitions, add new partition
            sizes.append(size_ideal.copy())
        p_balanced = p_edge if sizes[p_edge][kind] > 0 else p_edge + 1

        # [OPTIONAL] Overwrite partition of node if max_partition is set
        p = min(p_balanced, nodes[c]["p_max"]) if nodes[c]["p_max"] is not None else p_balanced

        # Update partition of node
        nodes[c].update(p=p)
        G_partition[p] = G_partition.get(p, []) + [c]

        # Update sizes
        if p >= len(sizes):  # If p is larger than the number of partitions, add new partition
            sizes.append(size_ideal.copy())
        sizes[p][kind] -= 1

        # Iterate over outgoing edges
        for _c, t, data in G.out_edges(c, data=True):
            # Add edge partition
            # [OPTIONAL] To ensure that root is always at the end of the partition, add 1 to p if kind == root_kind
            nodes[t]["p_edge"][c] = p if kind != root_kind else p + 1

            # Add node to candidates if in_degree == len(p_edge)
            if nodes[t]["in_degree"] == len(nodes[t]["p_edge"]):
                candidates.append(t)

    # Sort partitions nodes
    G_partition = {k: sorted(v, key=lambda n: n) for k, v in G_partition.items()}

    # Convert partition to subgraph
    P = {}
    for k, v in G_partition.items():
        g = nx.DiGraph()
        for n in v:
            g.add_node(n, **G.nodes[n])
        for (o, i, data) in G.edges(v, data=True):
            if o in v and i in v:
                g.add_edge(o, i, **data)
        # Add generation information to nodes
        generations = nx.topological_generations(g)
        for i_gen, gen in enumerate(generations):
            for n in gen:
                g.nodes[n]["generation"] = i_gen
                # g.nodes[n]["ancestors"] = list(nx.ancestors(g, n))
                # g.nodes[n]["descendants"] = list(nx.descendants(g, n))
                # g.nodes[n]["in_edges"] = list(g.predecessors(n))
                # g.nodes[n]["out_edges"] = list(g.successors(n))
        P[k] = g

    # NOTE! Lines below change order of nodes in subgraphs
    Psub = {k: G.subgraph(v).copy(as_view=False) for k, v in G_partition.items()}
    # Check that all nodes in P and Psub are the same
    assert all([sorted(P[k].nodes) == sorted(Psub[k].nodes) for k in P.keys()]), "Not all nodes in P and Psub are the same."
    # Check that all edges in P and Psub are the same
    assert all([sorted(P[k].edges) == sorted(Psub[k].edges) for k in P.keys()]), "Not all edges in P and Psub are the same."
    return P


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
    phase = dict(world=0., sensor=0.2 / 20, agent=0.4 / 20, actuator=1.2 / 20)
    ts = {k: _get_ts(fn, T) + phase[k] for k, fn in f.items()}

    # Initialize a Directed Graph
    G0 = nx.DiGraph()

    # Add nodes
    node_kinds = dict()
    for n in ts.keys():
        node_kinds[n] = []
        for i, _ts in enumerate(ts[n]):
            data = dict(kind=n, seq=i, ts=_ts, order=order[n], edgecolor=ecolor[n], facecolor=fcolor[n],
                        position=(_ts, order[n]), alpha=1.0)
            id = f"{n}_{i}"
            G0.add_node(id, **data)
            node_kinds[n].append(id)

    # Add edges
    edges = [('world', 'sensor'), ('sensor', 'agent'), ('agent', 'actuator'), ('actuator', 'world')]
    for (i, o) in edges:
        for idx, id_seq in enumerate(node_kinds[i]):
            data_source = G0.nodes[id_seq]
            # Add stateful edge
            if idx > 0:
                data = {"delay": 0., "pruned": False}
                data.update(**edge_data)
                G0.add_edge(node_kinds[i][idx - 1], id_seq, **data)
            for id_tar in node_kinds[o]:
                data_target = G0.nodes[id_tar]
                if data_target['ts'] >= data_source['ts']:
                    data = {"delay": 0., "pruned": False}
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
                if data_target['ts'] >= ts_recv:
                    delayed_data = undelayed_data.copy()
                    delayed_data.update(**{"delay": d, "pruned": False})
                    delayed_data.update(**delayed_edge_data)
                    _Gd.add_edge(source, id_tar, **delayed_data)
                    break
        return _Gd


    # Apply delays
    delays_1 = [("actuator_0", "world_2", 0.9 / 20),
                ("world_1", "sensor_1", 0.9 / 20),
                ("world_4", "sensor_4", 0.9 / 20)]
    G1 = get_delayed_graph(G0, delays_1, node_kinds)

    # Apply delays
    delays_2 = [("sensor_2", "agent_1", 0.9 / 20),
                ("actuator_1", "world_4", 0.9 / 20)]
    G2 = get_delayed_graph(G0, delays_2, node_kinds)

    return G0, G1, G2


def run_excalidraw_example():
    # Get excalidraw graph
    G0, G1, G2 = get_excalidraw_graph()

    # Split
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


def ornstein_uhlenbeck_generator(rng, theta, mu, sigma, dt, x0):
    """Ornstein-Uhlenbeck generator.
    X(t) is the value of the process at time t,
    θ is the speed of reversion to the mean,
    μ is the long-term mean,
    σ is the volatility (i.e. the standard deviation of the changes in the process),
    and dW(t) is a Wiener Process (or Brownian motion).
    """
    X = x0
    while True:
        dW = rng.normal(0, np.sqrt(dt))
        X = X + theta * (mu - X) * dt + sigma * dW
        yield X


def ornstein_uhlenbeck_samples(rng, theta, mu, sigma, dt, x0, n):
    X = np.zeros(n)
    X[0] = x0
    for i in range(1, n):
        dW = rng.normal(0, np.sqrt(dt))
        X[i] = X[i-1] + theta * (mu - X[i-1]) * dt + sigma * dW
    return X


def create_graph(fs: List[float], edges: Set[Tuple[int, int]], T: float, seed: int = 0, theta: float = 0.07, sigma: float = 0.1):
    assert all([T > 1/_f for _f in fs]), "The largest sampling time must be smaller than the simulated time T."

    _all_colors = ['red', 'pink', 'grape', 'violet', 'indigo', 'blue', 'cyan', 'teal', 'green', 'lime', 'yellow', 'orange',  'gray']
    cscheme = {i: _all_colors[i % len(_all_colors)] for i in range(len(fs))}
    ecolor, fcolor = oc.cscheme_fn(cscheme)

    # Function body
    rng = np.random.default_rng(seed=seed)
    fn_dt = [partial(ornstein_uhlenbeck_samples, rng=rng, theta=theta, mu=1 / f, sigma=sigma / f, dt=1 / f, x0=1 / f) for f in fs]
    dt = [np.stack((np.linspace(1/f, T, ceil(T * f)), np.clip(fn(n=ceil(T * f)), 1/f, np.inf))) for f, fn in zip(fs, fn_dt)]
    ts = [np.stack((_dt[0], np.cumsum(_dt[1]))) for _dt in dt]
    ts = [np.concatenate((np.array([[0.], [0.]]), _ts), axis=1) for _ts in ts]

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
            data = dict(kind=n, seq=i, ts=__ts, order=n, edgecolor=ecolor[n], facecolor=fcolor[n],
                        position=(__ts, n), alpha=1.0)
            id = f"{n}_{i}"
            G.add_node(id, **data)
            node_kinds[n].append(id)

    # Add edges
    for (o, i) in edges:
        for idx, id_seq in enumerate(node_kinds[o]):
            _seq = 0
            data_source = G.nodes[id_seq]
            if idx > 0 and o == i:  # Add stateful edge
                data = {"delay": 0., "pruned": False}
                data.update(**edge_data)
                G.add_edge(node_kinds[i][idx - 1], id_seq, **data)
            else:
                for id_tar in node_kinds[i][_seq:]:
                    data_target = G.nodes[id_tar]
                    _seq = data_target['seq']
                    if o < i:
                        must_connect = data_target['ts'] >= data_source['ts']
                    else:  # To break cycles
                        must_connect = data_target['ts'] > data_source['ts']
                    if must_connect:
                        data = {"delay": 0., "pruned": False}
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
    for n, data in G.nodes(data=True):
        pred = list(G.predecessors(n))
        # Sort by kind
        inputs = {}
        for n_out in pred:
            if G.nodes[n_out]['kind'] not in inputs:
                inputs[G.nodes[n_out]['kind']] = [n_out]
            else:
                inputs[G.nodes[n_out]['kind']].append(n_out)
        # Sort by seq number
        for k, v in inputs.items():
            inputs[k] = sorted(v, key=lambda x: G.nodes[x]['seq'], reverse=True)[1:]
        # Remove edges
        for k, v in inputs.items():
            edges_remove.extend([(n_out, n) for n_out in v])
    G.remove_edges_from(edges_remove)

    return G


def get_set_of_feasible_edges(G: Union[Dict[Any, nx.DiGraph], nx.DiGraph]) -> Set[Tuple[int, int]]:
    # Determine all feasible edges between node types (E_val) = S
    # [OPTIONAL] provide E_val as input instead of looking at all partitions --> Don't forget state-full edges.
    G = G if isinstance(G, dict) else {0: G}
    E_val = set()
    for k, p in G.items():
        for o, i in p.edges:
            i_type = p.nodes[i]["kind"]
            o_type = p.nodes[o]["kind"]
            E_val.add((o_type, i_type))
    return E_val


def as_S(P: nx.DiGraph, E_val: Set[Tuple[int, int]], num_topo: int = 1, as_tc: bool = False) -> Tuple[nx.DiGraph, Dict[Any, Any]]:
    # Grab topological sort of P and add all feasible edges (TC(topo(P)) | e in E_val) = S
    assert num_topo > 0, "num_topo must be greater than 0."
    gen_all_sorts = nx.all_topological_sorts(P)
    S, monomorphism = sort_to_S(P, next(gen_all_sorts), E_val, as_tc=as_tc)

    # [OPTIONAL] Repeat for all topological sorts of P and pick the one with the most feasible edges.
    for i in range(1, num_topo):
        try:
            new_S, new_monomorphism = sort_to_S(P, next(gen_all_sorts), E_val, as_tc=as_tc)
        except StopIteration:
            break
        # print(S.number_of_edges(), new_S.number_of_edges())
        if new_S.number_of_edges() > S.number_of_edges():
            S = new_S
            monomorphism = new_monomorphism
    return S, monomorphism


def sort_to_S(P: nx.DiGraph, sort, E_val, as_tc: bool = False) -> Tuple[nx.DiGraph, Dict[Any, Any]]:
    attribute_set = {"kind", "order", "edgecolor", "facecolor", "position", "alpha"}
    kinds = {P.nodes[n]["kind"]: data for n, data in P.nodes(data=True)}
    kinds = {k: {a: d for a, d in data.items() if a in attribute_set} for k, data in kinds.items()}
    S = nx.DiGraph()
    slots = {k: 0 for k in kinds}
    monomorphism = dict()
    for n in sort:
        k = P.nodes[n]["kind"]
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
        for j, n_in in enumerate(sort[i+1:]):
            name_in = monomorphism[n_in]
            e_P = (n_out, n_in)  # Edge in P
            e_S = (name_out, name_in)  # Corresponding edge in S
            e_kind = (S.nodes[name_out]["kind"], S.nodes[name_in]["kind"])  # Kind of edge

            # Add edge if it is in P or if we are adding all feasible edges
            if e_P in P.edges:
                S.add_edge(*e_S, **edge_data)
            elif as_tc and e_kind in E_val:
                S.add_edge(*e_S, **edge_data)

    # Set positions of nodes
    generations = list(nx.topological_generations(S))
    for i_gen, gen in enumerate(generations):
        for i_node, n in enumerate(gen):
            S.nodes[n]["position"] = (i_gen, i_node)
            S.nodes[n]["generation"] = i_gen
    return S, monomorphism


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
            for j, n_in in enumerate(sort[i+1:]):
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


def emb(G1, G2):
    # G1 is a subgraph of G2
    E = [e for e in G1.edges
         if e[0] in G1 and e[1] in G2
         or e[0] in G2 and e[1] in G1]
    return E


def unify(G1, G2, E):
    # E is the edge embedding of G1 in G2
    # G1 is unified with G2 by adding the edges in E
    G = G1.copy(as_view=False)
    G.add_nodes_from(G2.nodes(data=True))
    G.add_edges_from(G2.edges(data=True))
    G.add_edges_from(E)
    return G


def _is_node_attr_match(motif_node_id: str, host_node_id: str, motif: nx.DiGraph, host: nx.DiGraph):
    return host.nodes[host_node_id]["kind"] == motif.nodes[motif_node_id]["kind"]


def check_monomorphism(host: nx.DiGraph, motif: nx.DiGraph, mapping: Dict[str, str]) -> bool:
    check_edges = all([
        host.has_edge(mapping[motif_u], mapping[motif_v])  # todo: only check depth.
        for motif_u, motif_v in motif.edges if motif_u in mapping and motif_v in mapping
    ])
    check_nodes = all([_is_node_attr_match(motif_n, host_n, motif, host) for motif_n, host_n in mapping.items()])
    return check_nodes and check_edges


def generate_power_set(sequence, include_empty_set: bool = True, include_full_set: bool = True):
    power_set = []
    start = 0 if include_empty_set else 1
    stop = len(sequence) + 1 if include_full_set else len(sequence)

    for r in reversed(range(start, stop)):
        combinations = itertools.combinations(sequence, r)
        power_set.extend(combinations)

    return power_set


def find_monomorphism(host: nx.DiGraph, motif: nx.DiGraph) -> Dict[str, str]:
    # Get host generations
    generations_host = list(nx.topological_generations(host))

    # Use first generation of motif as the initial front (i.e. nodes with in_degree=0)
    front = [n for n in motif if motif.in_degree(n) == 0]

    # Prepare matched in_degree counter for motif
    matched_in_degree = {n: 0 for n in motif}

    # Initialize empty monomorphism map
    monomorphism = dict()

    # Iterate over generations
    # NOTE: Assumes that at any point in time, each node in the front is of a different kind
    for i_gen_host, gen_host in enumerate(generations_host):
        if len(front) == 0: break
        new_front = []
        for n_host in gen_host:
            if len(front) == 0: break
            for n_motif in front:
                if _is_node_attr_match(n_motif, n_host, motif, host):
                    monomorphism[n_motif] = n_host
                    front.remove(n_motif)
                    for n_motif_child in motif.successors(n_motif):
                        matched_in_degree[n_motif_child] += 1
                        if motif.in_degree(n_motif_child) == matched_in_degree[n_motif_child]:
                            new_front.append(n_motif_child)
                    break
        front += new_front
    assert check_monomorphism(host, motif, monomorphism)

    if len(monomorphism) != len(motif):
        return []
    else:
        return [monomorphism]


def find_largest_monomorphism(host: nx.DiGraph, motif: nx.DiGraph, max_evals: int = None) -> Tuple[int, Dict[str, str]]:
    # Get host generations
    generations_host = list(nx.topological_generations(host))
    generations_motif = list(nx.topological_generations(motif))

    # Prepare matched in_degree counter for motif
    gen_max_size = [len(motif)]
    gen_matched_in_degree = [{n: 0 for n in motif}]
    for i_gen, gen_motif in enumerate(generations_motif):
        gen_max_size.append(gen_max_size[-1] - len(gen_motif))
        _matched_in_degree = gen_matched_in_degree[i_gen].copy()
        for n_motif in gen_motif:
            for n_motif_child in motif.successors(n_motif):
                _matched_in_degree[n_motif_child] += 1
        gen_matched_in_degree.append(_matched_in_degree)
    assert all([gen_matched_in_degree[-1][n] == motif.in_degree(n) for n in motif]), "Not all nodes in motif have been matched"

    # Initialize empty largest monomorphism map
    largest_monomorphism = dict()
    largest_monomorphism_size = 0

    # Iterate over generations
    num_evals = 0
    short_circuit = False
    for i_gen_motif, gen_motif in enumerate(generations_motif):
        if short_circuit:  # short-circuit if search can never lead to a larger monomorphism or max_evals is reached
            break
        _matched_in_degree = gen_matched_in_degree[i_gen_motif].copy()
        _max_size = gen_max_size[i_gen_motif]

        # Test every possible combinations of nodes in the generation as the initial front
        power_set = generate_power_set(gen_motif, include_empty_set=False)

        # Iterate over possible fronts
        for front in power_set:
            # short-circuit if search can never lead to a larger monomorphism or max_evals is reached
            _max_size = gen_max_size[i_gen_motif] - (len(gen_motif) - len(front))
            if _max_size <= largest_monomorphism_size or (max_evals is not None and num_evals >= max_evals):
                short_circuit = True
                break

            num_evals += 1
            front = list(front)
            monomorphism = dict()
            _matched_in_degree = gen_matched_in_degree[i_gen_motif].copy()
            # Mutate matched_in_degree for nodes excluded in power_set (i.e. nodes not in front but in gen_motif)
            for n_motif in gen_motif:
                if n_motif in front:
                    continue
                for n_motif_child in motif.successors(n_motif):
                    _matched_in_degree[n_motif_child] += 1
                    if motif.in_degree(n_motif_child) == _matched_in_degree[n_motif_child]:
                        front.append(n_motif_child)

            for i_gen_host, gen_host in enumerate(generations_host):
                # NOTE: Assumes that at any point in time, each node in the front is of a different kind
                if len(front) == 0: break
                new_front = []
                for n_host in gen_host:
                    if len(front) == 0: break
                    for n_motif in front:
                        if _is_node_attr_match(n_motif, n_host, motif, host):
                            monomorphism[n_motif] = n_host
                            front.remove(n_motif)
                            for n_motif_child in motif.successors(n_motif):
                                _matched_in_degree[n_motif_child] += 1
                                if motif.in_degree(n_motif_child) == _matched_in_degree[n_motif_child]:
                                    new_front.append(n_motif_child)
                            break
                front += new_front

            if len(monomorphism) > largest_monomorphism_size:
                largest_monomorphism = monomorphism
                largest_monomorphism_size = len(monomorphism)
                assert check_monomorphism(host, motif, largest_monomorphism)

    return num_evals, largest_monomorphism


def evaluate_supergraph(G: nx.DiGraph, S: nx.DiGraph):
    # Get host generations
    generations_S = list(nx.topological_generations(S))

    # Use first generation of motif as the initial front (i.e. nodes with in_degree=0)
    front = [n for n in G if G.in_degree(n) == 0]

    # Prepare matched in_degree counter for motif
    matched_in_degree = {n: 0 for n in G}

    # Initialize empty monomorphism map
    monomorphism = dict()

    # Iterate over generations
    # NOTE: Assumes that at any point in time, each node in the front is of a different kind
    i_unit = 0
    mono_size = len(monomorphism)
    while mono_size < len(G):
        assert len(front) > 0, "No more nodes in front but not all nodes in G have been matched"
        # Iterate over all generations of supergraph
        for i_gen_S, gen_S in enumerate(generations_S):
            if len(front) == 0: break
            new_front = []
            for n_S in gen_S:
                if len(front) == 0: break
                for n_G in front:
                    if _is_node_attr_match(n_G, n_S, G, S):
                        monomorphism[n_G] = f"u{i_unit}_{n_S}"
                        front.remove(n_G)
                        for n_G_child in G.successors(n_G):
                            matched_in_degree[n_G_child] += 1
                            if G.in_degree(n_G_child) == matched_in_degree[n_G_child]:
                                new_front.append(n_G_child)
                        break
            front += new_front

        # Check if new nodes have been matched since last iteration
        assert len(monomorphism) > mono_size, "No new nodes have been matched"
        mono_size = len(monomorphism)

        # Increment unit counter
        i_unit += 1
    return i_unit, i_unit*len(S), monomorphism
    # raise NotImplementedError("TODO: Implement supergraph evaluation")
    # assert check_monomorphism(S, G, monomorphism)


def match_supergraph_iter(G, S, front: Set = None, generations_S: List[List[Any]] = None, matched_in_degree: Dict[Any, int] = None, delta_matched_in_degree: Dict[Any, int] = None):
    # Get host generations
    generations_S = list(nx.topological_generations(S)) if generations_S is None else generations_S

    # Use first generation of motif as the initial front (i.e. nodes with in_degree=0)
    front = [n for n in G if G.in_degree(n) == 0] if front is None else front

    # Prepare matched in_degree counter for motif
    matched_in_degree = {n: 0 for n in G} if matched_in_degree is None else matched_in_degree
    delta_matched_in_degree = {} if delta_matched_in_degree is None else delta_matched_in_degree

    # Iterate over generations
    i = 0
    monomorphism = {}
    monomorphism_size = 0
    while monomorphism_size < len(G):
        for i_gen_S, gen_S in enumerate(generations_S):
            if len(front) == 0:
                break

            new_front = set()
            for n_S in gen_S:
                if len(front) == 0:
                    break
                for n_G in front:
                    if _is_node_attr_match(n_G, n_S, G, S):
                        monomorphism[n_G] = n_S
                        front.remove(n_G)

                        for n_G_child in G.successors(n_G):
                            matched_in_degree[n_G_child] += 1
                            delta_matched_in_degree[n_G_child] = delta_matched_in_degree.get(n_G_child, 0) + 1
                            if G.in_degree(n_G_child) == matched_in_degree[n_G_child]:
                                new_front.add(n_G_child)

                        break

            front = front.union(new_front)

        # Check if new nodes have been matched since last iteration
        if len(monomorphism) == 0:
            yield i, front, monomorphism
            raise StopIteration("No new nodes have been matched. Does the supergraph S have at least one node of each kind?")

        # Increment unit counter
        monomorphism_size += len(monomorphism)
        i += 1

        yield i, front, monomorphism
    return


def grow_supergraph_iter(G: nx.DiGraph, S: nx.DiGraph, leaf_kind, E_val, backtrack: int = 3):
    # Get & sort leaf nodes
    leafs_G = {n: data for n, data in G.nodes(data=True) if data["kind"] == leaf_kind}
    leafs_G = [k for k in sorted(leafs_G.keys(), key=lambda k: leafs_G[k]["seq"])]
    num_partitions = len(leafs_G)
    assert num_partitions > 0, f"No leaf nodes of kind {leaf_kind} found in G"

    # Get & sort leaf nodes
    leafs_S = {n: data for n, data in S.nodes(data=True) if data["kind"] == leaf_kind}
    assert len(leafs_S) == 1, f"More than one leaf node of kind {leaf_kind} found in S"
    assert S.out_degree[list(leafs_S.keys())[0]] == 0, f"Leaf node of kind {leaf_kind} in S has out_degree > 0"
    leaf_S = list(leafs_S.keys())[0]

    # Get host generations (excludes leaf node from generations)
    num_nodes_S = len(S) - 1  # Excluding the leaf node
    generations_S = list(nx.topological_generations(S.subgraph(set(S.nodes()) - {leaf_S})))

    # Use first generation of motif as the initial front (i.e. nodes with in_degree=0)
    next_front = {n for n in G if G.in_degree(n) == 0}

    # Prepare matched in_degree counter for motif
    matched_in_degree = {n: 0 for n in G}

    # Initialize empty monomorphism map
    G_monomorphism = dict()

    # Initialize supergraph mapping
    S_init_to_S = {n: n for n in S}

    # Iterate over leafs
    i_backtrack = 0
    i_partition = 0
    G_unmatched = G.copy(as_view=False)
    state_history = deque(maxlen=backtrack+1)
    while i_partition < num_partitions:
        assert len(next_front) > 0, "No more nodes in next_front but not all nodes in G have been matched"

        # Get current leaf node
        leaf_k = leafs_G[i_partition]

        # Store current search state
        delta_matched_in_degree = dict()
        delta_G_monomorphism = dict()
        state = dict(i_partition=i_partition,
                     next_front=next_front.copy(),
                     delta_matched_in_degree=delta_matched_in_degree,
                     delta_G_monomorphism=delta_G_monomorphism)
        state_history.append(state)

        # Match nodes in G to nodes in S
        front = next_front.copy()
        _, front, monomorphism = next(match_supergraph_iter(G, S, front, generations_S, matched_in_degree, delta_matched_in_degree))

        # Determine nodes in G that are ancestors of current leaf node (i.e. that must be matched before leaf node)
        ancestors = nx.ancestors(G_unmatched, leaf_k)

        # Determine matched nodes in ancestors
        matched_nodes = ancestors.intersection(monomorphism.keys())
        if len(matched_nodes) == len(ancestors):  # Perfect match
            # All ancestors are matched, so we can proceed to next leaf node
            assert leaf_k in front, "leaf_k not in front"
            next_front = front  # Make sure leaf_S is not included in next front.
            next_front.remove(leaf_k)
            monomorphism[leaf_k] = leaf_S
            for n_G_child in G.successors(leaf_k):
                matched_in_degree[n_G_child] += 1
                delta_matched_in_degree[n_G_child] = delta_matched_in_degree.get(n_G_child, 0) + 1
                if G.in_degree(n_G_child) == matched_in_degree[n_G_child]:
                    next_front.add(n_G_child)

            # Add matched nodes to G_monomorphism
            G_monomorphism.update(monomorphism)
            delta_G_monomorphism.update(monomorphism)

            # Remove largest_monomorphism nodes from G_unmatched
            G_unmatched.remove_nodes_from(monomorphism.keys())

            # Reduce i_backtrack
            i_backtrack = max(i_backtrack - 1, 0)

            # Only yield if we have matched a new partition
            if i_backtrack == 0:
                yield i_partition, G_unmatched, S, G_monomorphism, monomorphism,  S_init_to_S

            # Increment i_partition
            i_partition += 1
        else:  # Not all ancestors are matched --> find largest match.
            assert i_backtrack == 0, "i_backtrack should be 0"

            # In general, the search works as follows.
            # Get a set of the "exploration front" of the motif (G) -- nodes that are not
            # yet assigned in a partition but are connected to at least one assigned
            # node in previous partitions.

            # For example, in the motif A -> B -> C, if A is already assigned, then the
            # front is [B] (C is not included because it has no connection to any
            # assigned node of a previous partition).

            # The front is split into:
            # - constrained_front: nodes that are ancestors of the next leaf_k, and must be matched.
            # - optional_front: nodes that are not a direct ancestors, and can optionally be matched.
            # The constrained_front either contains leaf_k or is non-empty.

            # In search of the largest match, we systematically exclude nodes from the constrained_front and try to find a match.
            # If no match is found, we exclude more nodes from the constrained_front and try again.

            # There was no perfect match, so we need to reset matched_in_degree with delta_matched_in_degree
            for k, v in delta_matched_in_degree.items():
                matched_in_degree[k] -= v
            delta_matched_in_degree = dict()

            # Short-circuit if no larger match is possible
            short_circuit = False
            largest_size = 0
            largest_monomorphism = {}
            excluded_from_search = set()
            while len(excluded_from_search) < len(ancestors):
                # See other short_circuit statement for explanation.
                if short_circuit:  # short-circuit if search can never lead to a larger match
                    break

                # The front is split into:
                # - constrained_front: nodes that are ancestors of the next leaf_k, and must be matched.
                # - optional_front: nodes that are not a direct ancestors, and can optionally be matched.
                constrained_front = ancestors.intersection(next_front)
                optional_front = next_front.difference(constrained_front)

                # In search of the largest match, we systematically exclude nodes from the constrained_front and try to find a match.
                # If no match is found, we exclude more nodes from the constrained_front and try again.
                assert len(constrained_front) > 0, "what if constrained_front is empty?"
                power_set = generate_power_set(constrained_front, include_empty_set=False)
                for var_front in power_set:
                    var_front = set(var_front)

                    # We short-circuit the largest search when we cannot find a larger match. This may happen in two cases:
                    #  - All nodes in S have been matched.
                    #  - As the search continues, more nodes are excluded, making the maximum possible match smaller than the
                    #    largest match we've found up until now.
                    _max_size = len(ancestors) - (len(excluded_from_search) + len(constrained_front) - len(var_front))
                    if num_nodes_S == largest_size or _max_size <= largest_size:
                        short_circuit = True
                        break

                    # Use union of var_front and optional_front as front
                    front = var_front.union(optional_front)

                    # Mutate matched_in_degree for nodes excluded in power_set (i.e. nodes not in constrained_front but in var_front)
                    _delta_matched_in_degree = dict()  # Record difference in matched degree
                    for n_G in constrained_front.difference(var_front):
                        for n_G_child in G.successors(n_G):
                            matched_in_degree[n_G_child] += 1
                            _delta_matched_in_degree[n_G_child] = _delta_matched_in_degree.get(n_G_child, 0) + 1
                            if G.in_degree(n_G_child) == matched_in_degree[n_G_child]:
                                front.add(n_G_child)

                    # Match nodes in G to nodes in S
                    _, front, monomorphism = next(match_supergraph_iter(G, S, front, generations_S, matched_in_degree, _delta_matched_in_degree))

                    # Reset matched_in_degree with _delta_matched_in_degree
                    for k, v in _delta_matched_in_degree.items():
                        matched_in_degree[k] -= v

                    # Determine matched nodes in ancestors
                    matched_nodes = ancestors.intersection(monomorphism.keys())
                    matched_size = len(matched_nodes)

                    if matched_size > largest_size:
                        largest_monomorphism = monomorphism
                        largest_size = matched_size

                # Mutate matched_in_degree for nodes excluded in power_set (i.e. nodes not in constrained_front but in var_front)
                excluded_from_search = excluded_from_search.union(constrained_front)
                next_front = optional_front
                for n_G in constrained_front:
                    for n_G_child in G.successors(n_G):
                        matched_in_degree[n_G_child] += 1
                        delta_matched_in_degree[n_G_child] = delta_matched_in_degree.get(n_G_child, 0) + 1
                        if G.in_degree(n_G_child) == matched_in_degree[n_G_child]:
                            next_front.add(n_G_child)

            # Add leaf node to largest monomorphism
            largest_monomorphism[leaf_k] = leaf_S

            # Determine the constrained subgraph P that must be matched
            nodes_P = ancestors.union(largest_monomorphism.keys())
            P = G.subgraph(nodes_P)
            assert check_monomorphism(S, P, largest_monomorphism)

            # Create mapping of nodes from S to P
            mcs = {node_S: node_P for node_P, node_S in largest_monomorphism.items()}

            # Extract subgraphs based on the mappings
            mcs_S = S.subgraph(mcs.keys())
            mcs_P = P.subgraph(mcs.values())

            # Compute embeddings of mcs_S and mcs_P
            E1 = emb(mcs_S, S)  # Embedding of mcs_S in S
            E2 = emb(mcs_P, P)  # Embedding of mcs_P in P

            # Unify the maximum common subgraph with the original graph
            mcs_P_unified_with_P = unify(mcs_P, P, E2)

            # Relabel the nodes of the unified graph using the largest monomorphism
            relabelled_graph = nx.relabel_nodes(mcs_P_unified_with_P, largest_monomorphism, copy=True)

            # Unify the relabelled graph with the second graph
            P_unified_with_S = unify(relabelled_graph, S, E1)

            # Check that the number of added nodes is equal to the number of nodes
            num_new_nodes = len(P_unified_with_S.nodes()) - len(S.nodes())
            check_num_new_nodes = len(P.nodes()) - len(largest_monomorphism)
            assert num_new_nodes == check_num_new_nodes, f"{num_new_nodes} != {check_num_new_nodes}"

            # Verify that P_unified_with_S is a DAG
            assert P_unified_with_S.out_degree(leaf_S) == 0, "Should be a leaf node."
            assert nx.is_directed_acyclic_graph(P_unified_with_S), "Should be a DAG."

            # Convert P_unified_with_S to S
            topo_sort = list(nx.topological_sort(P_unified_with_S.subgraph([n for n in P_unified_with_S.nodes() if n != leaf_S])))
            topo_sort.append(leaf_S)  # Ensure that leaf_S is the last node in the sort of P_unified_with_S
            new_S, new_mono = sort_to_S(P_unified_with_S, topo_sort, E_val, as_tc=True)
            assert S.out_degree[list(leafs_S.keys())[0]] == 0, f"Leaf node of kind {leaf_kind} in S has out_degree > 0"

            # Update S and related variables
            S_to_new_S = {node_p: node_S for node_p, node_S in new_mono.items() if node_p in S.nodes}
            S_init_to_S = {node_S_init: node_S for node_S_init, node_S in S_init_to_S.items()}
            S = new_S
            num_nodes_S = len(S) - 1  # Excluding the leaf node
            generations_S = list(nx.topological_generations(S.subgraph(set(S.nodes()) - {leaf_S})))

            # Reset search state
            i_partition, next_front = state_history[0]["i_partition"], state_history[0]["next_front"]
            state_history[-1]["delta_matched_in_degree"] = delta_matched_in_degree
            for s in state_history:
                _, _, delta_matched_in_degree, delta_G_monomorphism = s.values()
                # Reset matched_in_degree with delta_matched_in_degree from previous iterations
                for k, v in delta_matched_in_degree.items():
                    matched_in_degree[k] -= v
                # Reset G_monomorphism with delta_G_monomorphism from previous iterations
                for k in delta_G_monomorphism.keys():
                    G_monomorphism.pop(k)

            # Update G_monomorphism from S to new_S
            G_monomorphism = {node_G: S_to_new_S[node_S] for node_G, node_S in G_monomorphism.items()}

            # Reinitialize G_unmatched with nodes not already matched
            G_unmatched = G.copy()
            G_unmatched.remove_nodes_from(G_monomorphism.keys())

            # Now that we have a new S, we should backtrack the search.
            i_backtrack = len(state_history)  # Make sure we don't end up in an infinite loop.

            # Reset state history
            state_history = deque(maxlen=backtrack+1)


class GrowSuperGraphIter:
    def __init__(self, G: nx.DiGraph, S: nx.DiGraph, leaf_kind, E_val, backtrack: int = 3):
        leafs_G = {n: data for n, data in G.nodes(data=True) if data["kind"] == leaf_kind}
        self._num_partitions = len(leafs_G)
        self.iter = grow_supergraph_iter(G, S, leaf_kind, E_val, backtrack)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iter)

    def __len__(self):
        return self._num_partitions


def grow_supergraph(G: nx.DiGraph, S: nx.DiGraph, leaf_kind, E_val, backtrack: int = 3, progress_bar: bool = True):
    grow_iter = GrowSuperGraphIter(G, S, leaf_kind, E_val, backtrack=backtrack)
    num_nodes = len(G)
    pbar = tqdm.tqdm(grow_iter) if progress_bar else grow_iter

    S_init_to_S = {}
    G_monomorphism = {}
    for i_partition, _G_unmatched, S, G_monomorphism, _monomorphism, S_init_to_S in pbar:
        if progress_bar:
            size = len(S)
            supergraph_nodes = size*(i_partition+1)
            matched_nodes = len(G_monomorphism)
            efficiency = matched_nodes/supergraph_nodes
            pbar.set_postfix_str(f"matched {matched_nodes}/{num_nodes} ({efficiency:.2%} efficiency, {size} size)")

    return S, S_init_to_S, G_monomorphism
