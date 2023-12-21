from tqdm import tqdm
import itertools
from functools import partial
from typing import List, Set, Tuple, Union
from collections import deque
import numpy as np
from math import ceil

import networkx as nx

from supergraph import open_colors as oc, as_topological_supergraph

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
    draw_labels=True,
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
    node_labels = {n: data.get("seq", "") for n, data in nodes}

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
    if draw_labels:
        nx.draw_networkx_labels(_G, pos, node_labels, ax=ax, font_size=node_fontsize)

    # Set ticks
    # node_order = {data["kind"]: data["position"][1] for n, data in nodes}
    # yticks = list(node_order.values())
    # ylabels = list(node_order.keys())
    # ax.set_yticks(yticks, labels=ylabels)
    # ax.tick_params(left=False, bottom=True, labelleft=True, labelbottom=True)
    ax.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)


def ornstein_uhlenbeck_samples(rng, theta, mu, sigma, dt, x0, n):
    X = np.zeros(n, dtype="float32")
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
    scaling_mode: str = "after_generation",
    progress_bar: bool = False,
    return_ts: bool = False,
    with_attributes: bool = True,
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
    # Old scaling
    if scaling_mode == "old":
        fn_dt = [
            partial(ornstein_uhlenbeck_samples, rng=rng, theta=theta, mu=1 / f, sigma=sigma / f, dt=1 / f, x0=1 / f)
            for f in fs
        ]
    elif scaling_mode == "after_generation":
        # Scaling samples after generation
        fn_dt = [
            (lambda n, f=f: (1 + ornstein_uhlenbeck_samples(rng=rng, theta=theta, mu=0, sigma=sigma, dt=1, x0=0, n=n)) / f)
            for f in fs
        ]
    else:
        raise ValueError(f"Unknown scaling mode {scaling_mode}")
    # Scaling theta and sigma with dt
    # def scale_theta_sigma(f, n):
    #     dt = 1/f
    #     th_scaled = theta * dt
    #     sig_scaled = sigma * (th_scaled/theta)
    #     X = ornstein_uhlenbeck_samples(rng=rng, theta=th_scaled, mu=1 / f, sigma=sig_scaled, dt=1 / f, x0=1 / f, n=n)
    #     return X
    #
    # fn_dt = [partial(scale_theta_sigma, f) for f in fs]

    # Generate samples
    # for f, fn in zip(fs, fn_dt):
    #     X = fn(n=1000000)
    #     # Print statistics and scaled statistics
    #     est_var = sigma ** 2 / (2 * theta)
    #     est_std = np.sqrt(est_var)
    #     print(f"{f} | Mean: {np.mean(X)}, Std: {np.std(X)} | est_std: {est_std} | scaled(Mean): {np.mean(X)*f}, scaled(Std): {np.std(X)*f}")

    dt = [
        np.stack((np.linspace(1 / f, T, ceil(T * f)), np.clip(fn(n=ceil(T * f)), 1 / f, np.inf)), dtype="float32")
        for f, fn in zip(fs, fn_dt)
    ]
    ts = [np.stack((_dt[0], np.cumsum(_dt[1])), dtype="float32") for _dt in dt]
    ts = [np.concatenate((np.array([[0.0], [0.0]]), _ts), axis=1, dtype="float32") for _ts in ts]

    # Find where entries ts are ts < min_ts
    min_ts = np.min([_ts[1][-1] for _ts in ts])
    idx = [np.where(_ts[1] < min_ts)[0] for _ts in ts]
    ts = [_ts[:, _idx] for _idx, _ts in zip(idx, ts)]

    # import matplotlib.pyplot as plt
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
            data = dict(kind=n, seq=i, ts=__ts)
            if with_attributes:
                data.update(order=n, edgecolor=ecolor[n], facecolor=fcolor[n], position=(__ts, n), alpha=1.0)
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
                data = {}
                if with_attributes:
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
                        data = {}
                        if with_attributes:
                            data.update(**edge_data)
                        G.add_edge(id_seq, id_tar, **data)
                        break
    # Check that G is a DAG
    assert nx.is_directed_acyclic_graph(G), "The graph is not a DAG."
    if return_ts:
        return G, ts
    else:
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
        S, monomorphism = as_topological_supergraph(P, sort=sort)

        yield S, monomorphism


def baselines_S(Gs: Union[nx.DiGraph, List[nx.DiGraph]], leaf_kind, toposorts: List[List] = None):
    Gs = Gs if isinstance(Gs, list) else [Gs]

    if toposorts is None:
        toposorts = []
        for G in Gs:
            toposorts.append(list(nx.topological_sort(G)))

    # Get kinds
    attribute_set = {"inputs", "kind", "order", "edgecolor", "facecolor", "position", "alpha"}
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
    S_top, _ = as_topological_supergraph(S_all, sort=sort)

    # Create supergraph with only the largest depth
    sort_gen = sort[-(largest_depth + 1) :]
    S_gen, _ = as_topological_supergraph(S_all.subgraph([n for gen in sort_gen for n in gen]).copy(), sort=sort_gen)
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


def to_numpy(G: nx.DiGraph) -> Tuple[np.ndarray, np.ndarray]:
    ts = np.empty((len(G.nodes), 2), dtype="float32")
    topo = list(nx.topological_sort(G))
    for i, n in enumerate(topo):
        ts[i, 0] = G.nodes[n]["kind"]
        ts[i, 1] = G.nodes[n]["ts"]
    edges = np.empty((len(G.edges), 2, 2), dtype="uint16")
    for i, (u, v) in enumerate(G.edges):
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        u_kind, u_seq = u_data["kind"], u_data["seq"]
        v_kind, v_seq = v_data["kind"], v_data["seq"]
        edges[i, 0, 0] = u_kind
        edges[i, 0, 1] = u_seq
        edges[i, 1, 0] = v_kind
        edges[i, 1, 1] = v_seq

    # Get bytesize of names and kinds
    # ts_bytesize = reduce(mul, ts.shape, 1) * ts.dtype.itemsize
    # edges_bytesize = reduce(mul, edges.shape, 1) * edges.dtype.itemsize
    # total_bytesize = ts_bytesize + edges_bytesize
    # print(total_bytesize)
    return edges, ts


def from_numpy(edges: np.ndarray, ts: np.ndarray = None) -> nx.DiGraph:
    # Convert edges to graph
    G = nx.DiGraph()
    for i in range(edges.shape[0]):
        u_kind, u_seq = edges[i, 0, 0], edges[i, 0, 1]
        v_kind, v_seq = edges[i, 1, 0], edges[i, 1, 1]
        u = f"{u_kind}_{u_seq}"
        v = f"{v_kind}_{v_seq}"
        G.add_edge(u, v)
        G.nodes[u]["kind"] = u_kind
        G.nodes[u]["seq"] = u_seq
        G.nodes[v]["kind"] = v_kind
        G.nodes[v]["seq"] = v_seq

    # Add timestamps
    if ts is not None:
        kinds = {}
        for i in range(ts.shape[0]):
            k, t = ts[i]
            if k not in kinds:
                kinds[k] = 0
            seq = kinds[k]
            n = f"{int(k)}_{seq}"
            G.nodes[n]["ts"] = t
            kinds[k] += 1
    return G


def to_graph_name(
    seed, frequency_type, topology_type, theta, sigma, scaling_mode, window, num_nodes, max_freq, episodes, length, leaf_kind
):
    name = f"graph-{topology_type}-{frequency_type}-{theta}-{sigma}-{scaling_mode}-{window}-{num_nodes}-{max_freq}-{episodes}-{length}-{leaf_kind}-{seed}"
    return name


def to_rex_supergraph(S: nx.DiGraph, edges: Set[Tuple[int, int]] = None, window: int = 1) -> nx.DiGraph:
    assert window > 0, "Window must be an integer greater than 0"
    if edges is None:
        raise NotImplementedError("Not yet implemented")

    # Prepare node data
    node_data = {}
    for (u, v) in edges:
        if u == v:
            continue
        if u in node_data:
            udata = node_data[u]
        else:
            udata = {"kind": u, "inputs": {}, "stateful": True}
            node_data[u] = udata
        if v in node_data:
            vdata = node_data[v]
        else:
            vdata = {"kind": v, "inputs": {}, "stateful": True}
            node_data[v] = vdata
        vdata["inputs"][u] = {"input_name": u, "window": window}

    topo_sort = list(nx.topological_sort(S))
    for v in topo_sort:
        vdata = S.nodes[v]
        vdata.update(node_data[vdata["kind"]])
        vdata.update({"pruned": False, "super": True})
    return S


def to_rex(G: nx.DiGraph, edges: Set[Tuple[int, int]] = None, window: int = 1) -> nx.DiGraph:
    assert window > 0, "Window must be an integer greater than 0"
    if edges is None:
        raise NotImplementedError("Not yet implemented")

    # Prepare node data
    windowed = {}
    node_data = {}
    for (u, v) in edges:
        if u == v:
            continue
        windowed[(u, v)] = deque(window * [(-1, 0.0)], maxlen=window)
        if u in node_data:
            udata = node_data[u]
        else:
            udata = {"kind": u, "inputs": {}, "stateful": True}
            node_data[u] = udata
        if v in node_data:
            vdata = node_data[v]
        else:
            vdata = {"kind": v, "inputs": {}, "stateful": True}
            node_data[v] = vdata
        vdata["inputs"][u] = {"input_name": u, "window": window}

    topo_sort = list(nx.topological_sort(G))
    for v in topo_sort:
        vdata = G.nodes[v]
        vdata.update(node_data[vdata["kind"]])
        vdata.update({"pruned": False, "super": False, "ts_step": vdata["ts"]})
        # Disconnect input edges
        edges_to_remove = []
        for u, _, edata in G.in_edges(v, data=True):
            udata = G.nodes[u]
            if udata["kind"] == vdata["kind"]:
                # todo: may need to add window, kind to edge data
                edata.update({"stateful": True, "pruned": False})
                continue
            edges_to_remove.append((u, v))
            windowed[(udata["kind"], vdata["kind"])].append((udata["seq"], udata["ts"]))
        G.remove_edges_from(edges_to_remove)

        # Reconnect inputs edges based on window
        for (u, _) in vdata["inputs"].items():
            if (u, vdata["kind"]) in windowed:
                for seq, ts in windowed[(u, vdata["kind"])]:
                    if seq == -1:
                        continue
                    # todo: may need to add window, kind to edge data
                    G.add_edge(f"{u}_{seq}", v, ts_sent=ts, ts_recv=ts, seq=seq, stateful=False, pruned=False)

    return G

    # Add edges to reflect
    # prune to window=x
    # add inputs={kind_name: {input_name: kind_name, window: 1}} to node_data
    # add stateful and pruned to edge_data
