from typing import Dict, Union, Any, Set, Tuple

import networkx as nx
import numpy as np

from supergraph import generate_power_set, _is_node_attr_match


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
    assert (
        min([len(v) for v in node_types.values()]) > 0
    ), "The minimum number of nodes of every type per partition must be > 0."

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
        assert (
            len(nodes[c]["p_edge"]) == nodes[c]["in_degree"]
        ), "The number of edge partitions must be equal to the in_degree of the node."
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
        for _c, t in G.out_edges(c, data=False):
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


def get_set_of_feasible_edges(G: Union[Dict[Any, nx.DiGraph], nx.DiGraph]) -> Set[Tuple[int, int]]:
    # Determine all feasible edges between node types (E_val) = S
    # [OPTIONAL] provide E_val as input instead of looking at all partitions --> Don't forget state-full edges.
    G = G if isinstance(G, dict) else {0: G}
    E_val = set()
    for p in G.values():
        for o, i in p.edges:
            i_type = p.nodes[i]["kind"]
            o_type = p.nodes[o]["kind"]
            E_val.add((o_type, i_type))
    return E_val


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
    for _i_gen_host, gen_host in enumerate(generations_host):
        if len(front) == 0:
            break
        new_front = []
        for n_host in gen_host:
            if len(front) == 0:
                break
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

            for _i_gen_host, gen_host in enumerate(generations_host):
                # NOTE: Assumes that at any point in time, each node in the front is of a different kind
                if len(front) == 0:
                    break
                new_front = []
                for n_host in gen_host:
                    if len(front) == 0:
                        break
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


def as_S(
    P: nx.DiGraph, E_val: Set[Tuple[int, int]], num_topo: int = 1, as_tc: bool = False
) -> Tuple[nx.DiGraph, Dict[Any, Any]]:
    # Grab topological sort of P and add all feasible edges (TC(topo(P)) | e in E_val) = S
    assert num_topo > 0, "num_topo must be greater than 0."
    gen_all_sorts = nx.all_topological_sorts(P)
    S, monomorphism = sort_to_S(P, next(gen_all_sorts), E_val, as_tc=as_tc)

    # [OPTIONAL] Repeat for all topological sorts of P and pick the one with the most feasible edges.
    for _i in range(1, num_topo):
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
    edge_attribute_set = {"color", "linestyle", "alpha"}
    try:
        edge_data = {a: d for a, d in next(iter(P.edges(data=True)))[-1].items() if a in edge_attribute_set}
    except StopIteration:
        # No edges in P
        edge_data = dict()
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
        for _j, n_in in enumerate(sort[i + 1 :]):
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


def check_monomorphism(host: nx.DiGraph, motif: nx.DiGraph, mapping: Dict[str, str]) -> bool:
    check_edges = all(
        [
            host.has_edge(mapping[motif_u], mapping[motif_v])  # todo: only check depth.
            for motif_u, motif_v in motif.edges
            if motif_u in mapping and motif_v in mapping
        ]
    )
    check_nodes = all([_is_node_attr_match(motif_n, host_n, motif, host) for motif_n, host_n in mapping.items()])
    return check_nodes and check_edges
