__version__ = "0.0.4"

import itertools
import tqdm
import time
from typing import List, Dict, Tuple, Set, Any, Union, Callable
from collections import deque
import networkx as nx


def _convert_dict(input_dict):
    # This will hold the list of dictionaries
    result = []

    for key, value in input_dict.items():
        # The index is the first item in the tuple
        index = value[0]

        # If we don't have a dictionary for this index, create one
        while index >= len(result):
            result.append({})

        # Add the key-value mapping to the appropriate dictionary
        result[index][key] = value[1]  # Using value[1] as we want the second item of the tuple

    return result


def _is_node_attr_match(motif_node_id: str, host_node_id: str, motif: nx.DiGraph, host: nx.DiGraph):
    return host.nodes[host_node_id]["kind"] == motif.nodes[motif_node_id]["kind"]


def _count_elements(input_list):
    count_dict = {}
    for elem in input_list:
        if elem in count_dict:
            count_dict[elem] += 1
        else:
            count_dict[elem] = 1
    return count_dict


def _generate_coordinates(input_list):
    count_dict = _count_elements(input_list)
    coordinates_dict = {}

    for unique_val, count in count_dict.items():
        coordinates = []
        if count == 1:
            coordinates.append(unique_val)
        else:
            for i in range(count):
                coordinate = unique_val - 0.25 + 0.5 * (i / (count - 1))
                coordinates.append(coordinate)
        coordinates_dict[unique_val] = coordinates

    return coordinates_dict


def format_supergraph(S: nx.DiGraph) -> nx.DiGraph:
    generations = nx.topological_generations(S)
    for idx_gen, gen in enumerate(generations):
        positions = [round(S.nodes[u]["order"]) for u in gen]
        try:
            positions = _generate_coordinates(positions)
        except BaseException:
            print("Error in generating coordinates. Probably, because the 'kind' is not a number.")
            raise
        for idx_layer, u in enumerate(gen):
            p = positions[S.nodes[u]["order"]].pop(0)
            S.nodes[u]["position"] = (idx_gen, p)
    return S


def as_topological_supergraph(P: nx.DiGraph, leaf_kind=None, sort: List = None, sort_fn: Callable = None):
    attribute_set = {"kind", "inputs", "order", "edgecolor", "facecolor", "position", "alpha"}
    kinds = {P.nodes[n]["kind"]: data for n, data in P.nodes(data=True)}
    kinds = {k: {a: d for a, d in data.items() if a in attribute_set} for k, data in kinds.items()}
    edge_attribute_set = {"color", "linestyle", "alpha"}
    try:
        edge_data = {a: d for a, d in next(iter(P.edges(data=True)))[-1].items() if a in edge_attribute_set}
    except StopIteration:
        # No edges in P
        edge_data = dict()

    # Generate a topological sort if none is provided
    sort_fn = sort_fn or (lambda P: list(nx.topological_sort(P)))
    if sort is None:
        sort = sort_fn(P)
        if leaf_kind is not None:
            for n in reversed(sort):
                if P.nodes[n]["kind"] == leaf_kind:
                    sort.remove(n)
                    sort.append(n)

    # Convert sort to gen
    for i, s in enumerate(sort):
        if not isinstance(s, list):
            sort[i] = [s]
    sort: List[List]

    # Add nodes
    S = nx.DiGraph()
    slots = {k: 0 for k in kinds}
    monomorphism = dict()
    for i, gen in enumerate(sort):
        for n in gen:
            k = P.nodes[n]["kind"]
            s = slots[k]
            # Add monomorphism map
            name = f"s{k}_{s}"
            monomorphism[n] = name
            # Add node and data
            data = kinds[k].copy()
            order = P.nodes[n]["order"]
            data.update({"seq": s, "generation": i, "position": (i, order)})
            S.add_node(name, **data)
            # Increase slot count
            slots[k] += 1
            # Add edge to previous node
            if i > 0:
                for n_prev in sort[i - 1]:
                    S.add_edge(monomorphism[n_prev], name, **edge_data)

    generations = list(nx.topological_generations(S))
    assert all([S.nodes[g[0]]["generation"] == i_gen for i_gen, g in enumerate(generations)])
    assert leaf_kind is None or S.out_degree[f"s{leaf_kind}_0"] == 0, f"Leaf node of kind {leaf_kind} in S has out_degree > 0"
    return S, monomorphism


def as_compact_supergraph(Gs: List[nx.DiGraph], S_init: nx.DiGraph, S: nx.DiGraph, monomorphisms):
    # Determine edge data
    try:
        edge_attribute_set = {"color", "linestyle", "alpha"}
        edge_data = {a: d for a, d in next(iter(S.edges(data=True)))[-1].items() if a in edge_attribute_set}
    except StopIteration:
        # No edges in S
        edge_data = dict()

    # Convert monomorphisms to list of dicts
    monomorphisms = [_convert_dict(m) for m in monomorphisms]

    # Determine set of edges
    edges = list(S_init.edges())
    for G, monomorphism in zip(Gs, monomorphisms):
        for partition in monomorphism:
            G_sub = G.subgraph(partition.keys())
            G_relabeled = nx.relabel_nodes(G_sub, partition, copy=True)
            edges.extend(list(G_relabeled.edges()))
    edges = set(edges)
    # Create supergraph with nodes of S
    S_new = nx.DiGraph()
    S_new.add_nodes_from(S.nodes(data=True))
    S_new.add_edges_from(edges, **edge_data)
    S_new = format_supergraph(S_new)
    return S_new


def emb(G1, G2):
    # G1 is a subgraph of G2
    E = [e for e in G1.edges if e[0] in G1 and e[1] in G2 or e[0] in G2 and e[1] in G1]
    return E


def unify(G1, G2, E):
    # E is the edge embedding of G1 in G2
    # G1 is unified with G2 by adding the edges in E
    G = G1.copy(as_view=False)
    G.add_nodes_from(G2.nodes(data=True))
    G.add_edges_from(G2.edges(data=True))
    G.add_edges_from(E)
    return G


def check_order(host: nx.DiGraph, motif: nx.DiGraph, mapping: Dict[str, str]) -> bool:
    check_edges = all(
        [
            host.nodes[mapping[motif_u]]["generation"] < host.nodes[mapping[motif_v]]["generation"]  # todo: only check depth.
            for motif_u, motif_v in motif.edges
            if motif_u in mapping and motif_v in mapping
        ]
    )
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


def generate_linear_set(sequence, include_empty_set: bool = True, include_full_set: bool = True):
    power_set = []
    start = 0 if include_empty_set else 1
    stop = len(sequence) + 1 if include_full_set else len(sequence)

    for r in reversed(range(start, stop)):
        combinations = itertools.combinations(sequence, r)
        try:
            c = next(combinations)
            # print(r, len(c))
            power_set.append(c)
        except StopIteration:
            pass
        # power_set.append(combinations)

    return power_set


def evaluate_supergraph(Gs: Union[List[nx.DiGraph], nx.DiGraph], S: nx.DiGraph, progress_bar: bool = False, name: str = None):
    S, _ = as_topological_supergraph(S)
    Gs = [Gs] if isinstance(Gs, nx.DiGraph) else Gs

    # Chain the iterators together
    num_nodes = sum([G.number_of_nodes() for G in Gs])
    matched_nodes = 0
    desc = f"{name} | Matching nodes" if name is not None else "Matching nodes"
    pbar = tqdm.tqdm(total=num_nodes, desc=desc, disable=not progress_bar)

    # Get host generations
    generations_S = list(nx.topological_generations(S))

    Gs_monomorphisms = []
    Gs_units = []
    Gs_nodes = []
    for i_G, G in enumerate(Gs):
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
            for _i_gen_S, gen_S in enumerate(generations_S):
                if len(front) == 0:
                    break
                new_front = []
                for n_S in gen_S:
                    if len(front) == 0:
                        break
                    for n_G in front:
                        if _is_node_attr_match(n_G, n_S, G, S):
                            monomorphism[n_G] = (i_unit, n_S)
                            matched_nodes += 1
                            front.remove(n_G)
                            for n_G_child in G.successors(n_G):
                                matched_in_degree[n_G_child] += 1
                                if G.in_degree(n_G_child) == matched_in_degree[n_G_child]:
                                    new_front.append(n_G_child)
                            break
                front += new_front

            # Check if new nodes have been matched since last iteration
            new_size = len(monomorphism)
            new_matches = new_size - mono_size
            assert new_matches > 0, "No new nodes have been matched"
            mono_size = new_size

            # Increment unit counter
            i_unit += 1

            # Update progress bar
            if progress_bar:
                size = len(S)
                num_Gs = len(Gs)
                supergraph_nodes = size * (i_unit + sum(Gs_units))
                efficiency = matched_nodes / supergraph_nodes
                pbar.set_postfix_str(
                    f"{i_G+1}/{num_Gs} graphs, {matched_nodes}/{num_nodes} matched ({efficiency:.2%} efficiency, {size} nodes)"
                )
                pbar.update(new_matches)

        Gs_monomorphisms.append(monomorphism)
        Gs_units.append(i_unit)
        Gs_nodes.append(i_unit * len(S))
    assert matched_nodes == num_nodes, "Not all nodes in G have been matched"
    return Gs_monomorphisms


def match_supergraph_iter(
    G,
    S,
    front: Set = None,
    generations_S: List[List[Any]] = None,
    matched_in_degree: Dict[Any, int] = None,
    delta_matched_in_degree: Dict[Any, int] = None,
):
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
        for _i_gen_S, gen_S in enumerate(generations_S):
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


def _subgraph(G: nx.DiGraph, nodes):
    return G.subgraph(nodes)
    # g = nx.DiGraph()
    # for n in nodes:
    #     g.add_node(n, **G.nodes[n])
    # for (o, i, data) in G.edges(nodes, data=True):
    #     if o in nodes and i in nodes:
    #         g.add_edge(o, i, **data)
    # return g


def grow_supergraph_iter(
    G: nx.DiGraph, S: nx.DiGraph, leaf_kind, combination_mode: str = "linear", backtrack: int = 3, sort_fn: Callable = None
):
    assert combination_mode in ["linear", "power"], f"Invalid combination mode {combination_mode}"
    combinations_fn = generate_linear_set if combination_mode == "linear" else generate_power_set

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
    # generations_S = list(nx.topological_generations(S.subgraph(set(S.nodes()) - {leaf_S})))
    S_sub = _subgraph(S, set(S.nodes()) - {leaf_S})
    generations_S = list(nx.topological_generations(S_sub))

    # Use first generation of motif as the initial front (i.e. nodes with in_degree=0)
    next_front = {n for n in G if G.in_degree(n) == 0}

    # Prepare matched in_degree counter for motif
    matched_in_degree = {n: 0 for n in G}

    # Initialize empty monomorphism map
    G_monomorphism: Dict[Any, Tuple[int, Any]] = dict()

    # Initialize supergraph mapping
    S_init_to_S = {n: n for n in S}

    # Iterate over leafs
    i_backtrack = 0
    i_partition = 0
    G_unmatched = G.copy(as_view=False)
    state_history = deque(maxlen=backtrack + 1)
    while i_partition < num_partitions:
        assert len(next_front) > 0, "No more nodes in next_front but not all nodes in G have been matched"

        # Get current leaf node
        leaf_k = leafs_G[i_partition]

        # Store current search state
        delta_matched_in_degree = dict()
        delta_G_monomorphism = dict()
        state = dict(
            i_partition=i_partition,
            next_front=next_front.copy(),
            delta_matched_in_degree=delta_matched_in_degree,
            delta_G_monomorphism=delta_G_monomorphism,
        )
        state_history.append(state)

        # Match nodes in G to nodes in S
        front = next_front.copy()
        _, front, monomorphism = next(
            match_supergraph_iter(G, S, front, generations_S, matched_in_degree, delta_matched_in_degree)
        )

        # Determine nodes in G that are ancestors of current leaf node (i.e. that must be matched before leaf node)
        ancestors = nx.ancestors(G_unmatched, leaf_k)

        # Determine matched nodes in ancestorsSimplifications/approximations made in the algorithm
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
            i_monomorphism = {k: (i_partition, v) for k, v in monomorphism.items()}
            G_monomorphism.update(i_monomorphism)
            delta_G_monomorphism.update(monomorphism)

            # Remove largest_monomorphism nodes from G_unmatched
            G_unmatched.remove_nodes_from(monomorphism.keys())

            # Reduce i_backtrack
            i_backtrack = max(i_backtrack - 1, 0)

            # Only yield if we have matched a new partition
            if i_backtrack == 0:
                yield i_partition, G_unmatched, S, G_monomorphism, monomorphism, S_init_to_S

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
                combinations = combinations_fn(constrained_front, include_empty_set=False)
                for var_front in combinations:
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

                    # Mutate matched_in_degree for nodes excluded in combinations (i.e. nodes not in constrained_front but in var_front)
                    _delta_matched_in_degree = dict()  # Record difference in matched degree
                    for n_G in constrained_front.difference(var_front):
                        for n_G_child in G.successors(n_G):
                            matched_in_degree[n_G_child] += 1
                            _delta_matched_in_degree[n_G_child] = _delta_matched_in_degree.get(n_G_child, 0) + 1
                            if G.in_degree(n_G_child) == matched_in_degree[n_G_child]:
                                front.add(n_G_child)

                    # Match nodes in G to nodes in S
                    _, front, monomorphism = next(
                        match_supergraph_iter(G, S, front, generations_S, matched_in_degree, _delta_matched_in_degree)
                    )

                    # Reset matched_in_degree with _delta_matched_in_degree
                    for k, v in _delta_matched_in_degree.items():
                        matched_in_degree[k] -= v

                    # Determine matched nodes in ancestors
                    matched_nodes = ancestors.intersection(monomorphism.keys())
                    matched_size = len(matched_nodes)

                    if matched_size > largest_size:
                        largest_monomorphism = monomorphism
                        largest_size = matched_size

                # Mutate matched_in_degree for nodes excluded in combinations (i.e. nodes not in constrained_front but in var_front)
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
            # P = G.subgraph(nodes_P)
            P = _subgraph(G, nodes_P)
            assert check_order(S, P, largest_monomorphism)

            # Create mapping of nodes from S to P
            mcs = {node_S: node_P for node_P, node_S in largest_monomorphism.items()}

            # Extract subgraphs based on the mappings
            # mcs_S = S.subgraph(mcs.keys())
            # mcs_P = P.subgraph(mcs.values())
            mcs_S = _subgraph(S, mcs.keys())
            mcs_P = _subgraph(P, mcs.values())

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
            new_S, new_mono = as_topological_supergraph(P_unified_with_S, leaf_kind=leaf_kind, sort_fn=sort_fn)
            assert S.out_degree[list(leafs_S.keys())[0]] == 0, f"Leaf node of kind {leaf_kind} in S has out_degree > 0"

            # Update S and related variables
            S_to_new_S = {node_p: node_S for node_p, node_S in new_mono.items() if node_p in S.nodes}
            S_init_to_S = {node_S_init: new_mono[node_S] for node_S_init, node_S in S_init_to_S.items()}
            S = new_S
            num_nodes_S = len(S) - 1  # Excluding the leaf node
            # generations_S = list(nx.topological_generations(S.subgraph(set(S.nodes()) - {leaf_S})))
            S_sub = _subgraph(S, set(S.nodes()) - {leaf_S})
            generations_S = list(nx.topological_generations(S_sub))

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
            G_monomorphism = {node_G: (i, S_to_new_S[node_S]) for node_G, (i, node_S) in G_monomorphism.items()}

            # Reinitialize G_unmatched with nodes not already matched
            G_unmatched = G.copy()
            G_unmatched.remove_nodes_from(G_monomorphism.keys())

            # Now that we have a new S, we should backtrack the search.
            i_backtrack = len(state_history)  # Make sure we don't end up in an infinite loop.

            # Reset state history
            state_history = deque(maxlen=backtrack + 1)


def grow_supergraph(
    Gs: Union[List[nx.DiGraph], nx.DiGraph],
    leaf_kind,
    S_init: nx.DiGraph = None,
    combination_mode: str = "linear",
    backtrack: int = 3,
    sort_fn: Callable = None,
    progress_bar: bool = True,
    validate: bool = False,
    progress_fn: Callable = None,
):
    Gs = [Gs] if isinstance(Gs, nx.DiGraph) else Gs
    num_Gs = len(Gs)
    leafs_G = [{n: data for n, data in G.nodes(data=True) if data["kind"] == leaf_kind} for G in Gs]
    num_partitions = sum([len(leafs) for leafs in leafs_G])
    num_nodes = sum([len(G) for G in Gs])

    # Get
    S_init = S_init if S_init is not None else as_topological_supergraph(Gs[0], leaf_kind=leaf_kind, sort=[next(iter(leafs_G[0].keys()))])[0]
    S_init_size = len(S_init)
    S = as_topological_supergraph(S_init, leaf_kind=leaf_kind, sort_fn=sort_fn)[0]

    # Initialize progress bar
    pbar = tqdm.tqdm(total=num_partitions, desc="Growing supergraph", disable=not progress_bar)

    # Main loop to grow the supergraph
    t_elapsed = 0.0
    # S_history = [(0, S)]
    Gs_monomorphism = []
    Gs_S_init_to_S = []
    Gs_matched = []
    Gs_num_partitions = []
    G_monomorphism = {}
    for i_G, G in enumerate(Gs):
        grow_iter = grow_supergraph_iter(G, S, leaf_kind, combination_mode, backtrack, sort_fn)
        G_S_init_to_S = {n: n for n in S}
        t_start = time.perf_counter()
        for i_partition, _G_unmatched, S, _G_monomorphism, _monomorphism, _G_S_init_to_S in grow_iter:
            # Get time elapsed
            t_end = time.perf_counter()
            t_elapsed += t_end - t_start
            # if len(S) > len(S_history[-1][1]):
            #     S_history.append((i_G, S))
            #     if False:
            #         import matplotlib.pyplot as plt
            #         from supergraph.evaluate import plot_graph
            #         fig, axes = plt.subplots(nrows=len(S_history), sharey=True, sharex=True)
            #         fig.set_size_inches(12, 15)
            #         [plot_graph(axes[i], s) for i, (idx, s) in enumerate(S_history)]
            #         [axes[i].set_ylabel(f"graph {idx}") for i, (idx, s) in enumerate(S_history)]
            #         plt.show()
            G_S_init_to_S = _G_S_init_to_S
            G_monomorphism = _G_monomorphism
            if progress_fn:
                progress_fn(t_elapsed, Gs_num_partitions, Gs_matched, i_partition, G_monomorphism, G, S)
            if progress_bar:
                size = len(S)
                supergraph_nodes = size * (i_partition + 1 + sum(Gs_num_partitions))
                matched_nodes = len(G_monomorphism) + sum(Gs_matched)
                efficiency = matched_nodes / supergraph_nodes
                pbar.set_postfix_str(
                    f"{i_G+1}/{num_Gs} graphs, {matched_nodes}/{num_nodes} matched ({efficiency:.2%} efficiency, {size} nodes)"
                )
                pbar.update(1)
            t_start = time.perf_counter()

        # Save monomorphism
        Gs_num_partitions.append(i_partition + 1)
        Gs_matched.append(len(G_monomorphism))
        Gs_monomorphism.append(G_monomorphism)
        Gs_S_init_to_S.append(G_S_init_to_S)

    # Map intermediate S to final S
    Gs_S_to_S_final = []
    G_S_to_S_final = {n: n for n in S}
    for _, G_S_to_S in enumerate(reversed(Gs_S_init_to_S)):
        Gs_S_to_S_final.append(G_S_to_S_final)
        G_S_to_S_final = {n: G_S_to_S_final[v] for n, v in G_S_to_S.items()}
    S_init_to_S_final = G_S_to_S_final  # maps final S to S_init (the one provided as argument to this function)
    Gs_S_to_S_final = list(reversed(Gs_S_to_S_final))

    # Check that size of Gs_S_to_S_final is monotonically increasing
    assert all(
        [len(Gs_S_to_S_final[i]) <= len(Gs_S_to_S_final[i + 1]) for i in range(0, len(Gs_S_to_S_final) - 1)]
    ), "Size of Gs_S_to_S_final is not monotonically increasing"
    assert len(S_init_to_S_final) == S_init_size, "Size of S_init_to_S_final is not equal to size of S_init"

    # Map monomorphism to final S
    Gs_monomorphism_final = []
    for i_G, (G_S_to_S_final, G_monomorphism) in enumerate(zip(Gs_S_to_S_final, Gs_monomorphism)):
        G_monomorphism_final = {n: (i, G_S_to_S_final[v]) for n, (i, v) in G_monomorphism.items()}
        Gs_monomorphism_final.append(G_monomorphism_final)

        # Sort G_monomorphism_final by partition index
        if validate:
            partitions = {}
            for n, (i, v) in G_monomorphism_final.items():
                if i not in partitions:
                    partitions[i] = {}
                partitions[i][n] = v
            for _i, partition in partitions.items():
                # P = Gs[i_G].subgraph(partition.keys())
                P = _subgraph(Gs[i_G], partition.keys())
                is_mono = check_order(S, P, partition)
                assert is_mono, "Remapped monomorphism is not a valid with respect to the order of the supergraph"
                # if not is_mono:
                #     print(f"Monomorphism is not a valid order for graph {i_G} partition {i}")
                #     # raise ValueError("Monomorphism is not a valid order")
                #     host = S
                #     motif = P
                #     mapping = partition
                #     check_edges = [
                #         (
                #             motif_u,
                #             motif_v,
                #             host.nodes[mapping[motif_u]]["generation"] < host.nodes[mapping[motif_v]]["generation"],
                #         )
                #         for motif_u, motif_v in motif.edges
                #         if motif_u in mapping and motif_v in mapping
                #     ]
                #     check_nodes = all(
                #         [_is_node_attr_match(motif_n, host_n, motif, host) for motif_n, host_n in mapping.items()]
                #     )
                #     import matplotlib.pyplot as plt
                #     from supergraph.evaluate import plot_graph
                #
                #     fig, axes = plt.subplots(nrows=2)
                #     fig.set_size_inches(12, 15)
                #     plot_graph(axes[0], S)
                #     plot_graph(axes[1], P)
                #     plt.show()

    # Format supergraph
    S = as_compact_supergraph(Gs, S_init, S, Gs_monomorphism_final)
    return S, S_init_to_S_final, Gs_monomorphism_final


# def ancestral_partition(G, leaf_kind):
#     # Copy Graph
#     G = G.copy(as_view=False)
#
#     # Count number of nodes of each type
#     node_types: Dict[str, Dict[str, Dict]] = {}
#     for n, data in G.nodes(data=True):
#         if data["kind"] not in node_types:
#             node_types[data["kind"]] = dict()
#         node_types[data["kind"]][n] = data
#
#     # Sort nodes of each type by sequence number
#     for k, v in node_types.items():
#         node_types[k] = {k: v[k] for k in sorted(v.keys(), key=lambda k: v[k]["seq"])}
#
#     # Define ideal partition size
#     leafs = node_types[leaf_kind]
#     num_partitions = len(leafs)
#     size_ideal = {k: -(-len(v) // num_partitions) for k, v in node_types.items()}
#     assert (min([len(v) for v in node_types.values()]) > 0), "The minimum number of nodes of every type per partition must be > 0."
#
#     P = {}
#     G_partition = {}
#     for k, (leaf, data) in enumerate(leafs.items()):
#         ancestors = list(nx.ancestors(G, leaf))
#         ancestors_leaf = ancestors + [leaf]
#
#         # Get subgraph
#         g = nx.DiGraph()
#         g.add_node(leaf, **G.nodes[leaf])
#         for n in ancestors_leaf:
#             g.add_node(n, **G.nodes[n])
#         for (o, i, data) in G.edges(ancestors_leaf, data=True):
#             if o in ancestors_leaf and i in ancestors_leaf:
#                 g.add_edge(o, i, **data)
#         # Add generation information to nodes
#         generations = nx.topological_generations(g)
#         for i_gen, gen in enumerate(generations):
#             for n in gen:
#                 g.nodes[n]["generation"] = i_gen
#                 # g.nodes[n]["ancestors"] = list(nx.ancestors(g, n))
#                 # g.nodes[n]["descendants"] = list(nx.descendants(g, n))
#         P[k] = g
#
#         # Get subgraph
#         G_partition[k] = G.subgraph(ancestors_leaf).copy(as_view=False)
#
#         # Remove nodes
#         G.remove_nodes_from(ancestors_leaf)
#
#     # for k in P.keys():
#     #     assert P[k].edges() == G_partition[k].edges()
#     #     for n, data in P[k].nodes(data=True):
#     #         for key, val in data.items():
#     #             if key == "generation":
#     #                 continue
#     #             print(key, val==G_partition[k].nodes[n][key])
#
#     new_P = {}
#     for i, p in P.items():
#         s = size_ideal.copy()
#
#         # Prepare partitioning
#         matched_out_degree = dict(p.out_degree())
#         front = [n for n, od in matched_out_degree.items() if od == 0]
#         assert len(front) == 1, "There must be only one node with out_degree=0 (the partition's leaf node)."
#         ideal_partition = []
#         while len(front) > 0:
#             last_size = len(ideal_partition)
#
#             n_p = front.pop(0)
#             if s[p.nodes[n_p]["kind"]] > 0:
#
#                 # Add to ideal partition
#                 ideal_partition.append(n_p)
#                 s[p.nodes[n_p]["kind"]] -= 1  # Decrement size counter
#
#                 # increase matched out_degree by +1
#                 for n_p_parent in p.predecessors(n_p):
#                     matched_out_degree[n_p_parent] -= 1
#                     if matched_out_degree[n_p_parent] == 0:
#                         front.append(n_p_parent)
#
#             if len(ideal_partition) == last_size:
#                 break
#         matched_nodes = len(ideal_partition)
#         total_nodes = sum([num for k, num in size_ideal.items()])
#         # print(f"Partition {i}: {matched_nodes}/{total_nodes} nodes matched.")
#         new_P[i] = p.subgraph(ideal_partition).copy(as_view=False)
#
#     return new_P
