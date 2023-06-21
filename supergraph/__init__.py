__version__ = "0.0.1"

import itertools
import tqdm
from typing import List, Dict, Tuple, Set, Any
from collections import deque
import networkx as nx


def _is_node_attr_match(motif_node_id: str, host_node_id: str, motif: nx.DiGraph, host: nx.DiGraph):
    return host.nodes[host_node_id]["kind"] == motif.nodes[motif_node_id]["kind"]


def sort_to_S(P: nx.DiGraph, sort, E_val, as_tc: bool = False) -> Tuple[nx.DiGraph, Dict[Any, Any]]:
    attribute_set = {"kind", "order", "edgecolor", "facecolor", "position", "alpha"}
    kinds = {P.nodes[n]["kind"]: data for n, data in P.nodes(data=True)}
    kinds = {k: {a: d for a, d in data.items() if a in attribute_set} for k, data in kinds.items()}
    edge_attribute_set = {"color", "linestyle", "alpha"}
    edge_data = {a: d for a, d in next(iter(P.edges(data=True)))[-1].items() if a in edge_attribute_set}
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


def generate_power_set(sequence, include_empty_set: bool = True, include_full_set: bool = True):
    power_set = []
    start = 0 if include_empty_set else 1
    stop = len(sequence) + 1 if include_full_set else len(sequence)

    for r in reversed(range(start, stop)):
        combinations = itertools.combinations(sequence, r)
        power_set.extend(combinations)

    return power_set


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
        for _i_gen_S, gen_S in enumerate(generations_S):
            if len(front) == 0:
                break
            new_front = []
            for n_S in gen_S:
                if len(front) == 0:
                    break
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
    return i_unit, i_unit * len(S), monomorphism
    # raise NotImplementedError("TODO: Implement supergraph evaluation")
    # assert check_monomorphism(S, G, monomorphism)


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
            topo_sort = list(
                nx.topological_sort(P_unified_with_S.subgraph([n for n in P_unified_with_S.nodes() if n != leaf_S]))
            )
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
            state_history = deque(maxlen=backtrack + 1)


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
    for i_partition, _G_unmatched, S, G_monomorphism, _monomorphism, _S_init_to_S in pbar:
        S_init_to_S = _S_init_to_S
        if progress_bar:
            size = len(S)
            supergraph_nodes = size * (i_partition + 1)
            matched_nodes = len(G_monomorphism)
            efficiency = matched_nodes / supergraph_nodes
            pbar.set_postfix_str(f"matched {matched_nodes}/{num_nodes} ({efficiency:.2%} efficiency, {size} size)")

    return S, S_init_to_S, G_monomorphism
