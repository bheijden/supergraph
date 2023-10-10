from math import floor
import numpy as np
import itertools
import tqdm
from typing import List, Dict, Tuple, Set, Any, Union
from collections import deque
import networkx as nx
import supergraph.open_colors as oc
import supergraph.evaluate as eval

Mapping = Dict[str, str]
Partitions = Dict[int, Dict[int, nx.DiGraph]]
Mappings = Dict[int, Dict[int, Mapping]]


# Layout nodes
attribute_set = {"kind", "edgecolor", "facecolor", "position", "alpha"}


# Define custom exception called ShortCircuit
class ShortCircuit(Exception):
    pass


def algorithm_1(
    supervisor: Any,
    backtrack: int,
    Gs: Union[List[nx.DiGraph], nx.DiGraph],
    max_topo: int = None,
    max_comb: int = None,
    progress_bar: bool = True,
) -> Tuple[nx.DiGraph, Mappings, Partitions]:
    # Convert Gs to list
    Gs = [Gs] if isinstance(Gs, nx.DiGraph) else Gs
    assert all([nx.is_directed_acyclic_graph(G) for G in Gs]), "Not all G in Gs are DAG."

    # Initialize supergraph
    S = nx.DiGraph()
    data_supervisor = {}  # Store data of supervisor vertex
    for u, data in Gs[0].nodes(data=True):
        if data["kind"] == supervisor:
            data_supervisor = data
            break
    S.add_node(f"s{supervisor}_0", **{k: v for k, v in data_supervisor.items() if k in attribute_set}, seq="S")

    # Store partitions P and mappings Ms for each episode
    Ps: Partitions = {}  # Partitions[i][j] of all episodes
    Ms: Mappings = {}  # Mappings[i][j] of all episodes

    # Initialize progress bar
    num_nodes = sum([G.number_of_nodes() for G in Gs])
    num_graphs = len(Gs)
    matched_nodes = 0
    num_partitions = sum([len([u for u in G.nodes if G.nodes[u]["kind"] == supervisor]) for G in Gs])
    pbar = tqdm.tqdm(total=num_partitions, desc="Growing supergraph", disable=not progress_bar)

    # Iterate over all episode computation graphs
    for i, G in enumerate(Gs):
        # Store partitions and mappings of current episode
        Ps[i]: Dict[int, nx.DiGraph] = {}
        Ms[i]: Dict[int, Dict[str, str]] = {}

        # Initialize unmatched graph.
        G_u = G.copy(as_view=False)

        # Track unmatched supervisor vertices with V_lsup (sorted set of supervisor vertices)
        tau = list(nx.topological_sort(G_u))
        V_lsup = deque([u for u in tau if G_u.nodes[u]["kind"] == supervisor])
        V_lsup_matched = deque()  # Used to restore backtrack

        # Define index function for supervisor vertex w.r.t. complete graph G (instead of G_u)
        V_lsup_all = list([u for u in tau if G.nodes[u]["kind"] == supervisor])
        I = lambda u: V_lsup_all.index(u)

        # Until all supervisor vertices are matched
        while len(V_lsup) > 0:
            # Get next unmatched supervisor vertex from sorted set of supervisor vertices
            u_ij = V_lsup[0]
            j = I(u_ij)

            # Get all ancestors of u_ij
            ancestors = set(nx.ancestors(G_u, u_ij))
            A_u = ancestors.union({u_ij})

            # Get largest mapping M*
            Mstar = algorithm_2(supervisor, S, G_u, A_u, max_topo=max_topo, max_comb=max_comb)
            Pstar = G.subgraph(Mstar.keys())  # Induce partition P* from M*
            Pc = G.subgraph(ancestors.difference(set(Pstar.nodes())))  # Induce missing subgraph

            if len(Pc.nodes()) == 0:  # All ancestors of u_ij are matched
                Ms[i][j] = Mstar  # Store mapping
                Ps[i][j] = Pstar  # Store partition
                G_u.remove_nodes_from(Pstar.nodes())  # Remove matched vertices from unmatched graph
                V_lsup_matched.append(V_lsup.popleft())  # Remove matched supervisor vertex from V_lsup

                # Update progressbar
                matched_nodes += len(Mstar)
                matched_partitions = sum([len(M) for M in Ms.values()])
                size = len(S.nodes())
                efficiency = matched_nodes / (matched_partitions * size)
                pbar.n = matched_partitions
                pbar.set_postfix_str(
                    f"{i+1}/{num_graphs} graphs, {matched_nodes}/{num_nodes} matched ({efficiency:.2%} efficiency, {size} nodes)"
                )
                pbar.refresh()
            else:  # Partial match, i.e. some ancestors of u_ij are unmatched
                V_u = set(G_u.nodes())
                for b in reversed(range(max(j - backtrack, 0), max(j, 0))):
                    # Restore previously matched partitions in unmatched graph (ie backtrack)
                    V_u = V_u.union(Ps[i][b].nodes())
                    V_lsup.appendleft(V_lsup_matched.pop())  # Restore supervisor vertices
                    # Update progressbar
                    matched_nodes -= len(Ms[i][b])
                    # Remove stored partitions and mappings
                    del Ps[i][b], Ms[i][b]

                # Reinstate unmatched graph G_u
                G_u = G.copy()
                G_u.remove_nodes_from(set(G_u.nodes()).difference(V_u))

            # Update supergraph S to S' by:
            # - adding missing ancestor, i.e., Pc.nodes().
            # - adding missing edges that make the (partial) mapping a valid subgraph monomorphism.
            Sprime = S  # Initialize supergraph S' as copy of S
            Mc: Mapping = {}  # Vertices are added with re-mapped names according to supergraph convention.
            next_idx = {}  # Naming convention for supergraph vertices, i.e., s{kind}_{next_idx[kind]}.
            if len(Pc.nodes()) > 0:  # Add missing ancestor, i.e., Pc.nodes(), to S
                for u, data in S.nodes(data=True):  # Determine next index for each kind of vertex
                    next_idx[data["kind"]] = next_idx.get(data["kind"], 0) + 1
                for u, data in Pc.nodes(data=True):  # Add missing ancestor, i.e., Pc.nodes(), to S
                    next_idx[data["kind"]] = next_idx.get(data["kind"], 0)
                    name = f"s{data['kind']}_{next_idx[data['kind']]}"  # Supergraph vertex naming convention
                    Sprime.add_node(name, **{k: v for k, v in data.items() if k in attribute_set})  # Add node
                    Mc[u] = name  # Re-map name according to supergraph convention
                    next_idx[data["kind"]] += 1

            # Add missing edges that make the (partial) mapping a valid subgraph monomorphism.
            PstarPc = G.subgraph(set(Pstar.nodes()).union(set(Pc.nodes())))
            MstarMc = {**Mstar, **Mc}
            for u, v in PstarPc.edges():
                Sprime.add_edge(MstarMc[u], MstarMc[v], **eval.edge_data)  # Add edges to supergraph S' (with re-mapped names)
            assert nx.is_directed_acyclic_graph(Sprime), "The updated S is not DAG."
            S = Sprime  # Update supergraph S

            # Update positions of nodes in S (purely for visualization purposes)
            if len(Pc.nodes()) > 0:
                S = format_supergraph(S)

    # Format supergraph S
    S = format_supergraph(S)

    # Validate that Ms are monomorphisms
    for i in Ps.keys():
        for j in Ps[i].keys():
            assert is_subgraph_monomorphism(
                S, Ps[i][j], Ms[i][j]
            ), f"Ms[{i}][{j}] is not a subgraph monomorphism of S and Ps[{i}][{j}]."
    return S, Ms, Ps


def algorithm_2(
    supervisor: Any, S: nx.DiGraph, G_u: nx.DiGraph, A_u: Set[str], max_topo: int = None, max_comb: int = None
) -> Mapping:
    # Initialize empty mapping
    Mstar: Mapping = {}
    dom_Mstar = set()

    # Initialize search graph
    Gexcl = G_u.copy(as_view=False)

    # Define iterator that gets all topological sorts of S with supervisor vertices last
    S_no_lsup = S.subgraph([u for u in S.nodes() if S.nodes[u]["kind"] != supervisor])
    assert (
        len(S_no_lsup.nodes()) == len(S.nodes()) - 1
    ), "S_no_lsup should have one less node than S, namely a single instance of the supervisor vertex"
    assert S.out_degree(f"s{supervisor}_0") == 0, "The supervisor vertex should have no outgoing edges"

    def all_topological_sorts_with_lsup_last():
        for tau in nx.all_topological_sorts(S_no_lsup):
            yield list(tau) + [f"s{supervisor}_0"]

    try:  # Try except block to catch ShortCircuit exception
        iter = 0
        while True:
            # Initialize search front as roots of Gexcl
            Fexcl = set([n for n, d in Gexcl.in_degree() if d == 0])  # todo: determine in smarter way?

            # Determine constrained search front.
            # That is, the intersection of the search front with the ancestors A_u that must be matched.
            Fcon = Fexcl.intersection(A_u)

            # Iterate over all k-sized combinations
            assert len(Fcon) > 0, "The constrained search front should not be empty"
            for k in range(0, len(Fcon)):
                num_comb = 0  # Track number of combinations per size k
                for Fcomb in itertools.combinations(Fcon, len(Fcon) - k):
                    num_comb += 1
                    if max_comb is not None and num_comb > max_comb:
                        break  # if a maximum number of combinations per k is set, break if exceeded

                    # Iterate over all topological sorts of S
                    num_topo = 0  # Track number of topological sorts per combination
                    for tau in all_topological_sorts_with_lsup_last():
                        num_topo += 1
                        if max_topo is not None and num_topo > max_topo:
                            break  # if a maximum number of topological sorts per combination is set, break if exceeded

                        # Search for a valid mapping
                        iter += 1
                        Gc = Gexcl.copy(as_view=False)  # Initialize candidate graph
                        Gc.remove_nodes_from(Fcon.difference(Fcomb))  # Remove nodes not in combination
                        Fc = Fexcl.difference(Fcon.difference(Fcomb))  # Remove nodes not in combination
                        Mc: Mapping = {}  # Initialize empty candidate mapping

                        for v in tau:
                            for u in Fc:
                                if S.nodes[v]["kind"] == Gc.nodes[u]["kind"]:
                                    Mc[u] = v  # Extend mapping
                                    successors = Gc.successors(u)
                                    Gc.remove_node(u)  # Remove node from candidate graph
                                    # Add new roots of Gc (due to removal of u) to search front
                                    Fc.remove(u)  # Remove node from search front
                                    [Fc.add(s) for s in successors if Gc.in_degree(s) == 0]
                                    break

                        # Store mapping if largest
                        dom_Mc = set(Mc.keys())
                        if dom_Mc.intersection(A_u) > dom_Mstar.intersection(A_u):
                            Mstar = Mc
                            dom_Mstar = dom_Mc

                        # Determine if short circuit is possible
                        s_max = len(A_u) - len(A_u.difference(Gexcl.nodes())) - (len(Fcon) - len(Fcomb))
                        if len(dom_Mstar.intersection(A_u)) >= s_max or len(dom_Mstar) == len(S):
                            raise ShortCircuit  # No larger mapping can be found, so we can short circuit

            # Remove vertices from Gexcl that are not in the search front
            Gexcl.remove_nodes_from(Fcon)
    except ShortCircuit:
        pass
    return Mstar


def is_subgraph_monomorphism(S: nx.DiGraph, P: nx.DiGraph, M: Mapping) -> bool:
    check_mapping = all([u in M for u in P.nodes])
    check_edges = all([S.has_edge(M[u], M[v]) for u, v in P.edges if u in M and v in M])
    check_nodes = all([P.nodes[u_P]["kind"] == S.nodes[u_S]["kind"] for u_P, u_S in M.items()])
    return check_mapping and check_nodes and check_edges


def format_supergraph(S: nx.DiGraph) -> nx.DiGraph:
    generations = nx.topological_generations(S)
    for idx_gen, gen in enumerate(generations):
        positions = [round(S.nodes[u]["position"][1]) for u in gen]
        try:
            positions = generate_coordinates(positions)
        except BaseException:
            print("Error in generating coordinates. Probably, because the 'kind' is not a number.")
            raise
        for idx_layer, u in enumerate(gen):
            p = positions[round(S.nodes[u]["position"][1])].pop(0)
            S.nodes[u]["position"] = (idx_gen, p)
    return S


def generate_coordinates(input_list):
    count_dict = count_elements(input_list)
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


def count_elements(input_list):
    count_dict = {}
    for elem in input_list:
        if elem in count_dict:
            count_dict[elem] += 1
        else:
            count_dict[elem] = 1
    return count_dict


def get_example_graphs():
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
    for (i, o) in tqdm.tqdm(edges, desc="Generate graphs"):
        for idx, id_seq in enumerate(node_kinds[i]):
            data_source = G0.nodes[id_seq]
            # Add stateful edge
            if idx > 0:
                data = {"delay": 0.0, "pruned": False}
                data.update(**eval.edge_data)
                G0.add_edge(node_kinds[i][idx - 1], id_seq, **data)
            for id_tar in node_kinds[o]:
                data_target = G0.nodes[id_tar]
                if data_target["ts"] >= data_source["ts"]:
                    data = {"delay": 0.0, "pruned": False}
                    data.update(**eval.edge_data)
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
                    delayed_data.update(**eval.delayed_edge_data)
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
