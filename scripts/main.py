import time
from functools import partial, lru_cache
from typing import List, Dict, Tuple, Union, Callable, Set, Any
import networkx as nx
import matplotlib.pyplot as plt

# from networkx.algorithms import isomorphism
# import grandiso
# import rex.mcs as rex_mcs
import supergraph as sg
# import supergraph.mono as sg_mono

# Define graph isomorphism function
_is_node_attr_match = lambda motif_node_id, host_node_id, motif, host: host.nodes[host_node_id]["kind"] == motif.nodes[motif_node_id]["kind"]
_is_node_attr_match = lru_cache()(_is_node_attr_match)
_is_edge_attr_match = lambda motif_edge_id, host_edge_id, motif, host: True
_is_edge_attr_match = lru_cache()(_is_edge_attr_match)
# _L_is_edge_attr_match = lambda motif_edge_id, host_edge_id, motif, host: (host.nodes[host_edge_id[0]]["generation"] - host.nodes[host_edge_id[1]]["generation"]) >= (motif.nodes[motif_edge_id[0]]["generation"] - motif.nodes[motif_edge_id[1]]["generation"])
# _L_is_edge_attr_match = lambda motif_edge_id, host_edge_id, motif, host: host.nodes[host_edge_id[0]]["generation"] < \
#                                                                          host.nodes[host_edge_id[1]]["generation"]
# _L_is_edge_attr_match = lru_cache()(_L_is_edge_attr_match)
# _L_is_node_structural_match = lambda motif_node_id, host_node_id, motif, host: True
# _L_is_node_structural_match = lru_cache()(_L_is_node_structural_match)
_is_node_structural_match = lambda motif_node_id, host_node_id, motif, host: host.in_degree(host_node_id) >= motif.in_degree(
    motif_node_id) and host.out_degree(host_node_id) >= motif.out_degree(motif_node_id)
_is_node_structural_match = lru_cache()(_is_node_structural_match)


if __name__ == "__main__":
    # todo: Perform transitive reduction on P before matching? (i.e. remove redundant edges)
    # todo: Check both in and out degree in _is_node_structural_match for finding exact monomorphism.
    # todo: Remove degree constraint in _is_node_structural_match for finding largest monomorphism.
    # todo: Determine starting pattern
    #  - Determine all valid edges between node types (E_val) = S
    #  - Select partition P that has the most edges (i.e. edges are interpreted as constraints).
    #  - Grab topological sort of P and add all feasible edges (TC(topo(P)) | e in E_val) = S
    #  - [OPTIONAL] Repeat for all topological sorts of P and pick the one with the most feasible edges.
    #  - Sort partitions by the number of edges.
    #  - Iterate over partitions and find largest (subgraph) monomorphism with S.
    #  - If perfect match, store mapping (P -> S).
    #  - If no subgraph monomorphism, then merge S with with non-matched nodes in partition.
    #       - Grab topological sort of merged(S, P) and add all feasible edges (TC(topo(P)) | e in E_feas) = new_S
    #       - Remap mappings from S to new_S.
    #       - Add mapping (P -> new_S).
    #  - Repeat until all partitions are matched.

    # run_excalidraw_example()

    # todo: use Transitive Reduction to remove redundant edges in P.
    # todo: multi-start search with multi-processing?
    # todo: make _largest search deterministic

    # Function inputs
    MUST_PLOT = False
    SEED = 24  # 22
    THETA = 0.07
    SIGMA = 0.3
    WINDOW = 0
    NUM_NODES = 8  # 30
    T = 100

    ROOT_KIND = None
    LEAF_KIND = 0
    AS_TC = True
    NUM_TOPO = 1
    MUST_SORT = False
    DESCENDING = True
    MAX_EVALS = 100000

    # Define graph
    fs = [1, 2, 3, 4, 5, 10, 20, 20, 20, 40, 200]
    # fs = [float(i) for i in range(1, NUM_NODES + 1)] + [200]
    # fs = [float(i) for i in range(1, NUM_NODES + 1)]
    # fs = [2.**i for i in range(0, NUM_NODES)]
    NUM_NODES = len(fs)
    edges = {(i, (i + 1) % NUM_NODES) for i in range(NUM_NODES-1)}  # Add forward edges
    edges.update({(j, i) for i, j in edges})  # Add reverse edges
    # edges.update({(NUM_NODES-1, 0) for i, j in edges})  # Add reverse edges
    edges.update({(i, i) for i in range(NUM_NODES)})  # Stateful edges

    # Create graph
    G = sg.create_graph(fs, edges, T, seed=SEED, theta=THETA, sigma=SIGMA)
    G = sg.prune_by_window(G, WINDOW)

    # Get initial supergraph
    _E_val = sg.get_set_of_feasible_edges(G)
    S_init, monomorphism = sg.sort_to_S(G, [f"{LEAF_KIND}_0"], _E_val)

    # Supergraph
    S_rec, *_ = sg.grow_supergraph(G, S_init, LEAF_KIND, _E_val)

    # Partition
    P = sg.balanced_partition(G, root_kind=ROOT_KIND)

    # Record original mapping
    partition_order = list(P.keys())

    # Determine all feasible edges between node types (E_val) = S
    E_val = sg.get_set_of_feasible_edges(P)

    # Select partition P that has the most edges (i.e. edges are interpreted as constraints).
    key_S, P_S = sorted(P.items(), key=lambda item: item[1].number_of_edges(), reverse=True)[0]

    # Determine starting pattern. monomorphism = {P_node: S_0_node}
    S_init, monomorphism = sg.as_S(P_S, E_val, num_topo=NUM_TOPO, as_tc=AS_TC)
    S = S_init

    # Define linear supergraph (benchmark)
    S_lin, monomorphism_lin = next(sg.linear_S_iter(G, E_val))

    # Iterate over partitions sorted by the number of edges in descending order.
    P_sorted = {k: P for k, P in sorted(P.items(), key=lambda item: item[1].number_of_edges(), reverse=DESCENDING)} if MUST_SORT else P

    if MUST_PLOT:
        fig, axes = plt.subplots(nrows=2)
        fig.set_size_inches(12, 15)
        sg.plot_graph(axes[0], S)
        plt.show()

    # Find S
    metric = {"match": 0, "no_match": 0}
    S_history: Dict[nx.DiGraph, int] = {S: 0}
    S_to_S: Dict[nx.DiGraph, Dict[nx.DiGraph, Dict[str, str]]] = {}
    P_to_S: Dict[int, Dict[nx.DiGraph, Dict[str, str]]] = {}
    for k, P in P_sorted.items():
        # Find largest monomorphism
        start = time.time()
        num_evals, largest_mono = sg.find_largest_monomorphism(S, P)
        t_find_motifs = time.time() - start

        # Check if perfect match
        perfect_match = len(largest_mono) == len(P)

        # If no match, then merge S with non-matched nodes in partition.
        if perfect_match:
            P_monomorphism = largest_mono
            P_to_S[k] = {S: P_monomorphism}
            metric["match"] += 1
            print(f"k={k} | Match    | find={t_find_motifs:.3f} sec | nodes=({S.number_of_nodes()}/{P.number_of_nodes()}) | edges=({S.number_of_edges()}/{P.number_of_edges()})")
        else:
            # Prepare to unify S and P
            mcs = {node_S: node_p for node_p, node_S in largest_mono.items()}
            mcs_S = S.subgraph(mcs.keys())
            mcs_P = nx.relabel_nodes(S.subgraph(mcs.keys()), mcs, copy=True)
            E1 = sg.emb(mcs_S, S)
            E2 = sg.emb(mcs_P, P)
            tmp = nx.relabel_nodes(sg.unify(mcs_P, P, E2), largest_mono, copy=True)
            new_P_S = sg.unify(tmp, S, E1)

            # Verify that new_P_S is a DAG
            assert nx.is_directed_acyclic_graph(new_P_S), "Should be a DAG."

            # Determine S
            new_S, monomorphism = sg.as_S(new_P_S, E_val, num_topo=NUM_TOPO, as_tc=AS_TC)

            # Determine P_monomorphism
            P_monomorphism = {n: None for n in P.nodes}
            for node_p, _ in P_monomorphism.items():
                if node_p in largest_mono:
                    P_monomorphism[node_p] = monomorphism[largest_mono[node_p]]
                elif node_p in monomorphism:
                    P_monomorphism[node_p] = monomorphism[node_p]

            # Verify that P_monomorphism is a valid mapping
            assert sg.check_monomorphism(new_S, P, P_monomorphism), "P_monomorphism is not a valid mapping."
            P_to_S[k] = {new_S: P_monomorphism}

            # Filter out added nodes from S_monomorphism
            S_monomorphism = {node_p: node_S for node_p, node_S in monomorphism.items() if node_p in S.nodes}

            # Remap mappings from S to new_S.
            idx_S = len(S_history)  # NOTE! not thread safe
            S_history[new_S] = idx_S
            S_to_S[S] = {new_S: S_monomorphism}

            # Update S
            S = new_S
            metric["no_match"] += 1
            unmatched_nodes = len(P.nodes()) - len(largest_mono)
            print(f"k={k} | No Match | find={t_find_motifs:.3f} sec | nodes=({S.number_of_nodes()}/{P.number_of_nodes()}) | edges=({S.number_of_edges()}/{P.number_of_edges()}) | num_evals={num_evals} | unmatched_nodes={unmatched_nodes}")
    print(f"Matched {metric['match']} | No Match {metric['no_match']}")

    # Evaluate supergraph
    units_lin, pred_lin, m_lin = sg.evaluate_supergraph(G, S_lin)
    print(f"S_lin  | Number of nodes: {pred_lin} | number of units: {units_lin}")
    units_opt, pred_opt, m_opt = sg.evaluate_supergraph(G, S)
    print(f"S_opt | Number of nodes: {pred_opt} | number of units: {units_opt}/{len(P_sorted)}")
    units_rec, pred_rec, m_rec = sg.evaluate_supergraph(G, S_rec)
    leafs_G = {n: data for n, data in G.nodes(data=True) if data["kind"] == LEAF_KIND}
    print(f"S_rec | Number of nodes: {pred_rec} | number of units: {units_rec}/{len(leafs_G)}")
    units_init, pred_init, m_init = sg.evaluate_supergraph(G, S_init)
    print(f"S_init | Number of nodes: {pred_init} | number of units: {units_init}/{len(P_sorted)}")

    exit()

    # Find MCS
    # metric = {"match": 0, "no_match": 0}
    # MCS_history: Dict[nx.DiGraph, int] = {MCS: 0}
    # MCS_to_MCS: Dict[nx.DiGraph, Dict[nx.DiGraph, Dict[str, str]]] = {}
    # P_to_MCS: Dict[int, Dict[nx.DiGraph, Dict[str, str]]] = {}
    # for k, P in P_sorted.items():
    #
    #     # Find subgraph monomorphism using supergraph
    #     v2_start = time.time()
    #     v2_mono = sg.find_monomorphism(MCS, P)
    #     t_find_monomorphism = time.time() - v2_start
    #
    #     # Find subgraph monomorphism using grandiso
    #     # gr_start = time.time()
    #     # mono = sg_mono.find_motifs(P, MCS, limit=1)
    #     # t_find_motifs = time.time() - gr_start
    #     mono = v2_mono
    #     t_find_motifs = 0.
    #
    #     assert len(v2_mono) == len(mono), f"v2_mono={v2_mono} | mono={mono}"
    #
    #     if len(mono) > 0:  # Perfect match found! Save mapping: P -> subgraph(MCS)
    #         P_monomorphism = mono[0]
    #         P_to_MCS[k] = {MCS: P_monomorphism}
    #         metric["match"] += 1
    #         print(f"k={k} | Match    | find={t_find_motifs:.3f} sec | v2={t_find_monomorphism:.3f} sec | nodes=({MCS.number_of_nodes()}/{P.number_of_nodes()}) | edges=({MCS.number_of_edges()}/{P.number_of_edges()})")
    #     else:  # No match. Merge MCS with non-matched nodes in partition.
    #         rex_start = time.time()
    #         num_evals_v2, is_match_v2, largest_mono_lst_v2 = sg.find_largest_monomorphism(MCS, P)
    #         # num_evals, is_match, largest_mono_lst_old = sg_mono.find_largest_motifs(P, MCS, max_evals=MAX_EVALS)
    #         largest_mono_lst_old = largest_mono_lst_v2
    #         is_match = is_match_v2
    #         num_evals = num_evals_v2
    #         largest_mono_lst = largest_mono_lst_v2
    #         largest_mono = largest_mono_lst[0]
    #         # largest_mono = largest_mono_lst_old[0]
    #         t_find_large_motifs = time.time() - rex_start
    #
    #         assert not is_match, "Should not have found a match."
    #         diff = len(largest_mono_lst[0]) - len(largest_mono_lst_old[0])
    #         unmatched_nodes = len(P.nodes()) - len(largest_mono)
    #         print(f"difference={diff}")
    #
    #         if diff < 0:
    #             MUST_PLOT = True
    #
    #         # Prepare to unify MCS and P
    #         mcs = {node_MCS: node_p for node_p, node_MCS in largest_mono.items()}
    #         mcs_MCS = MCS.subgraph(mcs.keys())
    #         mcs_P = nx.relabel_nodes(MCS.subgraph(mcs.keys()), mcs, copy=True)
    #         E1 = sg.emb(mcs_MCS, MCS)
    #         E2 = sg.emb(mcs_P, P)
    #         tmp = nx.relabel_nodes(sg.unify(mcs_P, P, E2), largest_mono, copy=True)
    #         new_P_MCS = sg.unify(tmp, MCS, E1)
    #
    #         if True: #not nx.is_directed_acyclic_graph(new_P_MCS):
    #             _new_P_MCS = new_P_MCS.copy()
    #             _MCS = MCS.copy()
    #             _P = P.copy()
    #
    #             # Color nodes
    #             for _k, v in largest_mono.items():
    #                 _new_P_MCS.nodes[v].update({"edgecolor": "green"})
    #                 _MCS.nodes[v].update({"edgecolor": "green"})
    #                 _P.nodes[_k].update({"edgecolor": "green"})
    #             # Color edges
    #             for u, v in P.edges:
    #                 if u in largest_mono and v in largest_mono:
    #                     _MCS.edges[(largest_mono[u], largest_mono[v])].update({"color": "green"})
    #                     _new_P_MCS.edges[(largest_mono[u], largest_mono[v])].update({"color": "green"})
    #                     _P.edges[(u, v)].update({"color": "green"})
    #
    #             # Find cycles
    #             try:
    #                 cycle = list(nx.find_cycle(_new_P_MCS))
    #                 print(cycle)
    #                 num_evals, is_match, largest_mono_lst = sg_mono.find_largest_motifs(_P, _MCS, max_evals=MAX_EVALS)
    #                 for u, v in cycle:
    #                     if _new_P_MCS[u][v]["color"] == "green":
    #                         continue
    #                     _new_P_MCS[u][v]["color"] = "red"
    #             except nx.exception.NetworkXNoCycle:
    #                 pass
    #
    #             # Set edge alpha
    #             for u, v in _new_P_MCS.edges:
    #                 if _new_P_MCS[u][v]["color"] in ["green", "red"]:
    #                     _new_P_MCS[u][v]["alpha"] = 1.0
    #                 else:
    #                     _new_P_MCS[u][v]["alpha"] = 0.3
    #             for u, v in _MCS.edges:
    #                 if _MCS[u][v]["color"] in ["green", "red"]:
    #                     _MCS[u][v]["alpha"] = 1.0
    #                 else:
    #                     _MCS[u][v]["alpha"] = 0.3
    #             for u, v in _P.edges:
    #                 if _P[u][v]["color"] in ["green", "red"]:
    #                     _P[u][v]["alpha"] = 1.0
    #                 else:
    #                     _P[u][v]["alpha"] = 0.5
    #
    #             if MUST_PLOT or not nx.is_directed_acyclic_graph(new_P_MCS):
    #                 fig, axes = plt.subplots(nrows=4)
    #                 fig.set_size_inches(12, 15)
    #                 sg.plot_graph(axes[0], _P)
    #                 sg.plot_graph(axes[1], _MCS)
    #                 # _new_P_MCS.nodes["2_158"]["position"] = (5, 3)
    #                 sg.plot_graph(axes[2], _new_P_MCS)
    #                 plt.show()
    #
    #         assert nx.is_directed_acyclic_graph(new_P_MCS), "Should be a DAG."
    #
    #         new_MCS, monomorphism = sg.as_MCS(new_P_MCS, E_val, num_topo=NUM_TOPO, as_tc=AS_TC)
    #
    #         # Determine P_monomorphism
    #         P_monomorphism = {n: None for n in P.nodes}
    #         for node_p, _ in P_monomorphism.items():
    #             if node_p in largest_mono:
    #                 P_monomorphism[node_p] = monomorphism[largest_mono[node_p]]
    #             elif node_p in monomorphism:
    #                 P_monomorphism[node_p] = monomorphism[node_p]
    #
    #         # Verify that P_monomorphism is a valid mapping
    #         # They hint cannot already be a perfect match --> limitation of find_motifs.
    #         hints = [{node_p: node_MCS for idx, (node_p, node_MCS) in enumerate(P_monomorphism.items()) if idx > 0}]
    #         # mono = grandiso.find_motifs(P, new_MCS, limit=1, directed=True, is_node_attr_match=_is_node_attr_match,
    #         #                             is_edge_attr_match=_is_edge_attr_match, hints=hints)
    #         # assert len(mono) > 0, "P_monomorphism is not a valid mapping."
    #         P_to_MCS[k] = {new_MCS: P_monomorphism}
    #
    #         # Filter out added nodes from MCS_monomorphism
    #         MCS_monomorphism = {node_p: node_MCS for node_p, node_MCS in monomorphism.items() if k in MCS.nodes}
    #
    #         # Remap mappings from MCS to new_MCS.
    #         idx_MCS = len(MCS_history)  # NOTE! not thread safe
    #         MCS_history[new_MCS] = idx_MCS
    #         MCS_to_MCS[MCS] = {new_MCS: MCS_monomorphism}
    #
    #         # Update MCS
    #         MCS = new_MCS
    #         metric["no_match"] += 1
    #         print(f"k={k} | No Match | find={t_find_motifs:.3f} sec | v2={t_find_monomorphism:.3f} sec | nodes=({MCS.number_of_nodes()}/{P.number_of_nodes()}) | edges=({MCS.number_of_edges()}/{P.number_of_edges()}) | large={t_find_large_motifs:.3f} sec | num_evals={num_evals} | unmatched_nodes={unmatched_nodes}")
    # print(f"Matched {metric['match']} | No Match {metric['no_match']}")

    # # Create new plot
    # fig, axes = plt.subplots(nrows=3)
    # fig.set_size_inches(12, 15)
    # plot_graph(axes[0], G)
    #
    # for idx, G_partition in enumerate([G_partition]):
    #     for k, p in G_partition.items():
    #         # Create new plot
    #         print(f"Partition {k}")
    #         plot_graph(axes[idx+1], p)
    #         # tc = nx.transitive_closure(p)
    #         # plot_graph(axes[idx+3], tc)
    #
    # plt.show()


