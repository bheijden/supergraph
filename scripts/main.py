import time
from functools import partial, lru_cache
from typing import List, Dict, Tuple, Union, Callable, Set, Any
import networkx as nx
from networkx.algorithms import isomorphism
import matplotlib.pyplot as plt
import rex.mcs as rex_mcs
import grandiso
import supergraph as sg


if __name__ == "__main__":
    # todo: Perform transitive reduction on P before matching? (i.e. remove redundant edges)
    # todo: Check both in and out degree in _is_node_structural_match for finding exact monomorphism.
    # todo: Remove degree constraint in _is_node_structural_match for finding largest monomorphism.
    # todo: Determine starting pattern
    #  - Determine all valid edges between node types (E_val) = MCS
    #  - Select partition P that has the most edges (i.e. edges are interpreted as constraints).
    #  - Grab topological sort of P and add all feasible edges (TC(topo(P)) | e in E_val) = MCS
    #  - [OPTIONAL] Repeat for all topological sorts of P and pick the one with the most feasible edges.
    #  - Sort partitions by the number of edges.
    #  - Iterate over partitions and find largest (subgraph) monomorphism with MCS.
    #  - If perfect match, store mapping (P -> MCS).
    #  - If no subgraph monomorphism, then merge MCS with with non-matched nodes in partition.
    #       - Grab topological sort of merged(MCS, P) and add all feasible edges (TC(topo(P)) | e in E_feas) = new_MCS
    #       - Remap mappings from MCS to new_MCS.
    #       - Add mapping (P -> new_MCS).
    #  - Repeat until all partitions are matched.

    # run_excalidraw_example()

    # Record depth of each node in partition
    # Check tentative_candidates

    # Function inputs
    MUST_PLOT = False
    SEED = 29  # 50
    THETA = 0.07
    SIGMA = 0.3
    NUM_NODES = 3  # 7
    T = 100

    ROOT_KIND = None
    AS_TC = True
    NUM_TOPO = 1
    MAX_EVALS = 10000

    # Define graph
    fs = [float(i) for i in range(1, NUM_NODES + 1)]
    edges = {(i, (i + 1) % NUM_NODES) for i in range(NUM_NODES-1)}  # Add forward edges
    edges.update({(j, i) for i, j in edges})  # Add reverse edges
    edges.update({(i, i) for i in range(NUM_NODES)})  # Stateful edges

    # Create graph
    G = sg.create_graph(fs, edges, T, seed=SEED, theta=THETA, sigma=SIGMA)

    # Partition
    P = sg.balanced_partition(G, root_kind=ROOT_KIND)

    # Record original mapping
    partition_order = list(P.keys())

    # Determine all feasible edges between node types (E_val) = MCS
    E_val = sg.get_set_of_feasible_edges(P)

    # Select partition P that has the most edges (i.e. edges are interpreted as constraints).
    key_MCS, P_MCS = sorted(P.items(), key=lambda item: item[1].number_of_edges(), reverse=True)[0]

    # Determine starting pattern. monomorphism = {P_node: MCS_0_node}
    MCS, monomorphism = sg.as_MCS(P_MCS, E_val, num_topo=NUM_TOPO, as_tc=AS_TC)

    # Iterate over partitions sorted by the number of edges in descending order.
    P_sorted = {k: P for k, P in sorted(P.items(), key=lambda item: item[1].number_of_edges(), reverse=True)}

    # Find monomorphism
    # find_motif(P_sorted[102], MCS)

    # Define graph isomorphism function
    _is_node_attr_match = lambda motif_node_id, host_node_id, motif, host: host.nodes[host_node_id]["kind"] == motif.nodes[motif_node_id]["kind"]
    _is_node_attr_match = lru_cache()(_is_node_attr_match)
    _is_edge_attr_match = lambda motif_edge_id, host_edge_id, motif, host: True
    _is_edge_attr_match = lru_cache()(_is_edge_attr_match)
    # _L_is_edge_attr_match = lambda motif_edge_id, host_edge_id, motif, host: (host.nodes[host_edge_id[0]]["generation"] - host.nodes[host_edge_id[1]]["generation"]) >= (motif.nodes[motif_edge_id[0]]["generation"] - motif.nodes[motif_edge_id[1]]["generation"])
    _L_is_edge_attr_match = lambda motif_edge_id, host_edge_id, motif, host: host.nodes[host_edge_id[0]]["generation"] < host.nodes[host_edge_id[1]]["generation"]
    # _L_is_edge_attr_match = lru_cache()(_L_is_edge_attr_match)
    _L_is_node_structural_match = lambda motif_node_id, host_node_id, motif, host: True
    # _L_is_node_structural_match = lru_cache()(_L_is_node_structural_match)
    _is_node_structural_match = lambda motif_node_id, host_node_id, motif, host: host.in_degree(host_node_id) >= motif.in_degree(motif_node_id) and host.out_degree(host_node_id) >= motif.out_degree(motif_node_id)
    _is_node_structural_match = lru_cache()(_is_node_structural_match)

    if MUST_PLOT:
        fig, axes = plt.subplots(nrows=2)
        fig.set_size_inches(12, 15)
        sg.plot_graph(axes[0], MCS)
        plt.show()

    # Find MCS
    metric = {"match": 0, "no_match": 0}
    MCS_history: Dict[nx.DiGraph, int] = {MCS: 0}
    MCS_to_MCS: Dict[nx.DiGraph, Dict[nx.DiGraph, Dict[str, str]]] = {}
    P_to_MCS: Dict[int, Dict[nx.DiGraph, Dict[str, str]]] = {}
    for k, P in P_sorted.items():

        # Find subgraph monomorphism
        gr_start = time.time()
        mono = grandiso.find_motifs(P, MCS, limit=1, directed=True,
                                    is_node_attr_match=_is_node_attr_match,
                                    is_edge_attr_match=_is_edge_attr_match,
                                    is_node_structural_match=_is_node_structural_match)
        t_find_motifs = time.time() - gr_start

        if len(mono) > 0:  # Perfect match found! Save mapping: P -> subgraph(MCS)
            P_monomorphism = mono[0]
            P_to_MCS[k] = {MCS: P_monomorphism}
            metric["match"] += 1
            print(f"k={k} | Match    | find={t_find_motifs:.3f} sec | nodes=({MCS.number_of_nodes()}/{P.number_of_nodes()}) | edges=({MCS.number_of_edges()}/{P.number_of_edges()})")
        else:  # No match. Merge MCS with non-matched nodes in partition.
            rex_start = time.time()
            num_evals, is_match, largest_mono_lst = rex_mcs.find_largest_motifs(P, MCS,
                                                                                max_evals=MAX_EVALS,
                                                                                queue_=rex_mcs.Deque(policy=rex_mcs.QueuePolicy.BREADTHFIRST),
                                                                                is_node_attr_match=_is_node_attr_match,
                                                                                is_edge_attr_match=_L_is_edge_attr_match,
                                                                                is_node_structural_match=_L_is_node_structural_match)
            largest_mono = largest_mono_lst[0]
            t_find_large_motifs = time.time() - rex_start

            assert not is_match, "Should not have found a match."
            unmatched_nodes = len(P.nodes) - max([len(largest_mono), 0])

            # Prepare to unify MCS and P
            mcs = {v: k for k, v in largest_mono.items()}
            mcs_MCS = MCS.subgraph(mcs.keys())
            mcs_P = nx.relabel_nodes(MCS.subgraph(mcs.keys()), mcs, copy=True)
            E1 = sg.emb(mcs_MCS, MCS)
            E2 = sg.emb(mcs_P, P)
            tmp = nx.relabel_nodes(sg.unify(mcs_P, P, E2), largest_mono, copy=True)
            new_P_MCS = sg.unify(tmp, MCS, E1)

            if not nx.is_directed_acyclic_graph(new_P_MCS):
                print(largest_mono)
                # Color nodes
                for k, v in largest_mono.items():
                    new_P_MCS.nodes[v].update({"edgecolor": "green"})
                    MCS.nodes[v].update({"edgecolor": "green"})
                    P.nodes[k].update({"edgecolor": "green"})
                # Color edges
                for u, v in P.edges:
                    if u in largest_mono and v in largest_mono:
                        MCS.edges[(largest_mono[u], largest_mono[v])].update({"color": "green"})
                        new_P_MCS.edges[(largest_mono[u], largest_mono[v])].update({"color": "green"})
                        P.edges[(u, v)].update({"color": "green"})
                # Find cycles
                cycle = list(nx.find_cycle(new_P_MCS))
                print(cycle)
                for u, v in cycle:
                    if new_P_MCS[u][v]["color"] == "green":
                        continue
                    new_P_MCS[u][v]["color"] = "red"
                # Set edge alpha
                for u, v in new_P_MCS.edges:
                    if new_P_MCS[u][v]["color"] in ["green", "red"]:
                        new_P_MCS[u][v]["alpha"] = 1.0
                    else:
                        new_P_MCS[u][v]["alpha"] = 0.
                for u, v in MCS.edges:
                    if MCS[u][v]["color"] in ["green", "red"]:
                        MCS[u][v]["alpha"] = 1.0
                    else:
                        MCS[u][v]["alpha"] = 0.3
                for u, v in P.edges:
                    if P[u][v]["color"] in ["green", "red"]:
                        P[u][v]["alpha"] = 1.0
                    else:
                        P[u][v]["alpha"] = 0.5

                # Color cycles
                fig, axes = plt.subplots(nrows=3)
                fig.set_size_inches(12, 15)
                sg.plot_graph(axes[0], P)
                sg.plot_graph(axes[1], MCS)
                sg.plot_graph(axes[2], new_P_MCS)
                plt.show()

            assert nx.is_directed_acyclic_graph(new_P_MCS), "Should be a DAG."

            new_MCS, monomorphism = sg.as_MCS(new_P_MCS, E_val, num_topo=NUM_TOPO, as_tc=AS_TC)

            # Determine P_monomorphism
            P_monomorphism = {n: None for n in P.nodes}
            for k, v in P_monomorphism.items():
                if k in largest_mono:
                    P_monomorphism[k] = monomorphism[largest_mono[k]]
                elif k in monomorphism:
                    P_monomorphism[k] = monomorphism[k]

            # Verify that P_monomorphism is a valid mapping
            # They hint cannot already be a perfect match --> limitation of find_motifs.
            hints = [{k: v for idx, (k, v) in enumerate(P_monomorphism.items()) if idx > 0}]
            mono = grandiso.find_motifs(P, new_MCS, limit=1, directed=True, is_node_attr_match=_is_node_attr_match,
                                        is_edge_attr_match=_is_edge_attr_match, hints=hints)
            assert len(mono) > 0, "P_monomorphism is not a valid mapping."
            P_to_MCS[k] = {new_MCS: P_monomorphism}

            # Filter out added nodes from MCS_monomorphism
            MCS_monomorphism = {k: v for k, v in monomorphism.items() if k in MCS.nodes}

            # Remap mappings from MCS to new_MCS.
            idx_MCS = len(MCS_history)  # NOTE! not thread safe
            MCS_history[new_MCS] = idx_MCS
            MCS_to_MCS[MCS] = {new_MCS: MCS_monomorphism}

            # Update MCS
            # MCS = new_MCS
            metric["no_match"] += 1
            print(f"k={k} | No Match | find={t_find_motifs:.3f} sec | nodes=({MCS.number_of_nodes()}/{P.number_of_nodes()}) | edges=({MCS.number_of_edges()}/{P.number_of_edges()}) | large={t_find_large_motifs:.3f} sec | num_evals={num_evals} | unmatched_nodes={unmatched_nodes}")
    print(f"Matched {metric['match']} | No Match {metric['no_match']}")

    exit()

    metric = {"match": 0, "no_match": 0}
    for k, P in P_sorted.items():

        # Time function
        gr_start = time.time()
        queue = rex_mcs.Deque(policy=rex_mcs.QueuePolicy.BREADTHFIRST)
        try:
            x = next(grandiso.find_motifs_iter(P, MCS, directed=True, is_node_attr_match=_is_node_attr_match, is_edge_attr_match=_is_edge_attr_match))
            gr_match = True
        except StopIteration:
            gr_match = False
        gr_end = time.time()

        # Time function
        rex_start = time.time()
        queue = rex_mcs.Deque(policy=rex_mcs.QueuePolicy.BREADTHFIRST)
        num_evals, rex_match, largest_motif = rex_mcs.find_largest_motifs(P, MCS, queue_=queue, is_node_attr_match=_is_node_attr_match, is_edge_attr_match=_is_edge_attr_match)
        unmatched_nodes = len(P.nodes) - max([len(m) for m in largest_motif]+[0])
        rex_end = time.time()

        # Time function
        nx_start = time.time()
        try:
            matcher = isomorphism.GraphMatcher(MCS, P, node_match=isomorphism.categorical_node_match(attr="kind", default=None))
            x = next(matcher.subgraph_monomorphisms_iter())
            nx_match = True
            # nx_match = rex_match
        except StopIteration:
            nx_match = False
        nx_end = time.time()

        assert gr_match == rex_match, f"gr_match={gr_match} | rex_match={rex_match}"
        # assert gr_match == nx_match, f"gr_match={gr_match} | nx_match={nx_match}"
        if gr_match != nx_match:
            print(f"k={k} | gr_match={gr_match} | nx_match={nx_match}")

        if nx_match:
            metric["match"] += 1
            print(f"Match    | nx={nx_end-nx_start:.3f} sec | gr={gr_end-gr_start:.3f} sec | rex={rex_end-rex_start:.3f} sec | key={k} | nodes={P.number_of_nodes()} | edges={P.number_of_edges()}")
        else:
            metric["no_match"] += 1
            print(f"No Match | nx={nx_end-nx_start:.3f} sec | gr={gr_end-gr_start:.3f} sec | rex={rex_end-rex_start:.3f} sec | key={k} | nodes={P.number_of_nodes()} | edges={P.number_of_edges()} | unmatched_nodes={unmatched_nodes}")
    print(metric)
    input("Press Enter to continue...")
    exit()

    # # Define transitive closure (not perfect, because it does not add possible backward edges)
    # # Instead, should make transitive closure of each topo sort
    # G_tc = {}
    # for p_idx, g in G_partition.items():
    #     tc = g.copy()
    #     for o in tc:
    #         descendants = nx.descendants(g, o)
    #         tc_edges = [(o, i) for i in descendants if o not in tc[o] and (g.nodes[o]["kind"], g.nodes[i]["kind"]) in edge_set]
    #         tc.add_edges_from(tc_edges, **edge_data)
    #     G_tc[p_idx] = tc
    #
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
    #         plot_graph(axes[idx+2], G_tc[k])
    #         # tc = nx.transitive_closure(p)
    #         # plot_graph(axes[idx+3], tc)
    #
    # plt.show()


