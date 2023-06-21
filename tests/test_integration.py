import supergraph as sg


def test_integration():
	SEED = 20  # 22
	THETA = 0.07
	SIGMA = 0.3
	WINDOW = 0
	NUM_NODES = 12  # 30
	T = 100

	LEAF_KIND = 0
	BACKTRACK = 20

	# Define graph
	# fs = [1, 2, 3, 4, 5, 10, 20, 20, 20, 40, 200]
	fs = [float(i) for i in range(1, NUM_NODES + 1)] + [200]
	# fs = [float(i) for i in range(1, NUM_NODES + 1)]
	NUM_NODES = len(fs)
	edges = {(i, (i + 1) % len(fs)) for i in range(len(fs) - 1)}  # Add forward edges
	edges.update({(j, i) for i, j in edges})  # Add reverse edges
	# edges.update({(len(fs)-1, 0) for i, j in edges})  # Add reverse edges
	edges.update({(i, i) for i in range(len(fs))})  # Stateful edges

	# Define graph
	G = sg.create_graph(fs, edges, T, seed=SEED, theta=THETA, sigma=SIGMA)
	G = sg.prune_by_window(G, WINDOW)

	# Define leafs
	leafs_G = {n: data for n, data in G.nodes(data=True) if data["kind"] == LEAF_KIND}
	leafs_G = [k for k in sorted(leafs_G.keys(), key=lambda k: leafs_G[k]["seq"])]

	# Define initial supergraph
	S_init, _ = sg.sort_to_S(G, [f"{LEAF_KIND}_0"], edges)

	# Grow supergraph
	S_rec, _S_init_to_S, _monomorphism = sg.grow_supergraph(G, S_init, LEAF_KIND, edges, backtrack=BACKTRACK, progress_bar=True)

	# Define linear supergraph (benchmark)
	S_lin, monomorphism_lin = next(sg.linear_S_iter(G, edges))
	units_lin, pred_lin, m_lin = sg.evaluate_supergraph(G, S_lin)
	print(f"S_lin  | Number of nodes: {pred_lin}/{len(G)} | number of units: {units_lin}")
	units_rec, pred_rec, m_rec = sg.evaluate_supergraph(G, S_rec)
	print(f"S_rec | Number of nodes: {pred_rec}/{len(G)} | number of units: {units_rec}/{len(leafs_G)}")
	size = len(S_rec)
	supergraph_nodes = size * len(leafs_G)
	matched_nodes = len(_monomorphism)
	efficiency = matched_nodes / supergraph_nodes
	print(f"matched {matched_nodes}/{len(G)} ({efficiency:.2%} efficiency, {size} size)")