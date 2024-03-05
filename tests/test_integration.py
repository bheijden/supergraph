import supergraph as sg
import supergraph.evaluate
import supergraph.paper as paper


def test_integration():
	SEED = 20  # 22
	THETA = 0.07
	SIGMA = 0.3
	WINDOW = 0
	NUM_NODES = 5  # 30
	MAX_FREQ = 200
	COMBINATION_MODE = "linear"
	EPISODES = 2
	LENGTH = 100

	LEAF_KIND = 0
	BACKTRACK = 20
	SORT_FN = None

	# Define graph
	fs = [float(i) for i in range(1, NUM_NODES)] + [MAX_FREQ]
	edges = {(i, (i + 1) % len(fs)) for i in range(len(fs) - 1)}  # Add forward edges
	edges.update({(j, i) for i, j in edges})  # Add reverse edges
	# edges.update({(len(fs)-1, 0) for i, j in edges})  # Add reverse edges
	edges.update({(i, i) for i in range(len(fs))})  # Stateful edges

	# Define graph
	Gs = []
	for i in range(EPISODES):
		G = supergraph.evaluate.create_graph(fs, edges, LENGTH, seed=SEED+i, theta=THETA, sigma=SIGMA)
		G = supergraph.evaluate.prune_by_window(G, WINDOW)
		G = supergraph.evaluate.prune_by_leaf(G, LEAF_KIND)
		Gs.append(G)

	# Define leafs
	leafs_G = {n: data for n, data in G.nodes(data=True) if data["kind"] == LEAF_KIND}
	leafs_G = [k for k in sorted(leafs_G.keys(), key=lambda k: leafs_G[k]["seq"])]

	# Grow supergraph
	S_sup, _S_init_to_S, _monomorphism = sg.grow_supergraph(Gs, LEAF_KIND,
	                                                        combination_mode=COMBINATION_MODE,
	                                                        backtrack=BACKTRACK,
	                                                        sort_fn=SORT_FN,
	                                                        progress_bar=True, validate=True)

	# Baseline lin: forloop-like supergraph, one node of each kind, not taking leaf_kind into account.
	# Baseline top: topological sort, and equalize # of nodes in-between leafs
	# Baseline gen: topological sort, and perform generational sort on partitions of nodes in-between leafs, equalize # of generations in-between leafs
	linear_iter = supergraph.evaluate.linear_S_iter(Gs)
	S_lin, monomorphism_lin = next(linear_iter)  # todo: evaluate order of S_lin
	S_top, S_gen = supergraph.evaluate.baselines_S(Gs, LEAF_KIND)

	# Evaluate performance
	m_sup = sg.evaluate_supergraph(Gs, S_sup, progress_bar=True, name="S_sup")
	m_lin = sg.evaluate_supergraph(Gs, S_lin, progress_bar=True, name="S_lin")
	m_gen = sg.evaluate_supergraph(Gs, S_gen, progress_bar=True, name="S_gen")
	m_top = sg.evaluate_supergraph(Gs, S_top, progress_bar=True, name="S_top")

	# Print results
	print(sum([len(m) for m in m_sup]), sum([len(m) for m in m_lin]), sum([len(m) for m in m_gen]), sum([len(m) for m in m_top]))


def test_paper_integration():
	# Generate three example computation graphs
	G0, G1, G2 = paper.get_example_graphs()
	Gs = [G0, G1, G2]

	# Find the Minimum Common Supergraph (S) and mappings (Ms) to the corresponding partitions (Ps)
	# IMPORTANT! paper.algorithm_1(...) is optimized for readability, instead of speed.
	# Use supergraph.grow_supergraph(...) instead if speed is important.
	S, Ms, Ps = paper.algorithm_1(supervisor="agent", backtrack=3, Gs=Gs, max_topo=1, max_comb=1)


if __name__ == "__main__":
	test_integration()
	test_paper_integration()