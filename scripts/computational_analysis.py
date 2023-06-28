import networkx as nx
import supergraph as sg
import supergraph.evaluate

if __name__ == '__main__':
	# todo: Try perfect topological sort vs imperfect topological sort --> allow specifying custom topological sort algorithm
	# todo: store all edges in S
	# todo: make algorithm deterministic

	# Function inputs
	MUST_PLOT = False
	SEED = 11  # 22
	THETA = 0.07
	SIGMA = 0.1
	WINDOW = 0
	NUM_NODES = 20  # 10
	MAX_FREQ = 200
	COMBINATION_MODE = "linear"
	EPISODES = 2
	LENGTH = 100

	LEAF_KIND = 0
	BACKTRACK = 20
	SORT_FN = None  # supergraph.evaluate.perfect_sort
	# SORT_FN = supergraph.evaluate.perfect_sort

	# Define graph
	# fs = [1+i*(MAX_FREQ-1)/(NUM_NODES-1) for i in range(0, NUM_NODES)]
	fs = [float(i) for i in range(1, NUM_NODES)] + [MAX_FREQ]
	NUM_NODES = len(fs)
	edges = {(i, (i + 1) % len(fs)) for i in range(len(fs) - 1)}  # Add forward edges
	# edges.update({(j, i) for i, j in edges})  # Add all reverse edges
	edges.add((len(fs)-1, 0))  # Add one reverse edge
	edges.update({(i, i) for i in range(len(fs))})  # Stateful edges

	# Define graph
	Gs = []
	for i in range(EPISODES):
		G = supergraph.evaluate.create_graph(fs, edges, LENGTH, seed=SEED+i, theta=THETA, sigma=SIGMA)
		G = supergraph.evaluate.prune_by_window(G, WINDOW)
		G = supergraph.evaluate.prune_by_leaf(G, LEAF_KIND)
		Gs.append(G)

	# Define initial supergraph
	S_init, _ = sg.as_supergraph(Gs[0], leaf_kind=LEAF_KIND, sort=[f"{LEAF_KIND}_0"])

	# Grow supergraph
	S_sup, _S_init_to_S, _monomorphism = sg.grow_supergraph(Gs, S_init, LEAF_KIND,
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
