import networkx as nx
import supergraph as sg
import supergraph.evaluate
import supergraph.paper as paper

if __name__ == '__main__':
	# Function inputs
	MUST_PLOT = False
	SEED = 11  # 22
	THETA = 0.07
	SIGMA = 0.3
	WINDOW = 0
	NUM_NODES = 8  # 10
	MAX_FREQ = 20
	COMBINATION_MODE = "linear"
	EPISODES = 2
	LENGTH = 100

	LEAF_KIND = "agent"
	BACKTRACK = 3
	SORT_FN = None  # supergraph.evaluate.perfect_sort

	# Get excalidraw graph
	G0, G1, G2 = supergraph.paper.get_example_graphs()
	Gs = [G0, G1, G2]

	# Run paper algorithms
	S_paper, Ms, Ps = paper.algorithm_1(supervisor="agent", backtrack=BACKTRACK, Gs=Gs, max_topo=1, max_comb=1)

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
	# Does equate sg.grow_supergraph efficiency, because an additional partition is added for the last
	# actuator, which is not an ancestor of any of the leafs.
	m_sup = sg.evaluate_supergraph(Gs, S_sup, progress_bar=True, name="S_sup")
	m_lin = sg.evaluate_supergraph(Gs, S_lin, progress_bar=True, name="S_lin")
	m_gen = sg.evaluate_supergraph(Gs, S_gen, progress_bar=True, name="S_gen")
	m_top = sg.evaluate_supergraph(Gs, S_top, progress_bar=True, name="S_top")
