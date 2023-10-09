from functools import partial
import networkx as nx
import matplotlib.pyplot as plt
import supergraph as sg
import supergraph.evaluate as eval
import supergraph.paper as paper


# def progress_fn(t_elapsed, Gs_num_partitions, Gs_matched, i_partition, G_monomorphism, G, S):


if __name__ == '__main__':
	MUST_PLOT = True
	COMBINATION_MODE = "linear"
	LEAF_KIND = "agent"
	BACKTRACK = 20
	SORT_FN = None  # supergraph.evaluate.perfect_sort

	# Define graph
	Gs = list(paper.get_example_graphs())
	Gs = [Gs[1], Gs[2], Gs[0]]

	if MUST_PLOT:
		fig, axes = plt.subplots(nrows=3)
		fig.set_size_inches(12, 15)
		eval.plot_graph(axes[0], Gs[0])
		eval.plot_graph(axes[1], Gs[1])
		eval.plot_graph(axes[2], Gs[2])

	# Define initial supergraph
	S_init, _ = sg.as_supergraph(Gs[0], leaf_kind=LEAF_KIND, sort=[f"{LEAF_KIND}_0"])

	# Grow supergraphS
	Ss = []
	for G in Gs:
		S_sup, _S_init_to_S, _monomorphism = sg.grow_supergraph([G], S_init, LEAF_KIND,
		                                                        combination_mode=COMBINATION_MODE,
		                                                        backtrack=BACKTRACK,
		                                                        sort_fn=SORT_FN,
		                                                        # progress_fn=_progress_fn,
		                                                        progress_bar=True, validate=True)
		Ss.append(S_sup)

	# Plot supergraph
	if MUST_PLOT:
		fig, axes = plt.subplots(nrows=3)
		fig.set_size_inches(12, 15)
		eval.plot_graph(axes[0], Ss[0])
		eval.plot_graph(axes[1], Ss[1])
		eval.plot_graph(axes[2], Ss[2])
		plt.show()

