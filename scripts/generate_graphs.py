import networkx as nx
import supergraph as sg
import supergraph.evaluate


def generate(theta: float, sigma: float, window: int, seed: int, num_nodes: int, max_freq: int, T: int, eps: int):
	pass


if __name__ == '__main__':
	# todo: reduce computation time for generation graph
	# todo: store all edges in S

	# Function inputs
	SEED = 13  # 22
	THETA = 0.07
	SIGMA = 0.0
	WINDOW = 0
	NUM_NODES = 15  # 20
	MAX_FREQ = 200
	T = 100

	LEAF_KIND = 0
	BACKTRACK = 20

	# Define graph
	# fs = [1+i*(MAX_FREQ-1)/(NUM_NODES-1) for i in range(0, NUM_NODES)]
	fs = [float(i) for i in range(1, NUM_NODES + 1)] + [MAX_FREQ]
	# fs = [float(i) for i in range(1, NUM_NODES + 1)]
	NUM_NODES = len(fs)
	edges = {(i, (i + 1) % len(fs)) for i in range(len(fs) - 1)}  # Add forward edges
	# edges.update({(j, i) for i, j in edges})  # Add reverse edges
	edges.add((len(fs)-1, 0))  # Add reverse edges
	edges.update({(i, i) for i in range(len(fs))})  # Stateful edges

	# Define graph
	G = supergraph.evaluate.create_graph(fs, edges, T, seed=SEED, theta=THETA, sigma=SIGMA)
	G = supergraph.evaluate.prune_by_window(G, WINDOW)
	G = supergraph.evaluate.prune_by_leaf(G, LEAF_KIND)


