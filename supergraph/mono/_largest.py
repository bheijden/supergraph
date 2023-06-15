# from line_profiler_pycharm import profile
from typing import Dict, Generator, Hashable, List, Optional, Union, Tuple
from inspect import isclass
from functools import lru_cache
import itertools
import networkx as nx

import supergraph.mono._utils as utils


# @lru_cache()
def _is_node_attr_match(
		motif_node_id: str, host_node_id: str, motif: nx.DiGraph, host: nx.DiGraph
) -> bool:
	"""
	Check if a node in the host graph matches the attributes in the motif.

	Arguments:
		motif_node_id (str): The motif node ID
		host_node_id (str): The host graph ID
		motif (nx.DiGraph): The motif graph
		host (nx.DiGraph): The host graph

	Returns:
		bool: True if the host node matches the attributes in the motif

	"""
	motif_node = motif.nodes[motif_node_id]
	host_node = host.nodes[host_node_id]
	return host_node["kind"] == motif_node["kind"]


# @lru_cache()
def _is_node_structural_match(
		motif_node_id: str, host_node_id: str, motif: nx.DiGraph, host: nx.DiGraph
) -> bool:
	"""
	Check if the motif node here is a valid structural match.

	Specifically, this requires that a host node has at least the degree as the
	motif node.

	Arguments:
		motif_node_id (str): The motif node ID
		host_node_id (str): The host graph ID
		motif (nx.DiGraph): The motif graph
		host (nx.DiGraph): The host graph

	Returns:
		bool: True if the motif node maps to this host node

	"""
	return True


# @lru_cache()
def _is_edge_attr_match(
		motif_edge_id: Tuple[str, str],
		host_edge_id: Tuple[str, str],
		motif: nx.DiGraph,
		host: nx.DiGraph,
) -> bool:
	"""
	Check if an edge in the host graph matches the attributes in the motif.

	Arguments:
		motif_edge_id (str): The motif edge ID
		host_edge_id (str): The host edge ID
		motif (nx.DiGraph): The motif graph
		host (nx.DiGraph): The host graph

	Returns:
		bool: True (Always)

	"""

	return host.nodes[host_edge_id[0]]["generation"] < host.nodes[host_edge_id[1]]["generation"]


def find_largest_motifs(
		motif: nx.DiGraph,
		host: nx.DiGraph,
		interestingness: dict = None,
		queue_=None,
		max_evals: int = None,
		hints: List[Dict[Hashable, Hashable]] = None,
		is_node_structural_match=_is_node_structural_match,
		is_node_attr_match=_is_node_attr_match,
		is_edge_attr_match=_is_edge_attr_match,
		workers: int = 1,
) -> Tuple[int, bool, List[Dict[str, str]]]:
	"""
	Yield mappings from motif node IDs to host graph IDs.

	Results are of the form:

	```
	{motif_id: host_id, ...}
	```

	Arguments:
		motif (nx.DiGraph): The motif graph (needle) to search for
		host (nx.DiGraph): The host graph (haystack) to search within
		interestingness (dict: None): A map of each node in `motif` to a float
			number that indicates an ordinality in which to address each node
		queue_ (queue.SimpleQueue): What kind of queue to use.
		hints (dict): A dictionary of initial starting mappings. By default,
			searches for all instances. You can constrain a node by passing a
			list with a single dict item: `[{motifId: hostId}]`.

	Returns:
		Generator[dict, None, None]

	"""
	# TODO: Possible optimizations:
	# 1. Prepare possible structural matches beforehand.
	# 2. Prune candidates that are monomorphic with other candidates. (i.e. have already been evaluated or are monomorphic to a subgraph of previous candidates)
	# 3. [DONE] Add new candidates to the queue in order of interestingness.
	# 4. Exclude new candidates for which the max number of potentially addable nodes is less than the number of nodes in the largest candidate.
	# 5. [DONE] Add max number of candidate evaluations, and return the largest candidate when this number is reached.
	# 6. Investigate how to use multi-processing with this algorithm (i.e. share a queue?)
	# 7. Can we use lru_cache to detect candidates that have already been evaluated?
	# 8. Define initial starting mapping to use as hints.
	# 9. Can we determine the max number of nodes on the maximal common (monomorphic) subgraph i.e. stop searching when this number is reached?
	#    This should make use of structural mismatches in both graph (i.e. count of node types).
	interestingness = interestingness or utils.uniform_node_interestingness(motif)

	# Make sure all nodes are included in the interestingness dict
	interestingness = {n: interestingness.get(n, 0.) for n in motif.nodes}

	# Sort the interestingness dict by value:
	interestingness = {k: v for k, v in sorted(interestingness.items(), reverse=True, key=lambda item: item[1])}

	# Kick off the queue with hints:
	init_q = utils.Deque(policy=utils.QueuePolicy.DEPTHFIRST)
	hints = [] if hints is None else hints
	for hint in hints:
		init_q.put(hint)

	# Add to the queue with initial candidates:
	# todo: sort by interestingness/generation?
	components = list(nx.weakly_connected_components(motif))
	candidates = [[] for _ in range(len(components))]
	for i, comp in enumerate(components):
		for next_node in comp:
			for n in host.nodes:
				node_attr_match = is_node_attr_match(next_node, n, motif, host)
				node_struct_match = is_node_structural_match(next_node, n, motif, host)
				if node_attr_match and node_struct_match:
					candidates[i].append({next_node: n})
	for c in itertools.product(*candidates):
		c: Tuple[Dict[Hashable, Hashable], ...]
		candidate = {k: v for d in c for k, v in d.items()}
		# Merge all dicts into a single dict if all values are unique:
		if len(set(candidate.values())) == len(c):
			# Add to the queue:
			init_q.put(candidate)

	# Prepare the queue:
	queue_ = queue_ or utils.Deque(policy=utils.QueuePolicy.DEPTHFIRST)
	q_inner = queue_() if isclass(queue_) else queue_

	# Prepare and test all possible candidates
	largest_candidates = []
	largest_candidate_size = 0
	num_evals = 0
	while not init_q.empty():
		q_inner.put(init_q.get())
		while not q_inner.empty():
			if not (max_evals is None or num_evals < max_evals):
				# print(f"Reached max number of candidate evaluations per motif: {max_evals}. Returning largest candidates.")
				break
			num_evals += 1
			# NOTE! new_backbone: dict --> previous (incomplete) candidate
			new_backbone = q_inner.get()
			next_candidate_backbones = get_next_backbone_candidates(
				new_backbone,
				motif,
				host,
				interestingness,
				is_node_structural_match=is_node_structural_match,
				is_node_attr_match=is_node_attr_match,
				is_edge_attr_match=is_edge_attr_match,
			)

			for candidate in next_candidate_backbones:
				# print(candidate)
				if len(candidate) > largest_candidate_size:
					largest_candidate_size = len(candidate)
					largest_candidates = [candidate]
					# print(largest_candidate_size)
				elif len(candidate) == largest_candidate_size:
					largest_candidates.append(candidate)
				if len(candidate) == len(motif):
					return num_evals, True, largest_candidates
				else:
					q_inner.put(candidate)
	return num_evals, False, largest_candidates

# @profile
def get_next_backbone_candidates(
		backbone: dict,
		motif: nx.DiGraph,
		host: nx.DiGraph,
		interestingness: dict,
		next_node: str = None,
		is_node_structural_match=_is_node_structural_match,
		is_node_attr_match=_is_node_attr_match,
		is_edge_attr_match=_is_edge_attr_match,
) -> List[Dict[str, str]]:
	"""
	Get a list of candidate node assignments for the next "step" of this map.

	Arguments:
		backbone (dict): Mapping of motif node IDs to one set of host graph IDs
		motif (Graph): A graph representation of the motif
		host (Graph): The host graph, complete
		interestingness (dict): A mapping of motif node IDs to interestingness
		next_node (str: None): Optional suggestion for the next node to assign

	Returns:
		List[dict]: A new list of mappings with one additional element mapped

	"""
	# initial backbone:
	#   - Outside of this function.
	#   - Every initial backbone should include one match within every disconnected component in the motif.
	#   - Parallelize over initial candidates.
	# forward/backward: backbone_motif[-1]["depth] > backbone_motif[0]["depth] or len(backbone_motif[-1]["depth]) == 0
	# next_nodes: Start forward search:
	#   - front_out: Search for nodes not in backbone, but connected with (n_bb, n_front_out) out_edges to backbone nodes.
	#   - in_edges(n_front_out): Iterate over all (u or n_bb, n_front_out) in_edges of every node in front_out
	#       - filter(n_front_out): if u not already in backbone(=n_bb) AND descendant to any node already in backbone.
	# next_nodes: Backward search:
	#   - front_in: Search for nodes not in backbone, but connected with (n_front_in, n_bb) in_edges to backbone nodes.
	#   - out_edges(n_front_in): Iterate over all (n_front_in, v or n_bb) out_edges of every n_front_in in front_out
	#       - filter(n_font_in): if v not already in backbone(=n_bb) AND ancestor to any node already in backbone.
	# Sort (front_in, front_out): len(ancestors) and len(descendants) of each node in (front_in, front_out)
	# Find next_node as max(len(ancestors), len(descendants)) in (front_in, front_out)
	# Find host attr matches: for every node in (front_in, front_out) find all node attr matches in host.
	#   - node_attr_match: check if node type matches
	#   - edge_attr_match: check gen(host.u) <=gen(host.v)

	# NOTE!: is next_node ever != None? I don't think so, so we can remove it

	# Get a list of the "exploration front" of the motif -- nodes that are not
	# yet assigned in the backbone but are connected to at least one assigned
	# node in the backbone. NOTE! We modified this to propose new nodes that are disconnected.

	# For example, in the motif A -> B -> C, if A is already assigned, then the
	# front is [B] (c is not included because it has no connection to any
	# assigned node in the backbone).

	# We should prefer nodes that are connected to multiple assigned backbone
	# nodes, because these will filter more rapidly to a smaller set.

	# Initial backbones are always
	if next_node is None and len(backbone) == 0:
		raise NotImplementedError("Should not be here!")

	# _nodes_with_greatest_backbone_count: List[str] = []
	# _greatest_backbone_count = 0
	# for motif_node_id in motif.nodes:
	# 	# todo: here we find all nodes that are not IN the backbone, but are connected to nodes IN the backbone
	# 	if motif_node_id in backbone:
	# 		continue
	# 	# How many connections to existing backbone?
	# 	# Note that this number is certainly greater than or equal to 1,
	# 	# since a value of 0 would imply that the backbone dict is empty
	# 	# (which we have already handled) or that the motif has more than
	# 	# one connected component, which we check for at prep-time.
	# 	# todo: modify, to search for nodes that are
	# 	motif_backbone_connections_count = sum(
	# 		[
	# 			1
	# 			for v in list(
	# 			set(motif.adj[motif_node_id]).union(  # NOTE: adj=adjacent (i.e. neighboring nodes)
	# 				set(motif.pred[motif_node_id])  # NOTE: pred=predecessors (i.e. nodes that point to this node)
	# 			)
	# 		)
	# 			if v in backbone
	# 		]
	# 	)
	# 	# If this is the most highly connected node visited so far, then
	# 	# set it as the next node to explore:
	# 	if motif_backbone_connections_count > _greatest_backbone_count:
	# 		_nodes_with_greatest_backbone_count.append(motif_node_id)

	# forward/backward: backbone_motif[-1]["depth] > backbone_motif[0]["depth] or len(backbone_motif[-1]["depth]) == 0
	# next_nodes: Start forward search:
	#   - front_out: Search for nodes not in backbone, but connected with (n_bb, n_front_out) out_edges to backbone nodes.
	#   - in_edges(n_front_out): Iterate over all (u or n_bb, n_front_out) in_edges of every node in front_out
	#       - filter(n_front_out): if u not already in backbone(=n_bb) AND descendant to any node already in backbone.
	# next_nodes: Backward search:
	#   - front_in: Search for nodes not in backbone, but connected with (n_front_in, n_bb) in_edges to backbone nodes.
	#   - out_edges(n_front_in): Iterate over all (n_front_in, v or n_bb) out_edges of every n_front_in in front_out
	#       - filter(n_font_in): if v not already in backbone(=n_bb) AND ancestor to any node already in backbone.
	# Sort (front_in, front_out): len(ancestors) and len(descendants) of each node in (front_in, front_out)
	# Find next_node as max(len(ancestors), len(descendants)) in (front_in, front_out)

	# todo: use lru cache to speed up finding front_out and front_in given a backbone.
	MUST_BREAK = False
	# OUT_FILTER = "descendants"
	# IN_FILTER = "ancestors"
	OUT_FILTER = "ancestors"
	IN_FILTER = "descendants"
	front_out = {}
	front_in = {}
	for motif_node_id in motif.nodes:
		if motif_node_id in backbone:
			continue

		all_in_edges_to_bb = True
		any_in_edges_to_bb = False
		for u in motif.pred[motif_node_id]:
			if u in backbone:
				any_in_edges_to_bb = any_in_edges_to_bb or True
			else:
				all_in_edges_to_bb = False
		all_in_edges_to_bb = all_in_edges_to_bb and any_in_edges_to_bb
		some_in_edges_to_bb = any_in_edges_to_bb and not all_in_edges_to_bb

		if all_in_edges_to_bb:
			front_out[motif_node_id] = len(motif.nodes[motif_node_id]["descendants"])
			if MUST_BREAK:
				break
			continue  # short-circuit, we cannot be in both front_in and front_out
		elif some_in_edges_to_bb:
			_no_bb_ancestors = True
			for n_bb in backbone:
				if n_bb in motif.nodes[motif_node_id][OUT_FILTER]:  # todo
					_no_bb_ancestors = False
					break
			if _no_bb_ancestors:
				front_out[motif_node_id] = len(motif.nodes[motif_node_id]["descendants"])
				if MUST_BREAK:
					break
			continue  # short-circuit, we cannot be in both front_in and front_out

		all_out_edges_to_bb = True
		any_out_edges_to_bb = False
		for v in motif.adj[motif_node_id]:
			if v in backbone:
				any_out_edges_to_bb = any_out_edges_to_bb or True
			else:
				all_out_edges_to_bb = False
		all_out_edges_to_bb = all_out_edges_to_bb and any_out_edges_to_bb
		some_out_edges_to_bb = any_out_edges_to_bb and not all_out_edges_to_bb

		if all_out_edges_to_bb:
			front_in[motif_node_id] = len(motif.nodes[motif_node_id]["ancestors"])
			if MUST_BREAK:
				break
		elif some_out_edges_to_bb:
			_no_bb_descendants = True
			for n_bb in backbone:
				if n_bb in motif.nodes[motif_node_id][IN_FILTER]:  # todo
					_no_bb_descendants = False
					break
			if _no_bb_descendants:
				front_in[motif_node_id] = len(motif.nodes[motif_node_id]["ancestors"])
				if MUST_BREAK:
					break

	# Sort front_out and front_in based on len(descendants) and len(ancestors)
	# todo: iterate over all nodes in next_nodes.
	next_nodes = sorted(itertools.chain(front_out.items(), front_in.items()), key=lambda x: x[1], reverse=True)
	if len(next_nodes) == 0:
		return []
	next_node = next_nodes[0][0]

	# Now we have a node `next_node` which we know is connected to the current
	# backbone. Get all edges between `next_node` and nodes in the backbone,
	# and verify that they exist in the host graph:
	# `required_edges` has the form (prev, self, next), with non-values filled
	# with None. That way we can easily remember and store the roles of the
	# node IDs in the next step.
	required_edges = []
	for other in list(motif.adj[next_node]):
		if other in backbone:
			# edge is (next_node, other)
			required_edges.append((None, next_node, other))
	for other in list(motif.pred[next_node]):
		if other in backbone:
			# edge is (other, next_node)
			required_edges.append((other, next_node, None))

	# `required_edges` now contains a list of all edges that exist in the motif
	# graph, and we must find candidate nodes that have such edges in the host.

	candidate_nodes = []

	# In the worst-case, `required_edges` has length == 1. This is the worst
	# case because it means that ALL edges from/to `other` are valid options.
	if len(required_edges) == 1:
		# :(
		(source, _, target) = required_edges[0]
		if source is not None:
			# this is a "from" edge:
			candidate_nodes = list(host.adj[backbone[source]])
		elif target is not None:
			# this is a "from" edge:
			candidate_nodes = list(host.pred[backbone[target]])
	# Thus, all candidates for motif ID `$next_node` are stored in the
	# candidate_nodes list.

	elif len(required_edges) > 1:
		# This is neato :) It means that there are multiple edges in the host
		# graph that we can use to downselect the number of candidate nodes.
		candidate_nodes_set = None
		for (source, _, target) in required_edges:
			if source is not None:
				# this is a "from" edge:
				candidate_nodes_from_this_edge = host.adj[backbone[source]]
			# elif target is not None:
			else:  # target is not None:
				# this is a "from" edge:
				candidate_nodes_from_this_edge = host.pred[backbone[target]]
			# else:
			#     raise AssertionError("Encountered an impossible condition: At least one of source or target must be defined.")
			if candidate_nodes_set is None: #len(candidate_nodes_set) == 0:
				# This is the first edge we're checking, so set the candidate
				# nodes set to ALL possible candidates.
				candidate_nodes_set = set(candidate_nodes_from_this_edge)
			else:
				candidate_nodes_set = candidate_nodes_set.intersection(
					candidate_nodes_from_this_edge
				)
		candidate_nodes = list(candidate_nodes_set)

	elif len(required_edges) == 0:
		# Somehow you found a node that doesn't have any edges. This is bad.
		raise ValueError(
			f"Somehow you found a motif node {next_node} that doesn't have "
			+ "any motif-graph edges. This is bad. (Did you maybe pass an "
			+ "empty backbone to this function?)"
		)

	tentative_results = [
		{**backbone, next_node: c}
		for c in candidate_nodes
		if c not in backbone.values()
		   and is_node_attr_match(next_node, c, motif, host)
		   and is_node_structural_match(next_node, c, motif, host)
	]

	# One last filtering step here. This is to catch the cases where you have
	# successfully mapped each node, and the final node has some valid
	# candidate_nodes (and therefore `tentative_results`).
	# This is important: We must now check that for the assigned nodes, all
	# edges between them DO exist in the host graph. Otherwise, when we check
	# in find_motifs that len(motif) == len(mapping), we will discover that the
	# mapping is "complete" even though we haven't yet checked it at all.

	monomorphism_candidates = []
	for mapping in tentative_results:
		# todo: Can we get away with just checking the newly added by next_node to the backbone in the host?
		if len(mapping) == len(motif):
			# NOTE! This never happens....
			check_motif = all([
				host.has_edge(mapping[motif_u], mapping[motif_v])
				and is_edge_attr_match(
					(motif_u, motif_v),
					(mapping[motif_u], mapping[motif_v]),
					motif,
					host,
				)
				for motif_u, motif_v in motif.edges if motif_u in mapping and motif_v in mapping
			])
			if check_motif:
				# This is a "complete" match!
				monomorphism_candidates.append(mapping)
		else:
			# This is a partial match, so we'll continue building.
			monomorphism_candidates.append(mapping)


		# if check_motif:
		# 	monomorphism_candidates.append(mapping)
	# if len(mapping) == len(motif):
	#     if all(
	#         [
	#             host.has_edge(mapping[motif_u], mapping[motif_v])  # NOTE! Only checked for potential complete matches!
	#             and is_edge_attr_match(
	#                 (motif_u, motif_v),
	#                 (mapping[motif_u], mapping[motif_v]),
	#                 motif,
	#                 host,
	#             )
	#             for motif_u, motif_v in motif.edges
	#         ]
	#     ):
	#         # This is a "complete" match!
	#         monomorphism_candidates.append(mapping)
	# else:
	#     # This is a partial match, so we'll continue building.
	#     monomorphism_candidates.append(mapping)

	return monomorphism_candidates
#
# # Additionally, if isomorphisms_only == True, we can use this opportunity
# # to confirm that no spurious edges exist in the induced subgraph:
# isomorphism_candidates = []
# for result in monomorphism_candidates:
#     for (motif_u, motif_v) in itertools.product(result.keys(), result.keys()):
#         # if the motif has this edge, then it doesn't rule any of the
#         # above results out as an isomorphism.
#         # if the motif does NOT have the edge, then NO RESULT may have
#         # the equivalent edge in the host graph:
#         if not motif.has_edge(motif_u, motif_v) and host.has_edge(
#             result[motif_u], result[motif_v]
#         ):
#             # this is a violation.
#             break
#     else:
#         isomorphism_candidates.append(result)
# return isomorphism_candidates
