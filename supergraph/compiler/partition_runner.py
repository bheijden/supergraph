from typing import Any, Dict, List, Tuple, Callable, Union
import functools
import networkx as nx
from flax.core import FrozenDict
import jax
import jax.numpy as jnp
import numpy as onp
import supergraph.compiler.jax_utils as rjax
from supergraph.compiler.utils import check_generations_uniformity
from supergraph.compiler.node import BaseNode
from supergraph.compiler.base import InputState, StepState, GraphState, Output, GraphBuffer, Timings, SlotVertex


int32 = Union[jnp.int32, onp.int32]
float32 = Union[jnp.float32, onp.float32]


def invert_dict_with_list_values(input_dict):
    """
    Inverts a dictionary to create a new dictionary where the keys are the unique values
    of the input dictionary, and the values are lists of keys from the input dictionary
    that corresponded to each unique value.

    :param input_dict: The dictionary to invert.
    :return: An inverted dictionary with lists as values.
    """
    inverted_dict = {}
    for key, value in input_dict.items():
        # Add the key to the list of keys for the particular value
        if value not in inverted_dict:
            inverted_dict[value] = []
        inverted_dict[value].append(key)
    return inverted_dict


def get_buffer_size(buffer: GraphBuffer) -> jnp.int32:
    leaves = jax.tree_util.tree_leaves(buffer)
    size = leaves[0].shape[0] if len(leaves) > 0 else 1
    return size


def update_output(buffer: GraphBuffer, output: Output, seq: int32) -> Output:
    size = get_buffer_size(buffer)
    mod_seq = seq % size
    # new_buffer = jax.tree_map(lambda _b, _o: rjax.index_update(_b, mod_seq, _o, copy=True), buffer, output)
    new_buffer = jax.tree_map(lambda _b, _o: jnp.array(_b).at[mod_seq].set(jnp.array(_o)), buffer, output)
    return new_buffer


def make_update_state(name: str):
    def _update_state(graph_state: GraphState, timing: SlotVertex, step_state: StepState, output: Any) -> GraphState:
        # Define node's step state update
        new_nodes = dict()
        new_outputs = dict()

        # Increment sequence number
        new_ss = step_state.replace(seq=step_state.seq + 1)

        # Add node's step state update
        new_nodes[name] = new_ss
        new_outputs[name] = update_output(graph_state.buffer[name], output, timing.seq)

        graph_state = graph_state.replace_buffer(new_outputs)
        new_graph_state = graph_state.replace(nodes=graph_state.nodes.copy(new_nodes))
        return new_graph_state

    return _update_state


def make_update_inputs(name: str, inputs_data: Dict[str, Dict[str, str]]):
    def _update_inputs(graph_state: GraphState, timings_node: SlotVertex) -> StepState:
        ss = graph_state.nodes[name]
        ts_start = timings_node.ts_start
        eps = graph_state.eps
        seq = timings_node.seq
        new_inputs = dict()
        for node_name, input_name in inputs_data.items():
            t = timings_node.windows[node_name]
            buffer = graph_state.buffer[node_name]
            size = get_buffer_size(buffer)
            mod_seq = t.seq % size
            inputs = rjax.tree_take(buffer, mod_seq)
            _new = InputState.from_outputs(t.seq, t.ts_sent, t.ts_recv, inputs, is_data=True)
            new_inputs[input_name] = _new
        return ss.replace(eps=eps, seq=seq, ts=ts_start, inputs=FrozenDict(new_inputs))

    return _update_inputs


# def make_run_partition_excl_supervisor(nodes: Dict[str, "BaseNode"], S: nx.DiGraph, supervisor_slot: str, generations: List[List[str]], skip: List[str] = None):
def make_run_partition_excl_supervisor(
    nodes: Dict[str, "BaseNode"], timings: Timings, S: nx.DiGraph, supervisor_slot: str, skip: List[str] = None
):
    INTERMEDIATE_UPDATE = False

    # Determine input_name mapping
    input_name_mapping = {n: {} for n in nodes.keys()}
    for n, mapping in input_name_mapping.items():
        for input_name, c in nodes[n].inputs.items():
            mapping[c.output_node.name] = input_name
    update_input_fns = {name: make_update_inputs(name, mapping) for name, mapping in input_name_mapping.items()}

    # Define update function
    supervisor = timings.slots[supervisor_slot].kind
    supervisor_gen_idx = timings.slots[supervisor_slot].generation

    # Determine if all generations contain all slot_kinds
    # NOTE! This assumes that the supervisor is the only node in the last generation.
    generations = timings.to_generation()
    is_uniform = check_generations_uniformity(generations[:-1])
    slots_to_kinds = {n: v.kind for n, v in timings.slots.items()}
    kinds_to_slots = invert_dict_with_list_values(slots_to_kinds)
    kinds_to_slots.pop(supervisor)  # remove supervisor from kinds_to_slots
    for key, value in kinds_to_slots.items():
        # sort value based on the generation they belong to.
        kinds_to_slots[key] = sorted(value, key=lambda x: timings.slots[x].generation)

    # Determine which slots to skip
    skip_slots = [n for n, v in timings.slots.items() if v.kind in skip] if skip is not None else []
    skip_slots = (
        skip_slots + skip if skip is not None else skip_slots
    )  # also add kinds to skip slots, because if uniform, then kinds are also slots.

    def _run_node(kind: str, graph_state: GraphState, timings_node: SlotVertex):
        # Update inputs
        ss = update_input_fns[kind](graph_state, timings_node)
        # ss = _old_ss

        # Run node step
        _new_ss, output = nodes[kind].step(ss)

        # Increment sequence number
        _new_seq_ss = _new_ss.replace(seq=_new_ss.seq + 1)

        # Update output buffer
        _new_output = update_output(graph_state.buffer[kind], output, timings_node.seq)
        # _new_output = graph_state.outputs[kind]
        return _new_seq_ss, _new_output

    node_step_fns = {kind: functools.partial(_run_node, kind) for kind in nodes.keys()}

    def _run_generation(graph_state: GraphState, timings_gen: Dict[str, SlotVertex]):
        new_nodes = dict()
        new_outputs = dict()
        for slot_kind, timings_node in timings_gen.items():
            # Skip slots
            if slot_kind == supervisor_slot or slot_kind in skip_slots:
                continue

            if INTERMEDIATE_UPDATE:
                new_nodes = dict()
                new_outputs = dict()
            kind = timings.slots[slot_kind].kind  # Node kind to run
            pred = timings_gen[slot_kind].run  # Predicate for running node step

            # Prepare old states
            _old_ss = graph_state.nodes[kind]
            _old_output = graph_state.buffer[kind]

            # Add dummy inputs to old step_state (else jax complains about structural mismatch)
            if _old_ss.inputs is None:
                raise DeprecationWarning("Inputs should not be None, but pre-filled via graph.init")
                # _old_ss = update_input_fns[kind](graph_state, timings_node)

            # Run node step
            no_op = lambda *args: (_old_ss, _old_output)
            # no_op = jax.checkpoint(no_op) # todo: apply jax.checkpoint to no_op?
            try:
                new_ss, new_output = jax.lax.cond(pred, node_step_fns[kind], no_op, graph_state, timings_node)
            except TypeError as e:
                new_ss, new_output = node_step_fns[kind](graph_state, timings_node)
                print(f"TypeError: kind={kind}")
                raise e

            # Store new state
            new_nodes[kind] = new_ss
            new_outputs[kind] = new_output

            # Update buffer
            if INTERMEDIATE_UPDATE:
                graph_state = graph_state.replace_buffer(new_outputs)
                # new_buffer = graph_state.buffer.replace(outputs=graph_state.buffer.outputs.copy(new_outputs))
                graph_state = graph_state.replace(nodes=graph_state.nodes.copy(new_nodes))

        if INTERMEDIATE_UPDATE:
            new_graph_state = graph_state
        else:
            graph_state = graph_state.replace_buffer(new_outputs)
            # new_buffer = graph_state.buffer.replace(outputs=graph_state.buffer.outputs.copy(new_outputs))
            new_graph_state = graph_state.replace(nodes=graph_state.nodes.copy(new_nodes))
        return new_graph_state, new_graph_state

    def _run_S(graph_state: GraphState) -> GraphState:
        # Get eps & step  (used to index timings)
        graph_state = graph_state.replace_step(timings, step=graph_state.step)  # Make sure step is clipped to max_step size
        step = graph_state.step

        # Determine slice sizes (depends on window size)
        # [1:] because first dimension is step.
        timings_eps = graph_state.timings_eps
        slice_sizes = jax.tree_map(lambda _tb: list(_tb.shape[1:]), timings_eps)

        # Slice timings
        timings_mcs = jax.tree_map(
            lambda _tb, _size: jax.lax.dynamic_slice(_tb, [step] + [0 * s for s in _size], [1] + _size)[0],
            timings_eps,
            slice_sizes,
        )
        timings_mcs = timings_mcs.to_generation()

        # Run generations
        # NOTE! len(generations) = len(timings_mcs) --> last generation is the supervisor.
        if not is_uniform:
            for gen, timings_gen in zip(generations[:-1], timings_mcs):
                assert all([node in gen for node in timings_gen.keys()]), f"Missing nodes in timings: {gen}"
                graph_state, _ = _run_generation(graph_state, timings_gen)
        else:
            # raise NotImplementedError("Uniform generations not yet validated")
            flattened_timings = dict()
            # NOTE! This assumes that the supervisor is the only node in the last generation.
            [
                flattened_timings.update(timings_gen) for timings_gen in timings_mcs[:-1]
            ]  # Remember: this does include supervisor_slot
            slots_timings = {}
            for kind, slots in kinds_to_slots.items():  # Remember: kinds_to_slots does not include supervisor_slot
                timings_to_stack = [flattened_timings[slot].replace(generation=None) for slot in slots]
                slots_timings[slots[0]] = jax.tree_util.tree_map(lambda *args: jnp.stack(args, axis=0), *timings_to_stack)
            all_shapes = [v.run.shape for k, v in slots_timings.items()]
            assert all([s == all_shapes[0] for s in all_shapes]), "Shapes of slots are not equal"
            graph_state, _ = jax.lax.scan(_run_generation, graph_state, slots_timings)

        # Run supervisor input update
        new_ss_supervisor = update_input_fns[supervisor](graph_state, timings_mcs[supervisor_gen_idx][supervisor_slot])
        graph_state = graph_state.replace(nodes=graph_state.nodes.copy({supervisor: new_ss_supervisor}))

        # Increment step (new step may exceed max_step) --> clipping is done at the start of run_S.
        graph_state = graph_state.replace(step=graph_state.step + 1)
        return graph_state

    return _run_S
