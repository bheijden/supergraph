from typing import Any
import pickle
import tqdm
import functools
import jax
import jax.numpy as jnp
import numpy as onp
from flax import struct

import networkx as nx
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
from tensorflow_probability.substrates import jax as tfp  # Import tensorflow_probability with jax backend

tfd = tfp.distributions

import supergraph
from supergraph.evaluate import timer
import supergraph.compiler.utils as utils
import supergraph.compiler.crazyflie as cf
import supergraph.compiler.artificial as art
import supergraph.compiler.ppo as ppo
import supergraph.compiler.base as base
from supergraph.compiler.graph import Graph


def _plot_results(rollout):
    done = rollout.done.reshape(-1)
    ode_state = rollout.next_gs.nodes["mocap"].inputs["world"][:, -1].data
    ctrl = rollout.next_gs.nodes["world"].inputs["attitude"][:, -1].data
    ctrl = jax.tree_util.tree_map(lambda x: onp.where(done, onp.nan, onp.array(x)), ctrl)
    att = onp.where(done[:, None], onp.nan, onp.array(ode_state.att))
    pos = onp.where(done[:, None], onp.nan, onp.array(ode_state.pos))
    vel = onp.where(done[:, None], onp.nan, onp.array(ode_state.vel))

    fig, axes = plt.subplots(2, 4, figsize=(15, 10))
    axes[0, 0].plot(ctrl.pwm_ref, label="pwm", color="blue") if ctrl.pwm_ref is not None else None
    # axes[0, 0].plot(ctrl.z_ref, label="z_ref", color="green", linestyle="--") if ctrl.z_ref is not None else None
    axes[0, 0].legend()
    axes[0, 0].set_title("PWM")
    axes[0, 1].plot(att[:, 0], label="phi", color="orange")
    axes[0, 1].plot(ctrl.phi_ref, label="phi_ref", color="orange", linestyle="--")
    axes[0, 1].plot(att[:, 1], label="theta", color="blue")
    axes[0, 1].plot(ctrl.theta_ref, label="theta_ref", color="blue", linestyle="--")
    # axes[0, 1].plot(att[:, 2], label="psi", color="green")
    # axes[0, 1].plot(ctrl.psi_ref, label="psi_ref", color="green", linestyle="--")
    axes[0, 1].legend()
    axes[0, 1].set_title("Attitude")
    axes[0, 2].plot(pos[:, 0], label="x", color="blue")
    axes[0, 2].plot(pos[:, 1], label="y", color="orange")
    axes[0, 2].plot(pos[:, 2], label="z", color="green")
    axes[0, 2].plot(ctrl.z_ref, label="z_ref", color="green", linestyle="--") if ctrl.z_ref is not None else None
    axes[0, 2].legend()
    axes[0, 2].set_title("Position")
    axes[0, 3].plot(vel[:, 0], label="vx", color="blue")
    axes[0, 3].plot(vel[:, 1], label="vy", color="orange")
    axes[0, 3].plot(vel[:, 2], label="vz", color="green")
    axes[0, 3].legend()
    axes[0, 3].set_title("Velocity")

    # todo: DEBUG
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[1, 0].set_title("PWM")
    axes[1, 0].plot(ctrl.pwm_ref, label="pwm")
    axes[1, 0].plot(ctrl.pwm_unclipped, label="unclipped")
    axes[1, 0].plot(ctrl.pwm_hover, label="hover")
    axes[1, 0].legend()

    axes[1, 1].set_title("Force")
    axes[1, 1].plot(ctrl.force, label="force")
    axes[1, 1].plot(ctrl.z_force, label="z")
    axes[1, 1].plot(ctrl.force_hover, label="hover")
    # axes[1, 1].plot(ctrl.z_force_from_hover, label="z_from_hover")
    axes[1, 1].legend()

    axes[1, 2].set_title("PID")
    axes[1, 2].plot(ctrl.proportional, label="P")
    axes[1, 2].plot(ctrl.integral_unclipped, label="I (unclipped)")
    axes[1, 2].plot(ctrl.integral, label="I")
    axes[1, 2].plot(ctrl.derivative, label="D")
    axes[1, 2].legend()
    return fig, axes


if __name__ == "__main__":
    # Create nodes
    world = cf.nodes.OdeWorld(name="world", rate=50, color="grape", order=4)
    mocap = cf.nodes.MoCap(name="mocap", rate=50, color="pink", order=1)
    agent = cf.nodes.PPOAgent(name="agent", rate=50, color="teal", order=2)
    attitude = cf.nodes.ZPID(name="attitude", rate=50, color="orange", order=3)
    # attitude = cf.nodes.PID(name="attitude", rate=50, color="orange", order=3)
    # attitude = cf.nodes.AttitudeController(name="attitude", rate=50, color="orange", order=3)
    sentinel = cf.nodes.Sentinel(name="sentinel", rate=1, color="blue", order=0)
    nodes = dict(sentinel=sentinel, world=world, mocap=mocap, agent=agent, attitude=attitude)

    # Connect nodes
    world.connect(attitude, window=1, name="attitude")
    mocap.connect(world, window=1, name="world")
    attitude.connect(agent, window=1, name="agent")
    attitude.connect(mocap, window=1, name="mocap")
    agent.connect(mocap, window=1, name="mocap")

    # # Test API
    # step_states = dict()
    # for n in nodes.values():
    #     ss = n.init_step_state()
    #     next_ss, o = n.step(ss)
    #     n.step(next_ss)
    #     step_states[n.name] = ss

    # Define phase and delays
    phase = dict(mocap=tfd.Deterministic(loc=1e-4),
                 agent=tfd.Deterministic(loc=2e-4),
                 attitude=tfd.Deterministic(loc=3e-4),
                 sentinel=tfd.Deterministic(loc=4e-4),
                 world=tfd.Deterministic(loc=5e-4),
                 )
    computation_delays = dict(world=tfd.Deterministic(loc=0.),
                              mocap=tfd.Deterministic(loc=0.),
                              agent=tfd.Deterministic(loc=0.),
                              attitude=tfd.Deterministic(loc=0.),
                              sentinel=tfd.Deterministic(loc=0.))
    communication_delays = dict()
    for n in [world, mocap, agent, attitude]:
        for c in n.outputs.values():
            communication_delays[(c.output_node.name, c.input_node.name)] = tfd.Deterministic(loc=0.)

    # Artificially generate graphs
    with timer("generate_graphs"):  # Measure
        graphs_raw = art.generate_graphs(nodes, computation_delays, communication_delays, t_final=6.0, phase=phase, num_episodes=1)

    # Compile with supergraph
    with timer("Graph"):
        graph = Graph(nodes, agent, graphs_raw, debug=True, supergraph_mode="MCS")
        graph.run = jax.jit(graph.run)

    # Visualize raw graphs
    MAKE_PLOTS = False
    if MAKE_PLOTS:  # Visualize "raw" graph
        G = utils.to_networkx_graph(graphs_raw[0], nodes=nodes)
        supergraph.plot_graph(G, max_x=1.0)
        plt.show()

    # Visualize windowed graphs
    if MAKE_PLOTS:
        supergraph.plot_graph(graph.Gs[0], max_x=1.0)
        # plt.show()

    # Visualize supergraph
    if MAKE_PLOTS:
        supergraph.plot_graph(graph.S)
        # plt.show()

    # Load agent params

    with open("./main_crazyflie_ss_agent.pkl", "rb") as f:
        ss_agent = pickle.load(f)
    with open("./main_crazyflie_ss_attitude.pkl", "rb") as f:
        ss_att = pickle.load(f)
    with open("./main_crazyflie_ss_sentinel.pkl", "rb") as f:
        ss_sentinel = pickle.load(f)

    # Re-tune PID
    # new_params = ss_att.params.replace(kp=0.25, ki=0.25, kd=0.1)
    # ss_att = ss_att.replace(params=new_params)

    # Define step_states
    step_states = {
        "agent": ss_agent,
        "attitude": ss_att,
        # "sentinel": ss_sentinel,
    }

    # RL environment
    from supergraph.compiler.rl import AutoResetWrapper
    # env = cf.nodes.ReferenceTracking(graph, order=("sentinel", "world"), randomize_eps=True,
    #                                  step_states=step_states)
    env = cf.nodes.InclinedLanding(graph, order=("sentinel", "world"), randomize_eps=True,
                                   step_states=step_states)
    env = AutoResetWrapper(env, fixed_init=False)

    # Rollout
    _rollout = functools.partial(cf.nodes.rollout, env)
    rollout_vjit = jax.jit(jax.vmap(_rollout, in_axes=0))
    rng = jax.random.PRNGKey(1)
    rngs = jax.random.split(rng, num=2)
    vrollout = rollout_vjit(rngs)
    rollout = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), vrollout)
    _plot_results(rollout)
    print("Rollout done!")

    # Html visualization may not work properly, if it's already rendering somewhere else.
    # In such cases, comment-out all but one HTML(pendulum.render(rollout))
    rollout_json = cf.render(rollout.next_gs)
    cf.save("./main_crazyflie.html", rollout_json)
    plt.show()

