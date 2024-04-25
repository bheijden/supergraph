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
    axes[0, 3].plot(vel[:, 1], label="vx", color="orange")
    axes[0, 3].plot(vel[:, 2], label="vx", color="green")
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
    agent = cf.nodes.PPOAgent(name="agent", rate=25, color="teal", order=2)
    attitude = cf.nodes.PID(name="attitude", rate=50, color="orange", order=3)
    # attitude = cf.nodes.AttitudeController(name="attitude", rate=50, color="orange", order=3)
    sentinel = cf.nodes.Sentinel(name="sentinel", rate=1, color="blue", order=0)
    nodes = dict(sentinel=sentinel, world=world, mocap=mocap, agent=agent, attitude=attitude)

    # Connect nodes
    world.connect(attitude, window=1, name="attitude")
    mocap.connect(world, window=1, name="world")
    attitude.connect(agent, window=1, name="agent")
    attitude.connect(mocap, window=1, name="mocap")
    agent.connect(mocap, window=1, name="mocap")

    # Test API
    step_states = dict()
    for n in nodes.values():
        ss = n.init_step_state()
        next_ss, o = n.step(ss)
        n.step(next_ss)
        step_states[n.name] = ss

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
    # with open("./main_crazyflie_ss_agent_rwd_bug.pkl", "rb") as f:
        ss_agent = pickle.load(f)
    with open("./main_crazyflie_ss_attitude.pkl", "rb") as f:
    # with open("./main_crazyflie_ss_attitude_rwd_bug.pkl", "rb") as f:
        ss_att = pickle.load(f)

    # RL environment
    from supergraph.compiler.rl import AutoResetWrapper
    env = cf.nodes.InclinedLanding(graph, order=("sentinel", "world"), randomize_eps=True,
                                   step_states={"agent": ss_agent, "attitude": ss_att},)
    env = AutoResetWrapper(env, fixed_init=False)

    # Rollout
    _rollout = functools.partial(cf.nodes.rollout, env)
    rollout_vjit = jax.jit(jax.vmap(_rollout, in_axes=0))
    rng = jax.random.PRNGKey(1)
    rngs = jax.random.split(rng, num=2)
    with timer("Rollout"):
        vrollout = rollout_vjit(rngs)
    rollout = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), vrollout)

    def get_reward(graph_state, action):
        # Get denormalized action
        p_att = env.get_step_state(graph_state, "attitude").params
        output = p_att.to_output(action)
        z_ref = output.z_ref
        theta_ref = output.theta_ref
        phi_ref = output.phi_ref
        psi_ref = output.psi_ref
        att_ref = jnp.array([phi_ref, theta_ref, psi_ref])

        # Get current state
        ss = env.get_step_state(graph_state)
        last_mocap = ss.inputs["mocap"][-1].data
        _, theta, _ = last_mocap.att
        inclination = last_mocap.inclination
        vx, vy, vz = last_mocap.vel

        # Calculate components of the landing velocity
        ss_sentinel = env.get_step_state(graph_state, "sentinel")
        vx_land = -ss_sentinel.params.vel_land * jnp.sin(inclination)
        vz_land = -ss_sentinel.params.vel_land * jnp.cos(inclination)
        vel_target = jnp.array([vx_land, 0., vz_land])  # Final velocity target
        pos_target = jnp.array([0., 0., 0.])  # Final position target
        att_target = jnp.array([0., inclination, 0.])  # Final attitude target

        # running cost
        k1, k2, k3, k4 = env._rwd_params.k1, env._rwd_params.k2, env._rwd_params.k3, env._rwd_params.k4
        f1, f2 = env._rwd_params.f1, env._rwd_params.f2
        fp = env._rwd_params.fp
        p = env._rwd_params.p
        pos_error = jnp.linalg.norm(pos_target - last_mocap.pos)
        att_error = jnp.linalg.norm(att_target - last_mocap.att)
        act_att_error = jnp.linalg.norm(att_target - att_ref)
        vyz_error = vy ** 2 + vz ** 2
        vx_error = jnp.clip(vx * theta, None, 0)
        vel_error = jnp.linalg.norm(vel_target - last_mocap.vel)
        act_z_error = z_ref ** 2
        pos_perfect = (pos_error < (p * 1))
        att_perfect = (att_error < (p * 1))
        vel_perfect = (vel_error < (p * 10))
        is_perfect = pos_perfect * att_perfect * vel_perfect
        cost_eps = pos_error + k1 * att_error + k2 * vyz_error + k3 * vx_error + k1 * act_att_error + k4 * act_z_error
        cost_final = pos_error + f1 * att_error + f2 * vel_error
        cost_perfect = -fp * is_perfect

        # Get termination conditions
        gamma = ss_sentinel.params.gamma
        terminated = env.get_terminated(graph_state)
        truncated = env.get_truncated(graph_state)
        done = jnp.logical_or(terminated, truncated)
        cost = cost_eps + done * ((1 - terminated) * (1 / (1 - gamma)) + terminated) * cost_final + done * terminated * cost_perfect

        old = {
            "cost_final": cost_final,
            "cost_perfect": cost_perfect,
            "cost": cost,
            "pos_error": pos_error,
            "att_error": att_error,
            "vyz_error": vyz_error,
            "vx_error": vx_error,
            "vel_error": vel_error,
            "act_att_error": act_att_error,
            "act_z_error": act_z_error,
        }

        old_targets = {
            "vel_land_ref": last_mocap.vel,
            "vel_target": vel_target,
        }

        # Get rotation matrices
        R_cf2w_ref = cf.nodes.rpy_to_R(att_ref)
        R_cf2w = cf.nodes.rpy_to_R(last_mocap.att)
        R_is2w = cf.nodes.rpy_to_R(last_mocap.att_plat)
        z_cf_ref = R_cf2w_ref[:, 2]
        z_cf = R_cf2w[:, 2]
        z_is = R_is2w[:, 2]  # Final attitude target

        # Calculate attitude error
        att_error = jnp.arccos(jnp.dot(z_cf, z_is))  # Minimize angle between two z-axis vectors
        act_att_error = jnp.arccos(jnp.dot(z_cf_ref, z_is))  # Minimize angle between two z-axis vectors

        # Calculate components of the landing velocity
        ss_sentinel = env.get_step_state(graph_state, "sentinel")
        # vel_land = jnp.dot(z_is, last_mocap.vel)*z_is  # Calculate component in direction of z-axis
        vel_land = last_mocap.vel
        vel_land_ref = -ss_sentinel.params.vel_land*z_is   # target is landing velocity in negative direction of platform z-axis
        vel_land_error = jnp.linalg.norm(vel_land_ref-vel_land)
        z_cf_xy = jnp.array([z_cf[0], z_cf[1], 0]) / jnp.linalg.norm(jnp.array([z_cf[0], z_cf[1], 0]))  # Project z-axis to xy-plane
        vel_underact = 0.5*jnp.clip(jnp.dot(z_cf_xy, last_mocap.vel), None, 0)   # Promote underactuated motion (i.e. velocity in negative z-axis)

        # running cost
        k1, k2, k3, k4 = env._rwd_params.k1, env._rwd_params.k2, env._rwd_params.k3, env._rwd_params.k4
        f1, f2 = env._rwd_params.f1, env._rwd_params.f2
        fp = env._rwd_params.fp
        p = env._rwd_params.p
        pos_error = jnp.linalg.norm(pos_target - last_mocap.pos)
        vxyz_error = jnp.linalg.norm(last_mocap.vel - jnp.dot(z_is, last_mocap.vel) * z_is)
        act_z_error = z_ref**2
        pos_perfect = (pos_error < (p * 1))
        att_perfect = (att_error < (p * 1))
        vel_perfect = (vel_land_error < (p * 10))
        is_perfect = pos_perfect * att_perfect * vel_perfect
        cost_eps = pos_error + k1*att_error + k2*vxyz_error + k3*vel_underact + k1*act_att_error + k4*act_z_error
        cost_final = pos_error + f1*att_error + f2*vel_land_error
        cost_perfect = -fp * is_perfect
        cost = cost_eps + done * ((1 - terminated) * (1 / (1 - gamma)) + terminated) * cost_final + done * terminated * cost_perfect

        new ={
            "cost_final": cost_final,
            "cost_perfect": cost_perfect,
            "cost": cost,
            "pos_error": pos_error,
            "att_error": att_error,
            "vyz_error": vxyz_error,
            "vx_error": vel_underact,
            "vel_error": vel_land_error,
            "act_att_error": act_att_error,
            "act_z_error": act_z_error,
        }

        new_targets = {
            "z_is": z_is,
            "vel_land_ref": vel_land_ref,
            "vel_target": vel_land_ref,
        }

        return old, new, old_targets, new_targets

    # Investigate rollout
    # gs = jax.tree_util.tree_map((lambda x: x[0]), rollout.next_gs)
    # reward = env.get_new_reward(gs, actions[0])
    vgs = rollout.next_gs
    actions = rollout.action
    old, new, old_targets, new_targets = jax.vmap(get_reward)(rollout.next_gs, actions)

    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    axes = axes.flatten()
    for i, k in enumerate(old.keys()):
        axes[i].plot(old[k], label=k)
        axes[i].plot(new[k], label=k)
        axes[i].legend()
    plt.show()

    # Plot results
    _plot_results(rollout)
    print("Rollout done!")

    # Html visualization may not work properly, if it's already rendering somewhere else.
    # In such cases, comment-out all but one HTML(pendulum.render(rollout))
    rollout_json = cf.render(rollout.next_gs)
    cf.save("./main_crazyflie.html", rollout_json)
    plt.show()
