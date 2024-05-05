import pickle
import tqdm
import functools
import jax
import jax.numpy as jnp
import numpy as onp

import networkx as nx
import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    # todo: stream vicon quaternions?
    # todo: Initialize from same height as platform.
    # todo: Incentivize approaching from above.
    # todo: Improve RL performance in simulation
    # todo: increase agent rate to 50 (or 80 in real-world?)
    # todo: Make ZPID controller more aggressive (increase P, I)
    # todo: account for platform z-offset in combination of get_observation() and get_action().
    # todo: avoid using hardcoded yaw=0. in the PPOAgentParams.get_observation() method
    # todo: check intrinsic rpy conventions in supergraph.compiler.crazyflie.nodes:
    #       - spherical_to_R --> Check order Ry * Rz or reversed?
    # todo: real experiments checklist
    #       - Train with variations in yaw --> somehow, azimuth is also rolling the platform...
    #       - Check tracking offset Vicon vs. simulated Mocap
    #           - Simulation: [0,0,0] == perfect landing --> pos_offset=[0,0,0]
    #           - Real: [0,0,0.0193] == real landing --> pos_offset=[0,0,-0.0193]
    #       - Add yaw controller that fixes yaw to zero in the platform frame.
    # todo: Truncation in reference tracking policy
    # todo: Remove DebugAttitudeControllerOutput
    # todo: Save best policy

    # Create nodes
    world = cf.nodes.OdeWorld(name="world", rate=50, color="grape", order=4)
    mocap = cf.nodes.MoCap(name="mocap", rate=50, color="pink", order=1)
    agent = cf.nodes.PPOAgent(name="agent", rate=25, color="teal", order=2)
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

    # Grab sentinel
    params_sentinel = sentinel.init_params()

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
                              agent=tfd.Deterministic(loc=0.02),
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
        # plt.show()

    # Visualize windowed graphs
    if MAKE_PLOTS:
        supergraph.plot_graph(graph.Gs[0], max_x=1.0)
        # plt.show()

    # Visualize supergraph
    if MAKE_PLOTS:
        supergraph.plot_graph(graph.S)
        # plt.show()
    # plt.show()

    # RL environment
    env = cf.nodes.InclinedLanding(graph, order=("sentinel", "world"), randomize_eps=True)
    # env = cf.nodes.ReferenceTracking(graph, order=("sentinel", "world"), randomize_eps=True)
    # env = cf.nodes.ReferenceTrackingTerminate(graph, order=("sentinel", "world"), randomize_eps=True)
    # gs, obs, info = env.reset(jax.random.PRNGKey(0))
    # obs_space = env.observation_space(gs)
    # act_space = env.action_space(gs)
    # print(f"obs_space: {obs_space.low.shape}, act_space: {act_space.low.shape}")
    # print(f"obs: {obs}, info: {info}")
    # gs, obs, reward, terminated, truncated, info = env.step(gs, jnp.zeros(act_space.shape))
    # print(f"obs: {obs}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, info: {info}")

    # config = cf.ppo.ref_tracking
    # config = cf.ppo.term_ref_tracking
    config = cf.ppo.multi_inclination
    train = functools.partial(ppo.train, env)
    train = jax.jit(train)
    with timer("train"):
        res = train(config, jax.random.PRNGKey(2))
    print("Training done!")

    # Initialize agent params
    model_params = res["runner_state"][0].params["params"]
    ppo_params = cf.nodes.PPOAgentParams(res["act_scaling"], res["norm_obs"], model_params,
                                         action_dim=res["act_scaling"].low.shape[0],
                                         mapping=params_sentinel.ctrl_mapping,
                                         hidden_activation=config.HIDDEN_ACTIVATION, stochastic=False)
    ss_agent = base.StepState(rng=None, params=ppo_params, state=None)

    # Save agent params
    ss_agent_onp = jax.tree_util.tree_map(lambda x: onp.array(x), ss_agent)
    with open("main_crazyflie_ss_agent.pkl", "wb") as f:
        pickle.dump(ss_agent_onp, f)
    print("Agent params saved!")

    # Save PID params
    gs = graph.init(jax.random.PRNGKey(0), order=("sentinel", "world"), step_states=dict(agent=ss_agent))
    ss_att_onp = jax.tree_util.tree_map(lambda x: onp.array(x), gs.nodes["attitude"])
    with open("main_crazyflie_ss_attitude.pkl", "wb") as f:
        pickle.dump(ss_att_onp, f)
    print("Attitude params saved!")

    # Save sentinel params
    ss_sentinel = jax.tree_util.tree_map(lambda x: onp.array(x), gs.nodes["sentinel"])
    with open("main_crazyflie_ss_sentinel.pkl", "wb") as f:
        pickle.dump(ss_sentinel, f)
    print("Sentinel params saved!")

    # Initialize
    rng = jax.random.PRNGKey(0)
    gs = graph.init(rng, order=("sentinel", "world"), step_states=dict(agent=ss_agent))
    rollout = graph.rollout(gs)
    print("Rollout done!")

    actions = rollout.nodes["attitude"].inputs["agent"][:, -1].data.action
    ode_state = rollout.nodes["mocap"].inputs["world"][:, -1].data
    att = ode_state.att
    pos = ode_state.pos

    # Save attitude
    with open("main_crazyflie_att.pkl", "wb") as f:
        pickle.dump(onp.array(att), f)
    print("Attitude saved!")
    with open("main_crazyflie_pos.pkl", "wb") as f:
        pickle.dump(onp.array(pos), f)
    print("Position saved!")

    ctrl = rollout.nodes["world"].inputs["attitude"][:, -1].data

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(ctrl.pwm_ref, label="pwm", color="blue", linestyle="--") if ctrl.pwm_ref is not None else None
    # axes[0].plot(ctrl.z_ref, label="z_ref", color="green", linestyle="--") if ctrl.z_ref is not None else None
    axes[0].legend()
    axes[0].set_title("PWM")
    axes[1].plot(att[:, 0], label="phi", color="orange")
    axes[1].plot(ctrl.phi_ref, label="phi_ref", color="orange", linestyle="--")
    axes[1].plot(att[:, 1], label="theta", color="blue")
    axes[1].plot(ctrl.theta_ref, label="theta_ref", color="blue", linestyle="--")
    # axes[1].plot(att[:, 2], label="psi", color="green")
    # axes[1].plot(ctrl.psi_ref, label="psi_ref", color="green", linestyle="--")
    axes[1].legend()
    axes[1].set_title("Attitude")
    axes[2].plot(pos[:, 0], label="x", color="blue")
    axes[2].plot(pos[:, 1], label="y", color="orange")
    axes[2].plot(pos[:, 2], label="z", color="green")
    axes[2].plot(ctrl.z_ref, label="z_ref", color="green", linestyle="--") if ctrl.z_ref is not None else None
    axes[2].legend()
    axes[2].set_title("Position")

    # Html visualization may not work properly, if it's already rendering somewhere else.
    # In such cases, comment-out all but one HTML(pendulum.render(rollout))
    rollout_json = cf.render(rollout)
    cf.save("./main_crazyflie.html", rollout_json)
    plt.show()
    exit()

    # Rollout
    rollout = [gs]
    graph_run = jax.jit(graph.run)(gs)
    pbar = tqdm.tqdm(range(graph.max_steps))
    for _ in pbar:
        gs = graph.run(gs)  # Run the graph (incl. the agent's step() method)
        rollout.append(gs)
        # We can access the agent's state directly (this is the state *after* the step() method was called)
        ss = gs.nodes["agent"]
        # Print the current time, sensor reading, and action
        pbar.set_postfix_str(f"step: {ss.seq}, ts_start: {ss.ts:.2f}")
    pbar.close()

    # Html visualization may not work properly, if it's already rendering somewhere else.
    # In such cases, comment-out all but one HTML(pendulum.render(rollout))
    rollout_json = cf.render(rollout)
    cf.save("./main_crazyflie.html", rollout_json)
    exit()

    # Initialize supergraph
    rng = jax.random.PRNGKey(0)
    gs = graph.init(rng, order=("sentinel", "world"))

    # However, now we will repeatedly call the run() method, which will call the step() method of the agent node.
    # In our case, the agent node is a random agent, so it will also generate random actions.
    rollout = [gs]
    pbar = tqdm.tqdm(range(graph.max_steps))
    for _ in pbar:
        gs = graph.run(gs)  # Run the graph (incl. the agent's step() method)
        rollout.append(gs)
        # We can access the agent's state directly (this is the state *after* the step() method was called)
        ss = gs.nodes["agent"]
        # Print the current time, sensor reading, and action
        pbar.set_postfix_str(f"step: {ss.seq}, ts_start: {ss.ts:.2f}")
    pbar.close()

    # Html visualization may not work properly, if it's already rendering somewhere else.
    # In such cases, comment-out all but one HTML(pendulum.render(rollout))
    rollout_json = cf.render(rollout)
    cf.save("./main_crazyflie.html", rollout_json)
    exit()

    # Test vmap
    rngs = jax.random.split(rng, num=10)
    graph_init = functools.partial(graph.init, randomize_eps=True, order=("world",))
    graph_init_jv = jax.jit(jax.vmap(graph_init, in_axes=0))
    with timer("graph_init_jv[jit]"):
        _ = graph_init_jv(rngs)
    with timer("graph_init_jv", repeat=10):
        for _ in range(10):
            gs = graph_init_jv(rngs)

    graph_rollout_jv = jax.jit(jax.vmap(graph.rollout, in_axes=0))
    with timer("graph_rollout_jv[jit]"):
        gs_rollout = graph_rollout_jv(gs, eps=gs.eps)

    with timer("graph_rollout_jv", repeat=10):
        for _ in range(10):
            gs_rollout = graph_rollout_jv(gs, eps=gs.eps)

    if True:
        eps_gs_rollout = jax.tree_util.tree_map(lambda x: x[0], gs_rollout)
        rollout_json = pendulum_nodes.render(eps_gs_rollout)
        pendulum_nodes.save("./main_compiler.html", rollout_json)
        # from IPython.display import HTML
        # HTML(rollout_json)

    if True:
        action = gs_rollout.nodes["attitude"].inputs["agent"].data.action[:, :, 0, 0].T
        plt.plot(action)
    plt.show()
