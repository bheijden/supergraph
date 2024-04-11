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
    # todo: inclined landing reward:
    #       - Add state (1/0) that denotes whether we reached pre-landing pose
    #       - time-penalty -1
    #       - If not pre-landing:
    #           - Fly to xyz=(1.0, 0., 0.), att==(0,0,0)?
    #           - Euclidean cost
    #           - Penalize below height while on opposite side of platform
    #           - pre-landing=True
    #       - If pre-landing:
    #           - penalize z-ref when far away.
    #           - Euclidean cost
    #           - If outside bounds:
    #               - Add remaining time left
    #               - Add infinite horizon cost of state
    # todo: real experiments checklist
    #       - invite jelle for supergraph.
    #       - verify policy's rpy conventions
    #       - make minimal eagerx implementation for policy evaluation
    #       - remove angular velocity from observation
    #       - Subtract (fictitious) platform location from cf pos & att
    #       - PID or PWM control?
    # todo: Remove DebugAttitudeControllerOutput
    # todo: theta should affect the x-axis, phi should affect the y-axis
    # todo: Save best policy
    # todo: Optionally clip value function
    # todo: Properly handle truncated episodes (record terminal observation)
    # todo: Create eval_fn in PPO (e.g., for wandb logging)

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

    # RL environment
    env = cf.nodes.Environment(graph, order=("sentinel", "world"), randomize_eps=True)
    # gs, obs, info = env.reset(jax.random.PRNGKey(0))
    # obs_space = env.observation_space(gs)
    # act_space = env.action_space(gs)
    # print(f"obs_space: [{obs_space.low}, {obs_space.high}], act_space: [{act_space.low}, {act_space.high}]")
    # print(f"obs: {obs}, info: {info}")
    # gs, obs, reward, terminated, truncated, info = env.step(gs, jnp.zeros(act_space.shape))
    # print(f"obs: {obs}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, info: {info}")

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
    # plt.show()

    # Initialize PPO config
    config = ppo.Config(
        LR=1e-4,
        NUM_ENVS=64,
        NUM_STEPS=128,  # increased from 16 to 32 (to solve approx_kl divergence)
        TOTAL_TIMESTEPS=5e6,
        UPDATE_EPOCHS=8,
        NUM_MINIBATCHES=8,
        GAMMA=0.90,
        GAE_LAMBDA=0.983,
        CLIP_EPS=0.93,
        ENT_COEF=0.03,
        VF_COEF=0.58,
        MAX_GRAD_NORM=0.44,  # or 0.5?
        NUM_HIDDEN_LAYERS=2,
        NUM_HIDDEN_UNITS=64,
        KERNEL_INIT_TYPE="xavier_uniform",
        HIDDEN_ACTIVATION="tanh",
        STATE_INDEPENDENT_STD=True,
        SQUASH=True,
        ANNEAL_LR=False,
        NORMALIZE_ENV=True,
        DEBUG=False,
        VERBOSE=True,
        FIXED_INIT=True,
        OFFSET_STEP=True,
        NUM_EVAL_ENVS=20,
        EVAL_FREQ=20,
    )
    train = functools.partial(ppo.train, env)
    train = jax.jit(train)
    res = train(config, jax.random.PRNGKey(0))
    print("Training done!")

    # Initialize agent params
    model_params = res["runner_state"][0].params["params"]
    act_scaling = res["act_scaling"]
    obs_scaling = res["norm_obs"]
    ppo_params = cf.nodes.PPOAgentParams(act_scaling, obs_scaling, model_params,
                                         hidden_activation=config.HIDDEN_ACTIVATION, stochastic=False)
    ss_agent = base.StepState(rng=None, params=ppo_params, state=None)

    # Save agent params
    ss_agent_onp = jax.tree_util.tree_map(lambda x: onp.array(x), ss_agent)
    with open("main_crazyflie_ss_agent.pkl", "wb") as f:
        pickle.dump(ss_agent_onp, f)
    print("Agent params saved!")

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
    axes[1].plot(att[:, 0], label="theta", color="blue")
    axes[1].plot(ctrl.theta_ref, label="theta_ref", color="blue", linestyle="--")
    axes[1].plot(att[:, 1], label="phi", color="orange")
    axes[1].plot(ctrl.phi_ref, label="phi_ref", color="orange", linestyle="--")
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
    plt.show()
    rollout_json = cf.render(rollout)
    cf.save("./main_crazyflie.html", rollout_json)
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





    # config = ppo.Config(
    #     LR=5e-5,
    #     NUM_ENVS=64,
    #     NUM_STEPS=32,  # increased from 16 to 32 (to solve approx_kl divergence)
    #     TOTAL_TIMESTEPS=30e6,
    #     UPDATE_EPOCHS=4,
    #     NUM_MINIBATCHES=4,
    #     GAMMA=0.99,
    #     GAE_LAMBDA=0.95,
    #     CLIP_EPS=0.2,
    #     ENT_COEF=0.01,
    #     VF_COEF=0.5,
    #     MAX_GRAD_NORM=0.5,  # or 0.5?
    #     NUM_HIDDEN_LAYERS=2,
    #     NUM_HIDDEN_UNITS=64,
    #     KERNEL_INIT_TYPE="xavier_uniform",
    #     HIDDEN_ACTIVATION="tanh",
    #     STATE_INDEPENDENT_STD=True,
    #     SQUASH=True,
    #     ANNEAL_LR=False,
    #     NORMALIZE_ENV=True,
    #     DEBUG=False,
    #     VERBOSE=True,
    #     FIXED_INIT=True,
    #     OFFSET_STEP=True,
    #     NUM_EVAL_ENVS=20,
    #     EVAL_FREQ=20,
    # )