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

import rex.sysid.evo as evo
import rex.sysid.transform as transform

if __name__ == "__main__":
    # todo: limit to theta, and initialize at y=0, phi=0, with phi_ref=0.
    # todo: PID Attitude controller
    # todo: stop gradients, necessary?
    # todo: Save best policy
    # todo: Optionally clip value function
    # todo: Properly handle truncated episodes (record terminal observation)
    # todo: Create sweep script
    # todo: Create eval_fn in PPO (e.g., for wandb logging)
    # todo: Train reference tracking policy with zref tracking

    # Create nodes
    world = cf.nodes.OdeWorld(name="world", rate=50, color="grape", order=4)
    mocap = cf.nodes.MoCap(name="mocap", rate=50, color="pink", order=1)
    agent = cf.nodes.PPOAgent(name="agent", rate=50, color="teal", order=2)
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

    # rng = jax.random.PRNGKey(0)
    # gs = graph.init(rng, order=("sentinel", "world"))
    # gs, ss = graph.reset(gs)
    # with jax.disable_jit(True):
    #     for _ in range(10):
    #         pos = ss.inputs["mocap"][-1].data.pos
    #         z = pos[-1]
    #         action = jnp.zeros(env.action_space(gs).shape)
    #         z_ref, theta_ref = 0.0, 0.0
    #         error = z_ref - z
    #         print(f"z_ref: {z_ref}, z: {z}, error: {error}, action: {action}, pos: {onp.array(pos)}")
    #         output = cf.nodes.AgentOutput(action=action)
    #         gs, ss = graph.step(gs, step_state=ss, output=output)

    # Evo hyperparameters
    RNG = jax.random.PRNGKey(0)
    num_steps = 100
    num_train_rollouts = 50
    num_generations = 200
    strategy = "CMA_ES"
    fitness_kwargs = dict(maximize=False, centered_rank=True, z_score=False, w_decay=0.0)
    strategies = {
        "OpenES": dict(popsize=200, use_antithetic_sampling=False, opt_name="adam",
                       lrate_init=0.125, lrate_decay=0.999, lrate_limit=0.001,
                       sigma_init=0.05, sigma_decay=0.999, sigma_limit=0.01, mean_decay=0.0),
        "CMA_ES": dict(popsize=200, elite_ratio=0.1, sigma_init=0.5, mean_decay=0.),
    }
    u_min = dict(kp=0.0, ki=0.0, kd=0.0, max_integral=0.0)
    u_init = dict(kp=0.1, ki=0.01, kd=0.01, max_integral=0.15)
    u_max = dict(kp=10.0, ki=10.0, kd=10.0, max_integral=10.0)
    denorm = transform.Denormalize.init(u_min, u_max)

    def _rollout(_params, _rng: jax.Array = None):
        if _rng is None:
            _rng = jax.random.PRNGKey(0)

        rng_init, rng_mass, rng_pos = jax.random.split(_rng, 3)
        gs = graph.init(rng_init, order=("sentinel", "world"))
        # Update mass
        ss_world = gs.try_get_node("world")
        c = jax.random.uniform(rng_mass, (), minval=-0.15, maxval=0.15)  # Mass perturbation
        new_params = ss_world.params.replace(mass=ss_world.params.mass * (1 + c))
        z = jax.random.uniform(rng_pos, (), minval=-0., maxval=1.0)  # Initial z-position perturbation
        new_pos = ss_world.state.pos.at[-1].set(z)  # Replace z-position
        # new_pos = ss_world.state.pos.at[-1].set(0.)  # Replace z-position
        new_state = ss_world.state.replace(pos=new_pos)
        new_ss_world = ss_world.replace(params=new_params, state=new_state)
        # Update PID params
        ss_pid = gs.try_get_node("attitude")
        new_params = ss_pid.params.replace(kp=_params["kp"], ki=_params["ki"], kd=_params["kd"], max_integral=_params["max_integral"])
        new_ss_pid = ss_pid.replace(params=new_params)

        # Update graphstate &
        init_gs = gs.replace_nodes({"attitude": new_ss_pid, "world": new_ss_world})
        carry = graph.reset(init_gs)

        # Rollout with scan
        def _scan(_carry, _):
            _gs, _ss = _carry
            action = jnp.zeros(env.action_space(init_gs).shape, dtype=float)
            output = cf.nodes.AgentOutput(action=action)
            _gs, _ss = graph.step(_gs, step_state=_ss, output=output)
            return (_gs, _ss), _gs

        _, _rollout = jax.lax.scan(_scan, carry, jnp.arange(num_steps))
        ode_state = _rollout.nodes["mocap"].inputs["world"][:, -1].data
        z = ode_state.pos[:, -1]
        # costs = (z) ** 2
        costs = jnp.abs(z)
        return _rollout, costs, z

    def _loss(opt_params, args, _rng: jax.Array = None):
        if _rng is None:
            _rng = jax.random.PRNGKey(0)
        t = args[0]
        opt_params = t.apply(opt_params)
        _rngs = jax.random.split(_rng, num_train_rollouts)
        _, costs, z = jax.vmap(_rollout, in_axes=(None, 0))(opt_params, _rngs)
        total_cost = jnp.sum(costs, axis=-1)# + 100*(z[:, -1] -1) ** 2
        return total_cost.mean()

    # Create solver
    solver = evo.EvoSolver.init(denorm.normalize(u_min), denorm.normalize(u_max), strategy, strategies[strategy], fitness_kwargs)
    init_sol_state = solver.init_state(denorm.normalize(u_init))
    args = (denorm,)
    logger = solver.init_logger(num_generations=num_generations)

    def _plot_results(_policy_params, _num_episodes):
        _rollouts, _costs, _z = jax.vmap(_rollout, in_axes=(None, 0))(_policy_params, jax.random.split(RNG, _num_episodes))
        fig, axes = plt.subplots(4, 1, figsize=(10, 10))
        axes[0].plot(_costs.T, color="orange", label="cost")
        # axes[0].legend()
        axes[0].set_title("costs")
        axes[1].plot(_z.T, color="orange", label="z")
        axes[1].set_title("z")
        # axes[1].plot(_states.x.T, color="orange", label="x")
        # axes[1].plot(_states.x_ref.T, linestyle="--", color="orange", label="x_ref")
        # axes[1].plot(_states.xdot.T, color="blue", label="xdot")
        # # axes[1].legend()
        # axes[1].set_title("x")
        # axes[2].plot(_states.z.T, color="orange", label="z")
        # axes[2].plot(_states.z_ref.T, linestyle="--", color="orange", label="z_ref")
        # axes[2].plot(_states.zdot.T, color="blue", label="zdot")
        # # axes[2].legend()
        # axes[2].set_title("z")
        # axes[3].plot(_states.theta.T, color="orange", label="theta")
        # axes[3].plot(_states.theta_ref.T, linestyle="--", color="orange", label="theta_ref")
        # # axes[3].legend()
        # axes[3].set_title("theta")
        return fig, axes

    # Solve
    sol_state, log_state, losses = evo.evo(_loss, solver, init_sol_state, args,
                                           max_steps=num_generations, rng=jax.random.PRNGKey(1), verbose=True, logger=logger)
    best_evo = solver.unflatten(sol_state.best_member)
    best_policy_params = denorm.apply(best_evo)
    _plot_results(best_policy_params, 10)
    log_state.plot("Policy Learning")
    print("denorm:", best_policy_params)
    plt.show()
    exit()
    rng = jax.random.PRNGKey(0)
    gs = graph.init(rng, order=("sentinel", "world"))
    rollout = graph.rollout(gs)

    # Initialize PPO config
    config = ppo.Config(
        LR=5e-5,
        NUM_ENVS=64,
        NUM_STEPS=32,  # increased from 16 to 32 (to solve approx_kl divergence)
        TOTAL_TIMESTEPS=20e6,
        UPDATE_EPOCHS=4,
        NUM_MINIBATCHES=4,
        GAMMA=0.99,
        GAE_LAMBDA=0.95,
        CLIP_EPS=0.2,
        ENT_COEF=0.01,
        VF_COEF=0.5,
        MAX_GRAD_NORM=0.5,  # or 0.5?
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

    att_ctrl = rollout.nodes["world"].inputs["attitude"][:, -1].data
    pwm = att_ctrl.pwm
    phi_ref = att_ctrl.phi_ref
    theta_ref = att_ctrl.theta_ref
    psi_ref = att_ctrl.psi_ref

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].plot(actions[:, 0], label="pwm (norm)") if actions.shape[1] > 0 else None
    axes[0].plot(actions[:, 1], label="roll (norm)") if actions.shape[1] > 1 else None
    axes[0].plot(actions[:, 2], label="pitch (norm)") if actions.shape[1] > 2 else None
    axes[0].plot(actions[:, 3], label="yaw (norm)") if actions.shape[1] > 3 else None
    axes[0].legend()
    axes[0].set_title("Actions")
    axes[1].plot(att[:, 0], label="roll")
    axes[1].plot(att[:, 1], label="pitch")
    axes[1].plot(att[:, 2], label="yaw")
    axes[1].legend()
    axes[1].set_title("Attitude")
    axes[2].plot(phi_ref, label="phi_ref")
    axes[2].plot(theta_ref, label="theta_ref")
    axes[2].plot(psi_ref, label="psi_ref")
    axes[2].legend()
    axes[2].set_title("Attitude ref")

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





