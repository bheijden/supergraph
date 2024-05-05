import pickle
import tqdm
import functools
import jax
import jax.numpy as jnp
import numpy as onp

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
    ode_state = rollout.nodes["mocap"].inputs["world"][:, -1].data
    ctrl = rollout.nodes["world"].inputs["attitude"][:, -1].data
    att = ode_state.att
    pos = ode_state.pos

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].plot(ctrl.pwm_ref, label="pwm", color="blue", linestyle="--") if ctrl.pwm_ref is not None else None
    # axes[0, 0].plot(ctrl.z_ref, label="z_ref", color="green", linestyle="--") if ctrl.z_ref is not None else None
    axes[0, 0].legend()
    axes[0, 0].set_title("PWM")
    axes[0, 1].plot(att[:, 0], label="theta", color="blue")
    axes[0, 1].plot(ctrl.theta_ref, label="theta_ref", color="blue", linestyle="--")
    axes[0, 1].plot(att[:, 1], label="phi", color="orange")
    axes[0, 1].plot(ctrl.phi_ref, label="phi_ref", color="orange", linestyle="--")
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
    # todo: why constant offset in z-axis with PID controller?
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

    # # Initialize agent params
    # model_params = res["runner_state"][0].params["params"]
    # act_scaling = res["act_scaling"]
    # obs_scaling = res["norm_obs"]
    # ppo_params = cf.nodes.PPOAgentParams(act_scaling, obs_scaling, model_params, hidden_activation="tanh", stochastic=False)
    # ss_agent = base.StepState(rng=None, params=ppo_params, state=None)
    #
    # # Save agent params
    # ss_agent_onp = jax.tree_util.tree_map(lambda x: onp.array(x), ss_agent)
    # with open("main_crazyflie_ss_agent.pkl", "wb") as f:
    #     pickle.dump(ss_agent_onp, f)
    # print("Agent params saved!")

    # Load agent params
    with open("./main_crazyflie_ss_agent.pkl", "rb") as f:
        ss_agent = pickle.load(f)

    # Rollout
    def _rollout(_params, _rng: jax.Array = None):
        if _rng is None:
            _rng = jax.random.PRNGKey(0)

        rng_init, rng_mass, rng_pos = jax.random.split(_rng, 3)
        gs = graph.init(rng_init, order=("sentinel", "world"), step_states=dict(agent=ss_agent))
        # Get step_states
        ss_world = gs.try_get_node("world")
        ss_pid = gs.try_get_node("attitude")
        # # Update mass
        # c = jax.random.uniform(rng_mass, (), minval=-0.15, maxval=0.15)  # Mass perturbation
        # new_params = ss_world.params.replace(mass=ss_world.params.mass * (1 + c))
        # z = jax.random.uniform(rng_pos, (), minval=-0., maxval=1.0)  # Initial z-position perturbation
        # new_pos = ss_world.state.pos.at[-1].set(z)  # Replace z-position
        # # new_pos = ss_world.state.pos.at[-1].set(0.2)  # Replace z-position
        # new_state = ss_world.state.replace(pos=new_pos)
        # ss_world = ss_world.replace(params=new_params, state=new_state)
        # # Update PID params
        # new_params = ss_pid.params.replace(kp=_params["kp"], ki=_params["ki"], kd=_params["kd"], max_integral=_params["max_integral"])
        # ss_pid = ss_pid.replace(params=new_params)

        # Update graphstate &
        init_gs = gs.replace_nodes({"attitude": ss_pid, "world": ss_world})
        carry = graph.reset(init_gs)

        # Rollout with scan
        def _scan(_carry, _):
            _gs, _ss = _carry
            action = jnp.zeros(env.action_space(init_gs).shape, dtype=float)
            output = cf.nodes.AgentOutput(action=action)
            _gs, _ss = graph.step(_gs, step_state=_ss, output=output)
            # _gs, _ss = graph.step(_gs)
            return (_gs, _ss), _gs

        _, graph_states = jax.lax.scan(_scan, carry, jnp.arange(graph.max_steps))
        # ode_state = graph_states.nodes["mocap"].inputs["world"][:, -1].data
        # z = ode_state.pos[:, -1]
        # costs = (z) ** 2
        # costs = jnp.abs(z)
        return graph_states

    # Rollout
    rollout_jit = jax.jit(_rollout)
    rng = jax.random.PRNGKey(1)
    _params = dict(
        kp=0.25,  # 1.0,
        ki=0.25,  # 1.0,
        kd=0.1,  # 0.4,
        max_integral=0.1  # 0.1
    )
    rollout = rollout_jit(_params, rng)
    _plot_results(rollout)
    print("Rollout done!")
    plt.show()

    # Html visualization may not work properly, if it's already rendering somewhere else.
    # In such cases, comment-out all but one HTML(pendulum.render(rollout))

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