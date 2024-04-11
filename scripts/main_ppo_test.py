import functools
import jax
import jax.numpy as jnp

import networkx as nx
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp  # Import tensorflow_probability with jax backend

tfd = tfp.distributions

import supergraph
from supergraph.evaluate import timer
import supergraph.compiler.utils as utils
import supergraph.compiler.pendulum.nodes as pendulum_nodes
import supergraph.compiler.artificial as art
import supergraph.compiler.ppo as ppo
from supergraph.compiler.graph import Graph


if __name__ == "__main__":
    ode_world = pendulum_nodes.OdeWorld(name="world", rate=90, color="grape", order=0)
    brax_world = pendulum_nodes.BraxWorld(name="world", rate=90, color="grape", order=0)

    # Create nodes
    world = ode_world
    sensor = pendulum_nodes.Sensor(name="sensor", rate=30, color="pink", order=1)
    agent = pendulum_nodes.RandomAgent(name="agent", rate=30, color="teal", order=3)
    actuator = pendulum_nodes.Actuator(name="actuator", rate=30, color="orange", order=2)
    nodes = dict(world=world, sensor=sensor, agent=agent, actuator=actuator)

    # Connect nodes
    world.connect(actuator, window=1, name="actuator", skip=True)
    sensor.connect(world, window=1, name="world")
    actuator.connect(agent, window=1, name="agent")
    agent.connect(sensor, window=1, name="sensor")

    # Test API
    # step_states = dict()
    # for n in nodes.values():
    #     ss = n.init_step_state()
    #     next_ss, o = n.step(ss)
    #     n.step(next_ss)
    #     step_states[n.name] = ss

    # Define phase and delays
    phase = dict(sensor=tfd.Deterministic(loc=1e-4),
                 agent=tfd.Deterministic(loc=2e-4),
                 actuator=tfd.Deterministic(loc=3e-4),
                 world=tfd.Deterministic(loc=4e-4),
                 )
    computation_delays = dict(world=tfd.Deterministic(loc=0.),
                              sensor=tfd.Deterministic(loc=0.),
                              agent=tfd.Deterministic(loc=0.),
                              actuator=tfd.Deterministic(loc=0.))
    communication_delays = dict()
    for n in [world, sensor, agent, actuator]:
        for c in n.outputs.values():
            communication_delays[(c.output_node.name, c.input_node.name)] = tfd.Deterministic(loc=0.)

    # Artificially generate graphs
    with timer("generate_graphs"):  # Measure
        graphs_raw = art.generate_graphs(nodes, computation_delays, communication_delays, t_final=5.0, phase=phase, num_episodes=1)

    # Compile with supergraph
    with timer("Graph"):
        graph = Graph(nodes, agent, graphs_raw, debug=True, supergraph_mode="MCS")

    # Visualize raw graphs
    MAKE_PLOTS = False
    if MAKE_PLOTS:  # Visualize "raw" graph
        G = utils.to_networkx_graph(graph.graphs_raw[0], nodes=nodes)
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

    env = pendulum_nodes.Environment(graph, order=("world",), randomize_eps=True)
    # gs, obs, info = env.reset(jax.random.PRNGKey(0))
    # gs, obs, reward, terminated, truncated, info = env.step(gs, )

    # Initialize PPO config
    config = ppo.Config(
        LR=5e-5,
        NUM_ENVS=64,
        NUM_STEPS=32,  # increased from 16 to 32 (to solve approx_kl divergence)
        TOTAL_TIMESTEPS=10e6,
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
        NUM_EVAL_ENVS=20,
        EVAL_FREQ=20,
    )
    train = functools.partial(ppo.train, env)
    train = jax.jit(train)
    res = train(config, jax.random.PRNGKey(0))
    exit()

    # Initialize supergraph
    rng = jax.random.PRNGKey(0)
    gs = graph.init(rng)
    gs = graph.run(gs)

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
        action = gs_rollout.nodes["actuator"].inputs["agent"].data.action[:, :, 0, 0].T
        plt.plot(action)
    plt.show()





