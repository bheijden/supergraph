import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from supergraph.compiler.rl import Environment, LogWrapper, AutoResetWrapper, VecEnv, NormalizeVecObservation, NormalizeVecReward, ClipAction, Environment, Box
# from wrappers import (
#     LogWrapper,
#     BraxGymnaxWrapper,
#     VecEnv,
#     NormalizeVecObservation,
#     NormalizeVecReward,
#     ClipAction,
# )


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config, env: Environment, seed):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env = AutoResetWrapper(env)
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    # env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    # env = LogWrapper(env)
    # env = ClipAction(env)
    # env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    # INIT VECTORIZED ENV
    _, rng_reset = jax.random.split(seed)
    rngs_reset = jax.random.split(rng_reset, config["NUM_ENVS"])
    gsv, obsv, vinfo = env.reset(rngs_reset)
    gsv_excl_aux = gsv.replace(aux={})  # Some data in aux is not vectorized
    gs = jax.tree_util.tree_map(lambda x: x[0], gsv_excl_aux)  # Grab single gs
    env_params = gs

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(
            env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        env_state, obsv, _ = env.reset(reset_rng)
        # obsv, env_state = env.reset(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                env_state, obsv, reward, terminated, truncated, info = env.step(env_state, action)
                done = jnp.logical_or(terminated, truncated)  # todo: handle truncation correctly.
                # rng, _rng = jax.random.split(rng)
                # rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                # obsv, env_state, reward, done, info = env.step(
                #     rng_step, env_state, action, env_params
                # )
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    if len(timesteps) > 0:
                        global_step = timesteps[-1]
                        mean_return = np.mean(return_values)
                        std_return = np.std(return_values)
                        min_return = np.min(return_values)
                        max_return = np.max(return_values)
                        print(
                            f"global step={global_step} | mean return={mean_return:.2f} +- {std_return:.2f} | min return={min_return:.2f} | max return={max_return:.2f}")
                    # for t in range(len(timesteps)):
                    #     print(
                    #         f"global step={timesteps[t]}, episodic return={return_values[t]}"
                    #     )

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    config = dict(
        LR=5e-4,
        NUM_ENVS=64,
        NUM_STEPS=16,
        TOTAL_TIMESTEPS=2e6,
        UPDATE_EPOCHS=4,
        NUM_MINIBATCHES=4,
        GAMMA=0.99,
        GAE_LAMBDA=0.95,
        CLIP_EPS=0.2,
        ENT_COEF=0.01,
        VF_COEF=0.5,
        MAX_GRAD_NORM=0.5,
        ACTIVATION="tanh",
        ANNEAL_LR=False,
        NORMALIZE_ENV=True,
        DEBUG=False,
    )
    # config = {
    #     "LR": 3e-4,
    #     "NUM_ENVS": 2048,
    #     "NUM_STEPS": 10,
    #     "TOTAL_TIMESTEPS": 5e7,
    #     "UPDATE_EPOCHS": 4,
    #     "NUM_MINIBATCHES": 32,
    #     "GAMMA": 0.99,
    #     "GAE_LAMBDA": 0.95,
    #     "CLIP_EPS": 0.2,
    #     "ENT_COEF": 0.0,
    #     "VF_COEF": 0.5,
    #     "MAX_GRAD_NORM": 0.5,
    #     "ACTIVATION": "tanh",
    #     "ENV_NAME": "hopper",
    #     "ANNEAL_LR": False,
    #     "NORMALIZE_ENV": True,
    #     "DEBUG": True,
    # }
    from supergraph.compiler.pendulum.nodes import TestDiskPendulum, TestGymnaxPendulum
    env = TestGymnaxPendulum()

    # Evaluate
    num_seeds = 20
    rng = jax.random.PRNGKey(4)
    rng, seed = jax.random.split(rng, 2)
    train = make_train(config, env=env, seed=seed)
    vtrain = jax.vmap(train, in_axes=0)
    out = vtrain(jax.random.split(rng, num_seeds))
    metrics = out["metrics"]
    return_values = metrics["returned_episode_returns"][metrics["returned_episode"]].reshape(num_seeds, -1)
    import matplotlib.pyplot as plt
    plt.plot(return_values.mean(axis=0))
    plt.show()
    # train_jit = jax.jit(make_train(config, env=env))
    # with jax.disable_jit(False):
    #     out = train_jit(rng)
