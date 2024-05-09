import jax.numpy as jnp
from typing import Dict
import flax.struct as struct
import supergraph.compiler.ppo as ppo


@struct.dataclass
class InclinedLandingConfig(ppo.Config):
    def EVAL_METRICS_JAX_CB(self, total_steps, diagnostics: ppo.Diagnostics, eval_transitions: ppo.Transition = None) -> Dict:
        metrics = super().EVAL_METRICS_JAX_CB(total_steps, diagnostics, eval_transitions)
        total_done = eval_transitions.done.sum()
        done = eval_transitions.done
        info = eval_transitions.info

        metrics["eval/is_perfect"] = (jnp.roll(info["is_perfect"], shift=1, axis=-1) * done).sum() / total_done
        metrics["eval/pos_perfect"] = (jnp.roll(info["pos_perfect"], shift=1, axis=-1) * done).sum() / total_done
        metrics["eval/att_perfect"] = (jnp.roll(info["att_perfect"], shift=1, axis=-1) * done).sum() / total_done
        metrics["eval/vel_perfect"] = (jnp.roll(info["vel_perfect"], shift=1, axis=-1) * done).sum() / total_done
        pos_error_done = jnp.roll(info["pos_error"], shift=1, axis=-2) * done
        att_error_done = jnp.roll(info["att_error"], shift=1, axis=-2) * done
        vel_error_done = jnp.roll(info["vel_error"], shift=1, axis=-2) * done
        metrics["eval/mean_pos_error"] = pos_error_done.sum() / total_done
        metrics["eval/std_pos_error"] = jnp.sqrt(((pos_error_done - metrics["eval/mean_pos_error"]) ** 2 * done).sum() / total_done)
        metrics["eval/mean_att_error"] = att_error_done.sum() / total_done
        metrics["eval/std_att_error"] = jnp.sqrt(((att_error_done - metrics["eval/mean_att_error"]) ** 2 * done).sum() / total_done)
        metrics["eval/mean_vel_error"] = vel_error_done.sum() / total_done
        metrics["eval/std_vel_error"] = jnp.sqrt(((vel_error_done - metrics["eval/mean_vel_error"]) ** 2 * done).sum() / total_done)
        return metrics

    def EVAL_METRICS_HOST_CB(self, metrics: Dict):
        # Standard metrics
        global_step = metrics["train/total_steps"]
        mean_approxkl = metrics["train/mean_approxkl"]
        mean_return = metrics["eval/mean_returns"]
        std_return = metrics["eval/std_returns"]
        mean_length = metrics["eval/mean_lengths"]
        std_length = metrics["eval/std_lengths"]
        total_episodes = metrics["eval/total_episodes"]

        # Extra metrics
        is_perfect = metrics["eval/is_perfect"]
        pos_perfect = metrics["eval/pos_perfect"]
        att_perfect = metrics["eval/att_perfect"]
        vel_perfect = metrics["eval/vel_perfect"]

        if self.VERBOSE:
            print(f"train_steps={global_step:.0f} | eval_eps={total_episodes} | return={mean_return:.1f}+-{std_return:.1f} | "
                  f"length={int(mean_length)}+-{std_length:.1f} | approxkl={mean_approxkl:.4f} | "
                  f"is_perfect={is_perfect:.2f} | pos_perfect={pos_perfect:.2f} | "
                  f"att_perfect={att_perfect:.2f} | vel_perfect={vel_perfect:.2f}"
                  )


# Fixed inclination (no noise, fixed mass, vary initial x)
# env: InclinedLanding
fixed_inclination = InclinedLandingConfig(
    LR=1e-4,
    NUM_ENVS=64,
    NUM_STEPS=128,  # increased from 16 to 32 (to solve approx_kl divergence)
    TOTAL_TIMESTEPS=5e6,
    UPDATE_EPOCHS=32,
    NUM_MINIBATCHES=32,
    GAMMA=0.91,
    GAE_LAMBDA=0.97,
    CLIP_EPS=0.44,
    ENT_COEF=0.01,
    VF_COEF=0.77,
    MAX_GRAD_NORM=0.87,  # or 0.5?
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

# Reference tracking (no noise, fixed mass).
# env: Environment
ref_tracking = ppo.Config(
   LR=1e-4,
   NUM_ENVS=64,
   NUM_STEPS=128,  # increased from 16 to 32 (to solve approx_kl divergence)
   TOTAL_TIMESTEPS=2e6,
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
   EVAL_FREQ=10,
)

# Multi-inclination (noise, mass variation, vary initial x, y, z)
multi_inclination = InclinedLandingConfig(
    LR=5e-4,
    NUM_ENVS=128,  # todo: 128?
    NUM_STEPS=64,  # todo: 128?
    TOTAL_TIMESTEPS=2e6,  # todo: a lot.
    UPDATE_EPOCHS=16,  # todo: a lot --> 8?
    NUM_MINIBATCHES=8,
    GAMMA=0.978,
    GAE_LAMBDA=0.951,
    CLIP_EPS=0.131,
    ENT_COEF=0.01,
    VF_COEF=0.899,
    MAX_GRAD_NORM=0.87,
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
    FIXED_INIT=False,
    OFFSET_STEP=True,
    NUM_EVAL_ENVS=20,
    EVAL_FREQ=20,
)

term_ref_tracking = InclinedLandingConfig(
    LR=5e-4,
    NUM_ENVS=128,
    NUM_STEPS=64,
    TOTAL_TIMESTEPS=10e6,
    UPDATE_EPOCHS=16,
    NUM_MINIBATCHES=8,
    GAMMA=0.978,
    GAE_LAMBDA=0.951,
    CLIP_EPS=0.131,
    ENT_COEF=0.01,
    VF_COEF=0.899,
    MAX_GRAD_NORM=0.87,
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
    FIXED_INIT=False,
    OFFSET_STEP=True,
    NUM_EVAL_ENVS=20,
    EVAL_FREQ=20,
)

# Multi-inclination (noise, mass variation, vary initial x, y, z, azimuth)
multi_inclination_azi = InclinedLandingConfig(
    LR=9.23e-4,
    NUM_ENVS=128,
    NUM_STEPS=64,
    TOTAL_TIMESTEPS=4e6,
    UPDATE_EPOCHS=16,
    NUM_MINIBATCHES=8,
    GAMMA=0.9844,
    GAE_LAMBDA=0.939,
    CLIP_EPS=0.131,
    ENT_COEF=0.01,
    VF_COEF=0.756,
    MAX_GRAD_NORM=0.76,
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
    FIXED_INIT=False,
    OFFSET_STEP=True,
    NUM_EVAL_ENVS=20,
    EVAL_FREQ=20,
)
