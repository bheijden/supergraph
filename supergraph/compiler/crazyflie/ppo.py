import supergraph.compiler.ppo as ppo

# Fixed inclination (no noise, fixed mass, vary initial x)
# env: InclinedLanding
fixed_inclination = ppo.Config(
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

# Multi-inclination (noise, mass variation, vary initial x, y, z)
multi_inclination = ppo.Config(
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