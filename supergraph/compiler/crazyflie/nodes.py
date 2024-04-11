from typing import Tuple, Union, List, Any, Sequence, Dict
from math import ceil
import jax
from jax.tree_util import tree_map
import jax.numpy as jnp
import numpy as onp
from flax import struct
import flax.linen as nn
import distrax

try:
    from brax.generalized import pipeline as gen_pipeline
    from brax.io import mjcf

    BRAX_INSTALLED = True
except ModuleNotFoundError:
    print("Brax not installed. Install it with `pip install brax`")
    BRAX_INSTALLED = False

from supergraph.compiler.rl import NormalizeVec, SquashState
from supergraph.compiler.graph import Graph
from supergraph.compiler import base
from supergraph.compiler import rl
from supergraph.compiler.base import GraphState, StepState
from supergraph.compiler.node import BaseNode


@struct.dataclass
class CrazyflieBaseParams(base.Base):

    @property
    def pwm_hover(self) -> Union[float, jax.Array]:
        raise NotImplementedError


@struct.dataclass
class OdeParams(CrazyflieBaseParams):
    mass: Union[float, jax.typing.ArrayLike]  # 0.03303
    gain_constant: Union[float, jax.typing.ArrayLike]  # 1.1094
    time_constant: Union[float, jax.typing.ArrayLike]  # 0.183806
    state_space: jax.typing.ArrayLike  # [-15.4666, 1, 3.5616e-5, 7.2345e-8]  # [A,B,C,D]
    thrust_gap: Union[float, jax.typing.ArrayLike]  # 0.88 # This takes care of the motor thrust gap sim2real
    pwm_constants: jax.typing.ArrayLike  # [2.130295e-11, 1.032633e-6, 5.485e-4] # [a,b,c]
    dragxy_constants: jax.typing.ArrayLike  # [9.1785e-7, 0.04076521, 380.8359] # Fa,x
    dragz_constants: jax.typing.ArrayLike  # [10.311e-7, 0.04076521, 380.8359] # Fa,z
    clip_pos: jax.typing.ArrayLike  # [-5.0, 5.0]
    clip_vel: jax.typing.ArrayLike  # [-10.0, 10.0]

    @property
    def pwm_hover(self) -> Union[float, jax.Array]:
        return force_to_pwm(self.pwm_constants, self.mass * 9.81)


@struct.dataclass
class OdeState(base.Base):
    pos: jax.typing.ArrayLike  # [x, y, z]
    vel: jax.typing.ArrayLike  # [xdot, ydot, zdot]
    att: jax.typing.ArrayLike  # [phi, theta, psi]
    ang_vel: jax.typing.ArrayLike  # [p, q, r]
    thrust_state: Union[float, jax.typing.ArrayLike]  # Thrust state


@struct.dataclass
class WorldOutput(base.Base):
    pos: jax.typing.ArrayLike  # [x, y, z]
    vel: jax.typing.ArrayLike  # [xdot, ydot, zdot]
    att: jax.typing.ArrayLike  # [phi, theta, psi]
    ang_vel: jax.typing.ArrayLike  # [p, q, r]
    thrust_state: Union[float, jax.typing.ArrayLike]  # Thrust state


@struct.dataclass
class MoCapOutput(base.Base):
    pos: jax.typing.ArrayLike  # [x, y, z]
    vel: jax.typing.ArrayLike  # [xdot, ydot, zdot]
    att: jax.typing.ArrayLike  # [phi, theta, psi]
    ang_vel: jax.typing.ArrayLike  # [p, q, r]


@struct.dataclass
class AttitudeControllerOutput(base.Base):
    pwm_ref: Union[float, jax.typing.ArrayLike]  # Pwm thrust reference command 10000 to 60000
    phi_ref: Union[float, jax.typing.ArrayLike]  # Phi reference (roll), max: pi/6 rad
    theta_ref: Union[float, jax.typing.ArrayLike]  # Theta reference (pitch), max: pi/6 rad
    psi_ref: Union[float, jax.typing.ArrayLike]  # Psi reference (yaw)
    z_ref: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class DebugAttitudeControllerOutput(AttitudeControllerOutput):
    z: float
    error: float
    proportional: float
    derivative: float
    integral_unclipped: float
    integral: float
    z_force_from_hover: float
    z_force: float
    force: float
    force_hover: float
    pwm_unclipped: float
    pwm_hover: float


@struct.dataclass
class AgentParams(base.Base):
    action_dim: int = struct.field(pytree_node=False, default=2)


@struct.dataclass
class PPOAgentParams(base.Base):
    act_scaling: SquashState = struct.field(default=None)
    obs_scaling: NormalizeVec = struct.field(default=None)
    model: Dict[str, Dict[str, Union[jax.typing.ArrayLike, Any]]] = struct.field(default=None)
    hidden_activation: str = struct.field(pytree_node=False, default="tanh")
    output_activation: str = struct.field(pytree_node=False, default="gaussian")
    stochastic: bool = struct.field(pytree_node=False, default=False)
    action_dim: int = struct.field(pytree_node=False, default=3)

    def apply_actor(self, x: jax.typing.ArrayLike, rng: jax.Array = None) -> jax.Array:
        # Get parameters
        # actor_params = self.train_state.params["params"]["actor"]
        actor_params = self.model["actor"]
        num_layers = sum(["Dense" in k in k for k in actor_params.keys()])

        # Apply hidden layers
        ACTIVATIONS = dict(tanh=nn.tanh, relu=nn.relu, gelu=nn.gelu, softplus=nn.softplus)
        for i in range(num_layers-1):
            hl = actor_params[f"Dense_{i}"]
            num_output_units = hl["kernel"].shape[-1]
            if x is None:
                obs_dim = hl["kernel"].shape[-2]
                x = jnp.zeros((obs_dim,), dtype=float)
            x = nn.Dense(num_output_units).apply({"params": hl}, x)
            x = ACTIVATIONS[self.hidden_activation](x)

        # Apply output layer
        hl = actor_params[f"Dense_{num_layers-1}"]  # Index of final layer
        num_output_units = hl["kernel"].shape[-1]
        x_mean = nn.Dense(num_output_units).apply({"params": hl}, x)
        if self.output_activation == "gaussian":
            if rng is not None:
                log_std = actor_params["log_std"]
                pi = distrax.MultivariateNormalDiag(x_mean, jnp.exp(log_std))
                x = pi.sample(seed=rng)
            else:
                x = x_mean
        else:
            raise NotImplementedError("Gaussian output not implemented yet")
        return x

    def get_action(self, obs: jax.typing.ArrayLike) -> jax.Array:
        # Normalize observation
        norm_obs = self.obs_scaling.normalize(obs, clip=True, subtract_mean=True)
        # Get action
        action = self.apply_actor(norm_obs)
        # Scale action
        action = self.act_scaling.unsquash(action)
        return action


@struct.dataclass
class AgentOutput(base.Base):
    action: jax.typing.ArrayLike  # between -1 and 1 --> [pwm/thrust/zref, phi_ref, theta_ref, psi_ref]


@struct.dataclass
class AttitudeControllerParams(base.Base):
    pwm_hover: Union[float, jax.typing.ArrayLike]  # PWM hover value
    pwm_range: jax.typing.ArrayLike  # PWM range
    max_pwm_from_hover: Union[float, jax.typing.ArrayLike]  # Max PWM from hover
    max_phi_ref: Union[float, jax.typing.ArrayLike]  # Max phi reference (roll), max: pi/6 rad
    max_theta_ref: Union[float, jax.typing.ArrayLike]  # Max theta reference (pitch), max: pi/6 rad
    max_psi_ref: Union[float, jax.typing.ArrayLike]  # Max psi reference (yaw)
    mapping: List[str] = struct.field(pytree_node=False, default_factory=lambda: ["pwm_ref", "theta_ref", "phi_ref", "psi_ref"])

    def to_output(self, action: jax.Array) -> AttitudeControllerOutput:
        actions_mapped = {k: a for a, k in zip(action, self.mapping)}
        actions_mapped["pwm_ref"] = jnp.clip(self.pwm_hover + actions_mapped.get("pwm_ref", 0.0) * self.max_pwm_from_hover,
                                             self.pwm_range[0], self.pwm_range[1])
        actions_mapped["theta_ref"] = actions_mapped.get("theta_ref", 0.0) * self.max_theta_ref
        actions_mapped["phi_ref"] = actions_mapped.get("phi_ref", 0.0) * self.max_phi_ref
        actions_mapped["psi_ref"] = actions_mapped.get("psi_ref", 0.0) * self.max_psi_ref
        actions_mapped["z_ref"] = None
        return AttitudeControllerOutput(**actions_mapped)


@struct.dataclass
class SentinelState(base.Base):
    init_pos: jax.typing.ArrayLike
    init_vel: jax.typing.ArrayLike
    init_att: jax.typing.ArrayLike
    init_ang_vel: jax.typing.ArrayLike
    inclination: jax.typing.ArrayLike


@struct.dataclass
class SentinelParams(base.Base):
    # Ctrl limits
    pwm_from_hover: Union[float, jax.typing.ArrayLike]
    pwm_range: jax.typing.ArrayLike
    phi_max: Union[float, jax.typing.ArrayLike]
    theta_max: Union[float, jax.typing.ArrayLike]
    psi_max: Union[float, jax.typing.ArrayLike]
    # Init states
    x_range: jax.typing.ArrayLike
    y_range: jax.typing.ArrayLike
    z_range: jax.typing.ArrayLike
    inclination_range: jax.typing.ArrayLike
    fixed_position: jax.typing.ArrayLike
    fixed_inclination: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class PIDState(base.Base):
    integral: Union[float, jax.typing.ArrayLike]
    last_error: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class PIDParams(base.Base):
    # Attitude controller params
    pwm_range: jax.typing.ArrayLike  # PWM range
    max_pwm_from_hover: Union[float, jax.typing.ArrayLike]  # Max PWM from hover
    max_z_ref: Union[float, jax.typing.ArrayLike]
    min_z_ref: Union[float, jax.typing.ArrayLike]
    max_phi_ref: Union[float, jax.typing.ArrayLike]  # Max phi reference (roll), max: pi/6 rad
    max_theta_ref: Union[float, jax.typing.ArrayLike]  # Max theta reference (pitch), max: pi/6 rad
    max_psi_ref: Union[float, jax.typing.ArrayLike]  # Max psi reference (yaw)
    # Crazyflie params
    mass: Union[float, jax.typing.ArrayLike]  # 0.03303
    pwm_constants: jax.typing.ArrayLike  # (world) [2.130295e-11, 1.032633e-6, 5.485e-4] # [a,b,c]
    # PID
    kp: Union[float, jax.typing.ArrayLike]
    kd: Union[float, jax.typing.ArrayLike]
    ki: Union[float, jax.typing.ArrayLike]
    max_integral: Union[float, jax.typing.ArrayLike]  # Max integrator state
    mapping: List[str] = struct.field(pytree_node=False,
                                      default_factory=lambda: ["z_ref", "theta_ref", "phi_ref", "psi_ref"])

    @property
    def pwm_hover(self) -> Union[float, jax.Array]:
        return force_to_pwm(self.pwm_constants, self.force_hover)

    @property
    def force_hover(self) -> Union[float, jax.Array]:
        return self.mass * 9.81

    def to_output(self, action: jax.Array) -> AttitudeControllerOutput:
        actions_mapped = {k: a for a, k in zip(action, self.mapping)}
        actions_mapped["pwm_ref"] = None
        actions_mapped["theta_ref"] = actions_mapped.get("theta_ref", 0.0) * self.max_theta_ref
        actions_mapped["phi_ref"] = actions_mapped.get("phi_ref", 0.0) * self.max_phi_ref
        actions_mapped["psi_ref"] = actions_mapped.get("psi_ref", 0.0) * self.max_psi_ref
        if "z_ref" in actions_mapped:
            actions_mapped["z_ref"] = actions_mapped["z_ref"] * (self.max_z_ref - self.min_z_ref) / 2 + (self.max_z_ref + self.min_z_ref) / 2
        else:
            actions_mapped["z_ref"] = 0.
        # actions_mapped["z_ref"] = actions_mapped.get("z_ref", 0.0) * (self.max_z_ref - self.min_z_ref) / 2 + (self.max_z_ref + self.min_z_ref) / 2
        return AttitudeControllerOutput(**actions_mapped)


class Sentinel(BaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_method = dict(start_position="random", inclination="fixed")

    def set_init_method(self, start_position: str, inclination: str):
        self._init_method = dict(start_position=start_position, inclination=inclination)

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> SentinelParams:
        params = SentinelParams(
            pwm_from_hover=15000,
            pwm_range=jnp.array([10000, 60000]),
            phi_max=onp.pi / 6,
            theta_max=onp.pi / 6,
            psi_max=0.,  # No yaw
            x_range=jnp.array([-2.0, 2.0]),
            y_range=jnp.array([-2.0, 2.0]),
            z_range=jnp.array([-0.5, 2.0]),
            inclination_range=jnp.array([0., onp.pi / 7]),  # Max inclination (25.7 degrees)
            fixed_position=jnp.array([0.0, 0.0, 2.0]),  # Above the platform
            fixed_inclination=onp.pi / 7,  # Fixed inclination (25.7 degrees)
        )
        return params

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> SentinelState:
        """Default state of the root."""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        rngs = jax.random.split(rng, num=5)
        params = self.get_step_state(graph_state).params if graph_state else self.default_params(rng[0])
        # Start position
        if self._init_method["start_position"] == "random":
            init_x = jax.random.uniform(rngs[1], shape=(), minval=params.x_range[0], maxval=params.x_range[1])
            init_y = jax.random.uniform(rngs[2], shape=(), minval=params.y_range[0], maxval=params.y_range[1])
            init_z = jax.random.uniform(rngs[3], shape=(), minval=params.z_range[0], maxval=params.z_range[1])
            init_pos = jnp.array([init_x, init_y, init_z])
        elif self._init_method["start_position"] == "fixed":
            init_pos = jnp.array([0.0, 0.0, 2.0])
        else:
            raise ValueError(f"Unknown start position method: {self._init_method['start_position']}")

        # Inclination
        if self._init_method["inclination"] == "random":
            inclination = jax.random.uniform(rngs[4], shape=(), minval=params.inclination_range[0], maxval=params.inclination_range[1])
        elif self._init_method["inclination"] == "fixed":
            inclination = params.fixed_inclination
        else:
            raise ValueError(f"Unknown inclination method: {self._init_method['inclination']}")

        return SentinelState(
            init_pos=init_pos,
            init_vel=jnp.array([0.0, 0.0, 0.0]),
            init_att=jnp.array([0.0, 0.0, 0.0]),
            init_ang_vel=jnp.array([0.0, 0.0, 0.0]),
            inclination=inclination,
        )

    def step(self, step_state: StepState) -> Tuple[StepState, base.Empty]:
        """Step the node."""
        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Update state
        new_step_state = step_state

        # Prepare output
        output = self.init_output(step_state.rng)
        return new_step_state, output


class OdeWorld(BaseNode):
    def __init__(self, *args, dt_substeps: float = 1 / 50, **kwargs):
        super().__init__(*args, **kwargs)
        dt = 1 / self.rate
        self.substeps = ceil(dt / dt_substeps)
        self.dt_substeps = dt / self.substeps

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> OdeParams:
        return OdeParams(
            mass=0.03303,
            gain_constant=1.1094,
            time_constant=0.183806,
            state_space=onp.array([-15.4666, 1, 3.5616e-5, 7.2345e-8]),  # [A,B,C,D]
            thrust_gap=0.88,
            pwm_constants=onp.array([2.130295e-11, 1.032633e-6, 5.485e-4]),
            dragxy_constants=onp.array([9.1785e-7, 0.04076521, 380.8359]),
            dragz_constants=onp.array([10.311e-7, 0.04076521, 380.8359]),
            clip_pos=jnp.array([2., 2., 2.]),
            clip_vel=jnp.array([20., 20., 20.]),
        )

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> OdeState:
        """Default state of the node."""
        ss_world = self.get_step_state(graph_state)
        params = ss_world.params if ss_world else self.init_params(rng, graph_state)
        A, B, C, D = params.state_space
        init_thrust_state = B * params.pwm_hover / (-A)  # Assumes dthrust = 0.

        ss_sentinel = self.get_step_state(graph_state, "sentinel")
        if ss_sentinel is not None:
            state = OdeState(
                pos=ss_sentinel.state.init_pos,
                vel=ss_sentinel.state.init_vel,
                att=ss_sentinel.state.init_att,
                ang_vel=ss_sentinel.state.init_ang_vel,
                thrust_state=init_thrust_state,
            )
        else:
            state = OdeState(
                pos=jnp.array([0.0, 0.0, 2.0]),
                vel=jnp.array([0.0, 0.0, 0.0]),
                att=jnp.array([0.0, 0.0, 0.0]),
                ang_vel=jnp.array([0.0, 0.0, 0.0]),
                thrust_state=init_thrust_state,
            )
        return state

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> WorldOutput:
        """Default output of the node."""
        # Grab output from state
        ss_world = self.get_step_state(graph_state)
        world_state = ss_world.state if ss_world else self.init_state(rng, graph_state)
        return WorldOutput(
                    pos=world_state.pos,
                    vel=world_state.vel,
                    att=world_state.att,
                    ang_vel=world_state.ang_vel,
                    thrust_state=world_state.thrust_state,
                )

    def step(self, step_state: StepState) -> Tuple[StepState, WorldOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Get action
        action = inputs["attitude"][-1].data
        state = state
        next_state = state

        # Calculate next state
        for _ in range(self.substeps):
            next_state = self._runge_kutta4(self._ode_crazyflie, self.dt_substeps, params, next_state, action)

        # Clip position & velocity
        next_state = next_state.replace(pos=jnp.clip(next_state.pos, -params.clip_pos, params.clip_pos))
        next_state = next_state.replace(vel=jnp.clip(next_state.vel, -params.clip_vel, params.clip_vel))

        # Update state
        new_step_state = step_state.replace(state=next_state)

        # Correct for negative roll w.r.t brax & vicon convention
        # todo: THETA SIGN: check if this is necessary
        att = new_step_state.state.att.at[0].multiply(-1)

        # Prepare output
        output = WorldOutput(
            pos=new_step_state.state.pos,
            vel=new_step_state.state.vel,
            # att=new_step_state.state.att,
            att=att,
            ang_vel=new_step_state.state.ang_vel,
            thrust_state=new_step_state.state.thrust_state,
        )
        return new_step_state, output

    @staticmethod
    def _runge_kutta4(ode, dt, params: base.Base, state: base.Base, action: base.Base):
        k1 = ode(params, state, action)
        k2 = ode(params, state + k1 * dt * 0.5, action)
        k3 = ode(params, state + k2 * dt * 0.5, action)
        k4 = ode(params, state + k3 * dt, action)
        return state + (k1 + k2 * 2 + k3 * 2 + k4) * (dt / 6)

    @staticmethod
    def _ode_crazyflie(params: OdeParams, state: OdeState, u: AttitudeControllerOutput) -> OdeState:
        # Unpack params
        mass = params.mass
        gain_c = params.gain_constant
        time_c = params.time_constant
        A, B, C, D = params.state_space
        thrust_gap = params.thrust_gap
        pwm_constants = params.pwm_constants
        dragxy_c = params.dragxy_constants
        dragz_c = params.dragz_constants
        # Unpack state
        x, y, z = state.pos
        xdot, ydot, zdot = state.vel
        phi, theta, psi = state.att
        p, q, r = state.ang_vel
        thrust_state = state.thrust_state
        # Unpack action
        pwm = u.pwm_ref
        phi_ref = u.phi_ref
        theta_ref = -u.theta_ref  # todo: THETA SIGN: check if this is necessary
        psi_ref = u.psi_ref

        # Calculate static thrust offset (from hover)
        # Difference between the steady state value of eq (3.16) and eq (3.3) at the hover point (mass*g)
        # System Identification of the Crazyflie 2.0 Nano Quadrocopter. Julian Forster, 2016.
        #  -https://www.research-collection.ethz.ch/handle/20.500.11850/214143
        hover_force = 9.81 * mass
        hover_pwm = force_to_pwm(pwm_constants, hover_force)  # eq (3.3)

        # Steady state thrust_state for the given pwm
        ss_thrust_state = B / (-A) * hover_pwm  # steady-state with eq (3.16)
        ss_force = 4 * (C * ss_thrust_state + D * hover_pwm)  # Thrust force at steady state
        force_offset = ss_force - hover_force  # Offset from hover
        # ss_pwm = force_to_pwm(pwm_constants, ss_force)   # PWM at steady state
        # pwm_offset = ss_pwm - hover_pwm  # Offset from hover

        # Calculate forces
        # force = pwm_to_force(pwm_constants, pwm)  # Thrust force
        force = 4*(C*thrust_state + D*pwm)  # Thrust force
        force = onp.clip(force - force_offset, 0, None)  # Correct for offset
        # force *= thrust_gap # This takes care of the motor thrust gap sim2real
        pwm_drag = force_to_pwm(pwm_constants, force)  # Symbolic PWM to approximate rotor drag
        dragxy = dragxy_c[0] * 4 * (dragxy_c[1] * pwm_drag + dragxy_c[2])  # Fa,x
        dragz = dragz_c[0] * 4 * (dragz_c[1] * pwm_drag + dragz_c[2])  # Fa,z
        # Calculate dstate
        dpos = jnp.array([xdot, ydot, zdot])
        dvel = jnp.array([
            (jnp.sin(theta) * (force - dragxy * xdot)) / mass,  # x_ddot
            (jnp.sin(phi) * jnp.cos(theta) * (force - dragxy * xdot)) / mass,  # y_ddot
            (jnp.cos(phi) * jnp.cos(theta) * (force - dragz * zdot)) / mass - 9.81  # z_ddot
        ])
        datt = jnp.array([
            (gain_c * phi_ref - phi) / time_c,  # phi_dot
            (gain_c * theta_ref - theta) / time_c,  # theta_dot
            0.  # (gain_c * psi_ref - psi) / time_c  # psi_dot
        ])
        dang_vel = jnp.array([0.0, 0.0, 0.0])  # No angular velocity
        dthrust_state = A * thrust_state + B * pwm  # Thrust_state dot
        dstate = OdeState(pos=dpos, vel=dvel, att=datt, ang_vel=dang_vel, thrust_state=dthrust_state)
        return dstate


def pwm_to_force(pwm_constants: jax.typing.ArrayLike, pwm: Union[float, jax.typing.ArrayLike]) -> Union[float, jax.Array]:
    # Modified formula from Julian Forster's Crazyflie identification
    a, b, c = pwm_constants
    force = 4 * (a * (pwm ** 2) + b * pwm + c)
    return force


def force_to_pwm(pwm_constants: jax.typing.ArrayLike, force: Union[float, jax.typing.ArrayLike]) -> Union[float, jax.Array]:
    # Just the inversion of pwm_to_force
    a, b, c = pwm_constants
    a = 4 * a
    b = 4 * b
    c = 4 * c - force
    d = b ** 2 - 4 * a * c
    pwm = (-b + jnp.sqrt(d)) / (2 * a)
    return pwm


def rpy_to_wxyz(rpy: jax.typing.ArrayLike) -> jax.typing.ArrayLike:
    x = jnp.sin(rpy[0] / 2) * jnp.cos(rpy[1] / 2) * jnp.cos(rpy[2] / 2) - jnp.cos(rpy[0] / 2) * jnp.sin(rpy[1] / 2) * jnp.sin(rpy[2] / 2)
    y = jnp.cos(rpy[0] / 2) * jnp.sin(rpy[1] / 2) * jnp.cos(rpy[2] / 2) + jnp.sin(rpy[0] / 2) * jnp.cos(rpy[1] / 2) * jnp.sin(rpy[2] / 2)
    z = jnp.cos(rpy[0] / 2) * jnp.cos(rpy[1] / 2) * jnp.sin(rpy[2] / 2) - jnp.sin(rpy[0] / 2) * jnp.sin(rpy[1] / 2) * jnp.cos(rpy[2] / 2)
    w = jnp.cos(rpy[0] / 2) * jnp.cos(rpy[1] / 2) * jnp.cos(rpy[2] / 2) + jnp.sin(rpy[0] / 2) * jnp.sin(rpy[1] / 2) * jnp.sin(rpy[2] / 2)
    quat = jnp.array([w, x, y, z])
    return quat

# def eom2d_crazyflie_closedloop(x, u, param):
#     # States are: [x, z, x_dot. z_dot, Theta, thrust_state]
#     # u = [PWM_c, Theta_c] = [10000 to 60000, -1 to 1]
#     # param = [mass, gain constant, time constant]
#
#     pwm_commanded = u[0]
#     a_ss = -15.4666  # State space A
#     b_ss = 1  # State space B
#     c_ss = 3.5616e-5  # State space C
#     d_ss = 7.2345e-8  # State space AD
#     force = 4 * (c_ss * x[5] + d_ss * pwm_commanded)  # Thrust force
#     # force *= 0.88                                                  # This takes care of the motor thrust gap sim2real
#     pwm_drag = force_to_pwm(force)  # Symbolic PWM to approximate rotor drag
#     dragx = 9.1785e-7 * 4 * (0.04076521 * pwm_drag + 380.8359)  # Fa,x
#     dragz = 10.311e-7 * 4 * (0.04076521 * pwm_drag + 380.8359)  # Fa,z
#     theta_commanded = u[1] * pi / 6  # Commanded theta in radians
#     dx = np.array([x[2],  # x_dot
#                    x[3],  # z_dot
#                    (sin(x[4]) * (force - dragx * x[2])) / param[0],  # x_ddot
#                    (cos(x[4]) * (force - dragz * x[3])) / param[0] - 9.81,  # z_ddot
#                    (param[1] * theta_commanded - x[4]) / param[2],  # Theta_dot
#                    a_ss * x[5] + b_ss * pwm_commanded],  # Thrust_state dot
#                   dtype=np.float32)
#     return dx


class MoCap(BaseNode):
    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> MoCapOutput:
        """Default output of the node."""
        # Randomly define some initial sensor values
        output = MoCapOutput(
            pos=jnp.array([0.0, 0.0, 2.0]),
            vel=jnp.array([0.0, 0.0, 0.0]),
            att=jnp.array([0.0, 0.0, 0.0]),
            ang_vel=jnp.array([0.0, 0.0, 0.0]),
        )
        return output

    def step(self, step_state: StepState) -> Tuple[StepState, MoCapOutput]:
        """Step the node."""
        world = step_state.inputs["world"][-1].data

        # Prepare output
        output = MoCapOutput(
            pos=world.pos,
            vel=world.vel,
            att=world.att,
            ang_vel=world.ang_vel,
        )

        # Update state (NOOP)
        new_step_state = step_state

        return new_step_state, output


class AttitudeController(BaseNode):
    def init_params(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> AttitudeControllerParams:
        # Get hover PWM
        ss_world = self.get_step_state(graph_state, "world")
        pwm_hover = ss_world.params.pwm_hover if ss_world is not None else 17000
        # Get sentinel params
        ss_sentinel = self.get_step_state(graph_state, "sentinel")
        pwm_from_hover = ss_sentinel.params.pwm_from_hover if ss_sentinel is not None else 15000
        pwm_range = ss_sentinel.params.pwm_range if ss_sentinel is not None else jnp.array([10000, 60000])
        phi_max = ss_sentinel.params.phi_max if ss_sentinel is not None else onp.pi / 6
        theta_max = ss_sentinel.params.theta_max if ss_sentinel is not None else onp.pi / 6
        psi_max = ss_sentinel.params.psi_max if ss_sentinel is not None else 0.0
        return AttitudeControllerParams(
            pwm_hover=pwm_hover,
            pwm_range=pwm_range,
            max_pwm_from_hover=pwm_from_hover,
            max_phi_ref=phi_max,
            max_theta_ref=theta_max,
            max_psi_ref=psi_max,
        )

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> AttitudeControllerOutput:
        ss_world = self.get_step_state(graph_state, "world")
        hover_pwm = ss_world.params.pwm_hover if ss_world is not None else 17000
        output = AttitudeControllerOutput(
            pwm_ref=hover_pwm,
            phi_ref=0.0,
            theta_ref=0.0,
            psi_ref=0.0,
            z_ref=None,
        )
        return output

    def step(self, step_state: StepState) -> Tuple[StepState, AttitudeControllerOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs
        params: AttitudeControllerParams

        # Update state
        new_step_state = step_state

        # Prepare output
        action = inputs["agent"][-1].data.action
        output = params.to_output(action)
        return new_step_state, output


class PID(AttitudeController):
    def init_params(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> PIDParams:
        # Get base params
        p = super().init_params(rng, graph_state)
        # Get mass
        ss_world = self.get_step_state(graph_state, "world")
        mass = ss_world.params.mass if ss_world is not None else 0.03303
        pwm_constants = ss_world.params.pwm_constants if ss_world is not None else onp.array([2.130295e-11, 1.032633e-6, 5.485e-4])
        # PID
        kp: Union[float, jax.typing.ArrayLike]
        kv: Union[float, jax.typing.ArrayLike]
        ki: Union[float, jax.typing.ArrayLike]
        max_integral: Union[float, jax.typing.ArrayLike]  # Max integrator state
        return PIDParams(
            pwm_range=p.pwm_range,
            max_pwm_from_hover=p.max_pwm_from_hover,
            max_z_ref=2.0,
            min_z_ref=-2.0,
            max_phi_ref=p.max_phi_ref,
            max_theta_ref=p.max_theta_ref,
            max_psi_ref=p.max_psi_ref,
            mass=mass,
            pwm_constants=pwm_constants,
            # kp=8.06,  # 0.447,
            # kd=1.91,  # 0.221
            # ki=0.263,  # 0.246
            kp=1.0,  # 0.447,
            kd=0.4,  # 0.221
            ki=1.0,  # 0.246
            max_integral=0.1  # [N]
            # kp=8.47,  # 0.447,
            # kd=2.283,  # 0.221
            # ki=0.668,  # 0.246
            # max_integral=2.01  # [N]
        )

    def init_state(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> PIDState:
        return PIDState(
            integral=0.0,
            last_error=0.0,
        )

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> AttitudeControllerOutput:
        ss_world = self.get_step_state(graph_state, "world")
        hover_pwm = ss_world.params.pwm_hover if ss_world is not None else 17000
        output = AttitudeControllerOutput(
            pwm_ref=hover_pwm,
            phi_ref=0.0,
            theta_ref=0.0,
            psi_ref=0.0,
            z_ref=0.0,
        )
        # todo: DEBUG
        output = DebugAttitudeControllerOutput(
            # original
            pwm_ref=hover_pwm,
            phi_ref=0.0,
            theta_ref=0.0,
            psi_ref=0.0,
            z_ref=0.0,
            # debug
            z=0.0,
            error=0.0,
            proportional=0.0,
            derivative=0.0,
            integral_unclipped=0.0,
            integral=0.0,
            z_force_from_hover=0.0,
            z_force=0.0,
            force=0.0,
            force_hover=0.0,
            pwm_unclipped=hover_pwm,
            pwm_hover=hover_pwm,
        )
        return output

    def step(self, step_state: base.StepState) -> Tuple[base.StepState, AttitudeControllerOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs
        state: PIDState
        params: PIDParams

        # Prepare output
        action = inputs["agent"][-1].data.action
        output = params.to_output(action)

        # PID
        dt = 1 / self.rate
        z = inputs["mocap"][-1].data.pos[-1]
        zdot = inputs["mocap"][-1].data.vel[-1]
        error = output.z_ref - z
        proportional = params.kp * error
        # last_error = jnp.where(step_state.seq == 0, error, state.last_error)
        # derivative = params.kd * (error - last_error) / dt
        derivative = -params.kd * zdot
        integral_unclipped = state.integral + params.ki * error * dt
        integral = jnp.clip(integral_unclipped, -params.max_integral, params.max_integral)

        z_force_from_hover = proportional + derivative + integral
        z_force = z_force_from_hover + params.mass * 9.81
        z_force = jnp.clip(z_force, 0.0, None)

        # Given the z_force component, the total force can be calculated
        # by finding the force that would be required to maintain the desired z_force given that the force is directed based
        # on the roll and pitch references.
        force = z_force / jnp.cos(output.phi_ref) / jnp.cos(output.theta_ref)
        pwm_unclippped = force_to_pwm(params.pwm_constants, force)

        # clip pwm
        max_pwm = params.pwm_hover + params.max_pwm_from_hover
        min_pwm = params.pwm_hover - params.max_pwm_from_hover
        pwm = jnp.clip(pwm_unclippped, min_pwm, max_pwm)

        # Update output
        new_output = output.replace(pwm_ref=pwm)

        # Update state
        new_state = state.replace(integral=integral, last_error=error)
        new_step_state = step_state.replace(state=new_state)

        # todo: DEBUG
        new_output = DebugAttitudeControllerOutput(
            # original
            pwm_ref=new_output.pwm_ref,
            phi_ref=new_output.phi_ref,
            theta_ref=new_output.theta_ref,
            psi_ref=new_output.psi_ref,
            z_ref=new_output.z_ref,
            # debug
            z=z,
            error=error,
            proportional=proportional,
            derivative=derivative,
            integral_unclipped=integral_unclipped,
            integral=integral,
            z_force_from_hover=z_force_from_hover,
            z_force=z_force,
            force=force,
            force_hover=params.force_hover,
            pwm_unclipped=pwm_unclippped,
            pwm_hover=params.pwm_hover,
        )

        return new_step_state, new_output


class RandomAgent(BaseNode):
    def init_params(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> AgentParams:
        return AgentParams()

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> AgentOutput:
        """Default output of the node."""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        ss = self.get_step_state(graph_state)
        params = ss.params if ss else self.init_params(rng, graph_state)
        action = jax.random.uniform(rng, shape=(params.action_dim,), minval=-1.0, maxval=1.0)
        output = AgentOutput(action=action)
        return output

    def step(self, step_state: StepState) -> Tuple[StepState, AgentOutput]:
        """Step the node."""
        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Force attitude references to be zero
        new_rng, rng_action = jax.random.split(step_state.rng)
        action = jax.random.uniform(rng_action, shape=(params.action_dim,), minval=-1.0, maxval=1.0)
        action = action.at[1:].set(0.) if action.shape[0] > 1 else action
        output = AgentOutput(action=action)

        # Update state
        new_step_state = step_state.replace(rng=new_rng)
        return new_step_state, output


class PPOAgent(RandomAgent):

    def init_params(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> PPOAgentParams:
        return PPOAgentParams(
            act_scaling=None,
            obs_scaling=None,
            model=None,
        )

    def step(self, step_state: base.StepState) -> Tuple[base.StepState, AgentOutput]:
        params = step_state.params
        if any([params.act_scaling is None, params.obs_scaling is None, params.model is None]):
            new_ss, output = super().step(step_state)
            return new_ss, output  # Random action if not initialized

        # Evaluate policy
        obs = self.get_observation(step_state)
        if params.stochastic:
            rng = step_state.rng
            rng, rng_policy = jax.random.split(rng)
            new_ss = step_state.replace(rng=rng)
            action = params.get_action(obs, rng_policy)
        else:
            action = params.get_action(obs)
            new_ss = step_state
        output = AgentOutput(action=action)
        return new_ss, output

    @staticmethod
    def get_observation(step_state: base.StepState) -> jax.Array:
        # Flatten all inputs and state of the supervisor as the observation
        all_data = [i.data for i in step_state.inputs.values()] + [step_state.state]
        all_fdata = []
        for data in all_data:
            # Vectorize data
            vdata = jax.tree_util.tree_map(lambda x: jnp.array(x).reshape(-1), data)
            # Flatten pytree
            fdata, _ = jax.tree_util.tree_flatten(vdata)
            # Add to all_fdata
            all_fdata += fdata
        # Concatenate all_fdata
        cdata = jnp.concatenate(all_fdata)
        return cdata


def save(path, json_rollout):
    """Saves trajectory as an HTML text file."""
    from etils import epath

    path = epath.Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    path.write_text(json_rollout)


def render(rollout: Union[List[GraphState], GraphState]):
    """Render the rollout as an HTML file."""
    if not BRAX_INSTALLED:
        raise ImportError("Brax not installed. Install it with `pip install brax`")
    from brax.io import html

    if isinstance(rollout, list):
        rollout = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *rollout)

    # Extract rollout data
    inclination_rollout = rollout.nodes["sentinel"].state.inclination
    world_output_rollout = rollout.nodes["mocap"].inputs["world"][:, -1].data

    # Determine fps
    dt = rollout.nodes["world"].ts[-1] / rollout.nodes["world"].ts.shape[-1]

    # Initialize system
    CRAZYFLIE_PLATFORM_XML = "/home/r2ci/supergraph/supergraph/compiler/crazyflie/cf2.xml"
    sys = mjcf.load(CRAZYFLIE_PLATFORM_XML)
    sys = sys.replace(dt=dt)

    def _set_pipeline_state(inclination, world_output):
        # Set platform state
        xyz = jnp.array([0., 0., 0.])  # todo: always at the origin

        # Set crazyflie state
        pos = world_output.pos
        rpy = world_output.att  # todo: probably not correct...
        quat = rpy_to_wxyz(rpy)

        # Set initial state
        extra = jnp.array([0.])
        qpos = jnp.concatenate([xyz, jnp.array([inclination]), pos, quat, extra])
        qvel = jnp.zeros_like(qpos)
        x, xd = gen_pipeline.kinematics.forward(sys, qpos, qvel)
        pipeline_state = gen_pipeline.State.init(qpos, qvel, x, xd)  # pytype: disable=wrong-arg-types  # jax-ndarray
        # pipeline_state = gen_pipeline.init(sys, qpos, qvel)
        return pipeline_state

    pipeline_state_rollout = jax.vmap(_set_pipeline_state)(inclination_rollout, world_output_rollout)
    pipeline_state_lst = []
    for i in range(inclination_rollout.shape[0]):
        pipeline_state_i = jax.tree_util.tree_map(lambda x: x[i], pipeline_state_rollout)
        pipeline_state_lst.append(pipeline_state_i)
    rollout_json = html.render(sys, pipeline_state_lst)
    return rollout_json


class Environment(rl.Environment):
    def __len__(self, graph: Graph, step_states: Dict[str, base.StepState] = None, only_init: bool = False, starting_eps: int = 0, randomize_eps: bool = False, order: Tuple[str, ...] = None):
        super().__init__(graph, step_states, only_init, starting_eps, randomize_eps, order)

    def observation_space(self, graph_state: base.GraphState) -> rl.Box:
        cdata = self.get_observation(graph_state)
        low = jnp.full(cdata.shape, -1e6)
        high = jnp.full(cdata.shape, 1e6)
        return rl.Box(low, high, shape=cdata.shape, dtype=cdata.dtype)

    def action_space(self, graph_state: base.GraphState) -> rl.Box:
        ss = self.get_step_state(graph_state)
        high = jnp.ones((ss.params.action_dim,), dtype=float)
        return rl.Box(-high, high, shape=high.shape, dtype=float)

    def get_observation(self, graph_state: base.GraphState) -> jax.Array:
        # Flatten all inputs and state of the supervisor as the observation
        ss = self.get_step_state(graph_state)
        obs = PPOAgent.get_observation(ss)
        return obs

    def get_truncated(self, graph_state: base.GraphState) -> Union[bool, jax.Array]:
        ss = self.get_step_state(graph_state)
        return ss.seq >= self.graph.max_steps

    def get_terminated(self, graph_state: base.GraphState) -> Union[bool, jax.Array]:
        ss = self.get_step_state(graph_state)
        terminated = False
        # terminated = jnp.logical_or(terminated, jnp.any(jnp.abs(ss.inputs["mocap"][-1].data.pos > 10)))
        return terminated  # Not terminating prematurely

    def get_reward(self, graph_state: base.GraphState, action: jax.Array) -> Union[float, jax.Array]:
        # Get goal
        ss_sentinel = self.get_step_state(graph_state, "sentinel")
        state_sentinel: SentinelState = ss_sentinel.state
        inclination = state_sentinel.inclination

        # Get current state
        ss = self.get_step_state(graph_state)
        last_mocap: MoCapOutput = ss.inputs["mocap"][-1].data
        pos = last_mocap.pos
        x, y, z = pos
        vx, vy, vz = last_mocap.vel

        # Get denormalized action
        p_att: PIDParams = self.get_step_state(graph_state, "attitude").params
        output = p_att.to_output(action)
        pwm_ref = output.pwm_ref
        z_ref = output.z_ref
        theta_ref = output.theta_ref
        phi_ref = output.phi_ref
        psi_ref = output.psi_ref

        # Multiplier when the drone is close to the goal
        # c = jnp.clip(jnp.linalg.norm(pos), 0.001, None)  # 0.001 to infinity
        c = 1

        # use jnp.where to add rewards
        reward = 0.0
        reward = reward - jnp.sqrt(x**2 + y**2 + z**2)
        reward = reward - 0.2*jnp.sqrt(vx**2 + vy**2 + vz**2)
        reward = reward - 0.1*jnp.abs(theta_ref)/c
        reward = reward - 0.1*jnp.abs(phi_ref)/c
        reward = reward - 0.1*jnp.abs(psi_ref)/c
        reward = reward - 0.1*jnp.abs(z_ref)/c if z_ref is not None else reward
        reward = reward - 0.1*jnp.abs(z_ref - z)/c if z_ref is not None else reward
        return reward

    def get_info(self, graph_state: base.GraphState, action: jax.Array = None) -> Dict[str, Any]:
        """Override this method if you want to add additional info."""
        return {}

    def get_output(self, graph_state: base.GraphState, action: jax.Array) -> AgentOutput:
        return AgentOutput(action=action)
        # ss_world = self.get_step_state(graph_state, "world")
        # pwm_from_hover, phi_ref, theta_ref = action
        # pwm = pwm_from_hover + ss_world.params.pwm_hover
        # return AttitudeControllerOutput(pwm=pwm, phi_ref=phi_ref, theta_ref=theta_ref, psi_ref=0.0)

    def update_step_state(self, graph_state: base.GraphState, action: jax.Array = None) -> Tuple[base.GraphState, base.StepState]:
        """Override this method if you want to update the step state."""
        step_state = self.get_step_state(graph_state)
        return graph_state, step_state


if __name__ == "__main__":
    """Test the crazyflie pipeline."""
    pwm_constants = jnp.array([2.130295e-11, 1.032633e-6, 5.485e-4])
    state_space = jnp.array([-15.4666, 1, 3.5616e-5, 7.2345e-8])
    A, B, C, D = state_space
    mass = 0.03303
    static_force = 9.81 * mass
    static_pwm = force_to_pwm(pwm_constants, static_force)
    # Calculate thrust_state in steady_state
    thrust_state = B/(-A) * static_pwm
    dynamic_force = 4 * (C * thrust_state + D * static_pwm)  # Thrust force
    dynamic_pwm = force_to_pwm(pwm_constants, dynamic_force)
    print(f"Static PWM: {static_pwm}, Dynamic PWM: {dynamic_pwm}")

    # Initialize system
    import brax.generalized.pipeline as gen_pipeline
    from brax.io import mjcf
    from brax.io import html
    xml_path = "/home/r2ci/supergraph/supergraph/compiler/crazyflie/cf2.xml"
    sys = mjcf.load(xml_path)
    sys = sys.replace(dt=1/50)

    import pickle
    # Set platform state
    xyz = jnp.array([0., 0., 0.])
    inclination = jnp.pi / 7
    # Set crazyflie state
    pos_path = "/home/r2ci/supergraph/scripts/main_crazyflie_pos.pkl"
    att_path = "/home/r2ci/supergraph/scripts/main_crazyflie_att.pkl"
    with open(pos_path, "rb") as f:
        pos = pickle.load(f)
    with open(att_path, "rb") as f:
        rpy = pickle.load(f)

    def _set_pipeline_state(xyz, inclination, rpy, pos):
        # Change the sign of the roll angle
        rpy = rpy.at[0].multiply(-1)

        # Set crazyflie state
        quat = rpy_to_wxyz(rpy)

        # Set initial state
        extra = jnp.array([0.])
        qpos = jnp.concatenate([xyz, jnp.array([inclination]), pos, quat, extra])
        qvel = jnp.zeros_like(qpos)
        x, xd = gen_pipeline.kinematics.forward(sys, qpos, qvel)
        pipeline_state = gen_pipeline.State.init(qpos, qvel, x, xd)  # pytype: disable=wrong-arg-types  # jax-ndarray
        # pipeline_state = gen_pipeline.init(sys, qpos, qvel)
        return pipeline_state

    v_set_pipeline_state = jax.vmap(_set_pipeline_state, in_axes=(None, None, 0, 0))
    rollout = v_set_pipeline_state(xyz, inclination, rpy, pos)
    pipeline_state_lst = []
    for i in range(rpy.shape[0]):
        pipeline_state_i = jax.tree_util.tree_map(lambda x: x[i], rollout)
        pipeline_state_lst.append(pipeline_state_i)
    rollout_json = html.render(sys, pipeline_state_lst)
    save("./nodes_crazyflie.html", rollout_json)
    exit()
    # Set platform state
    xyz = jnp.array([0., 0., 0.5])
    inclination = jnp.pi / 7
    # Set crazyflie state
    pos = jnp.array([0., 0., 0.5])
    rpy = jnp.array([0., inclination, 0.])
    x = jnp.sin(rpy[0] / 2) * jnp.cos(rpy[1] / 2) * jnp.cos(rpy[2] / 2) - jnp.cos(rpy[0] / 2) * jnp.sin(rpy[1] / 2) * jnp.sin(rpy[2] / 2)
    y = jnp.cos(rpy[0] / 2) * jnp.sin(rpy[1] / 2) * jnp.cos(rpy[2] / 2) + jnp.sin(rpy[0] / 2) * jnp.cos(rpy[1] / 2) * jnp.sin(rpy[2] / 2)
    z = jnp.cos(rpy[0] / 2) * jnp.cos(rpy[1] / 2) * jnp.sin(rpy[2] / 2) - jnp.sin(rpy[0] / 2) * jnp.sin(rpy[1] / 2) * jnp.cos(rpy[2] / 2)
    w = jnp.cos(rpy[0] / 2) * jnp.cos(rpy[1] / 2) * jnp.cos(rpy[2] / 2) + jnp.sin(rpy[0] / 2) * jnp.sin(rpy[1] / 2) * jnp.sin(rpy[2] / 2)
    att = jnp.array([w, x, y, z])

    # Set initial state
    # extra = jnp.array([0., 0.])
    extra = jnp.array([0.])
    # extra = jnp.array([])
    qpos = jnp.concatenate([xyz, jnp.array([inclination]), pos, att, extra])
    qvel = jnp.zeros_like(qpos)

    # Hacky way to get the rollout
    x, xd = gen_pipeline.kinematics.forward(sys, qpos, qvel)
    state = gen_pipeline.State.init(qpos, qvel, x, xd)  # pytype: disable=wrong-arg-types  # jax-ndarray
    rollout_json = html.render(sys, [state])
    save("./nodes_crazyflie.html", rollout_json)

    # Correct way to get the rollout
    # pipeline_state = gen_pipeline.init(sys, qpos, qvel)
    # rollout_json = html.render(sys, [pipeline_state])
    # save("./main_crazyflie.html", rollout_json)
