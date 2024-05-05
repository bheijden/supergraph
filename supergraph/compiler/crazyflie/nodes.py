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


try:
    import mujoco
    import mujoco.viewer
    from mujoco import mjx
    MUJOCO_INSTALLED = True
except ModuleNotFoundError:
    print("Mujoco or mujoco-mjx not installed. Install it with `pip install mujoco` or `pip install mujoco-mjx`")
    MUJOCO_INSTALLED = False

from supergraph.compiler.rl import NormalizeVec, SquashState
from supergraph.compiler.graph import Graph
from supergraph.compiler import base
from supergraph.compiler import rl
from supergraph.compiler.base import GraphState, StepState
from supergraph.compiler.node import BaseNode
from supergraph.compiler.crazyflie.pid import PidObject


CRAZYFLIE_PLATFORM_BRAX_XML = "/home/r2ci/supergraph/supergraph/compiler/crazyflie/cf2_brax.xml"
CRAZYFLIE_PLATFORM_MJX_XML = "/home/r2ci/supergraph/supergraph/compiler/crazyflie/cf2.xml"
CRAZYFLIE_PLATFORM_MJX_LW_XML = "/home/r2ci/supergraph/supergraph/compiler/crazyflie/cf2_lightweight.xml"


@struct.dataclass
class Rollout:
    next_gs: base.GraphState
    next_obs: jax.Array
    action: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    done: jax.Array
    info: Any


@struct.dataclass
class CrazyflieBaseParams(base.Base):
    dr: Union[bool, jax.typing.ArrayLike]  # Domain randomization
    mass_var: jax.typing.ArrayLike  # [%]

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
    mass: Union[float, jax.typing.ArrayLike]  # 0.03303
    pos: jax.typing.ArrayLike  # [x, y, z]
    vel: jax.typing.ArrayLike  # [xdot, ydot, zdot]
    att: jax.typing.ArrayLike  # [phi, theta, psi]
    ang_vel: jax.typing.ArrayLike  # [p, q, r]
    thrust_state: Union[float, jax.typing.ArrayLike]  # Thrust state
    pos_plat: jax.typing.ArrayLike  # [x, y, z] # Platform position
    att_plat: jax.typing.ArrayLike  # [phi, theta, psi] # Platform attitude
    vel_plat: jax.typing.ArrayLike  # [xdot, ydot, zdot]  # Platform velocity


@struct.dataclass
class WorldOutput(base.Base):
    pos: jax.typing.ArrayLike  # [x, y, z]
    vel: jax.typing.ArrayLike  # [xdot, ydot, zdot]
    att: jax.typing.ArrayLike  # [phi, theta, psi]
    ang_vel: jax.typing.ArrayLike  # [p, q, r]
    thrust_state: Union[float, jax.typing.ArrayLike]  # Thrust state
    pos_plat: jax.typing.ArrayLike  # [x, y, z] # Platform position
    att_plat: jax.typing.ArrayLike  # [phi, theta, psi] # Platform attitude
    vel_plat: jax.typing.ArrayLike  # [xdot, ydot, zdot]  # Platform velocity


@struct.dataclass
class MoCapParams(base.Base):
    noise: Union[bool, jax.typing.ArrayLike]  # Noise in the measurements
    pos_std: jax.typing.ArrayLike  # [x, y, z]
    vel_std: jax.typing.ArrayLike  # [xdot, ydot, zdot]
    att_std: jax.typing.ArrayLike  # [phi, theta, psi]
    ang_vel_std: jax.typing.ArrayLike  # [p, q, r]
    pos_plat_std: jax.typing.ArrayLike  # [x, y, z]
    att_plat_std: jax.typing.ArrayLike  # [phi, theta, psi]
    vel_plat_std: jax.typing.ArrayLike  # [xdot, ydot, zdot]


@struct.dataclass
class MoCapOutput(base.Base):
    pos: jax.typing.ArrayLike  # [x, y, z]  # Crazyflie position
    vel: jax.typing.ArrayLike  # [xdot, ydot, zdot]  # Crazyflie velocity
    att: jax.typing.ArrayLike  # [phi, theta, psi]  # Crazyflie attitude
    ang_vel: jax.typing.ArrayLike  # [p, q, r]
    pos_plat: jax.typing.ArrayLike  # [x, y, z] # Platform position
    att_plat: jax.typing.ArrayLike  # [phi, theta, psi] # Platform attitude
    vel_plat: jax.typing.ArrayLike  # [xdot, ydot, zdot]  # Platform velocity

    @property
    def inclination(self):
        # polar, azimuth = rpy_to_spherical(self.att_plat)
        # inclination = polar
        return self.att_plat[1]


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
    mapping: List[str] = struct.field(pytree_node=False, default=None)

    def apply_actor(self, x: jax.typing.ArrayLike, rng: jax.Array = None) -> jax.Array:
        # Get parameters
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

    @staticmethod
    def get_observation(pos, vel, att, pos_plat=None, vel_plat=None, att_plat=None, pos_offset=None, prev_action=None) -> jax.Array:
        """Get observation from position, velocity and attitude.

        Either pos, vel, att are of shape (3,) or (window_size, 3),
        where (-1, 3) is the shape of the most recent single observation.

        All are in world frame except for pos_offset which is in cf local frame.

        :param pos: Position of the crazyflie with shape (window_size, 3) or (3,)
        :param vel: Velocity of the crazyflie with shape (window_size, 3) or (3,)
        :param att: Attitude (rpy) of the crazyflie with shape (window_size, 3) or (3,)
        :param pos_plat: Position of the platform with shape (window_size, 3) or (3,)
        :param vel_plat: Velocity of the platform with shape (window_size, 3) or (3,)
        :param att_plat: Attitude (rpy) of the platform with shape (window_size, 3) or (3,)
        :param pos_offset: Offset to add to the cf position (in cf local frame)
        :return: Flattened observation of the policy
        """
        for arg in [pos, vel, att, pos_plat, vel_plat, att_plat, pos_offset]:
            if arg is not None and len(arg.shape) > 1:
                raise NotImplementedError("Batched observations not implemented yet")

        # offset cf_pos based on pos_offset in local frame
        cf2w_R = rpy_to_R(att)
        cf_pos = pos + cf2w_R @ pos_offset if pos_offset is not None else pos
        cf_att = att
        cf_vel = vel
        assert (pos_plat is None) == (att_plat is None), "All or none of pos_plat, att_plat should be provided."
        if pos_plat is not None and att_plat is not None:
            is_pos = pos_plat
            polar, azimuth = rpy_to_spherical(att_plat)
            inclination = polar

            # Make cf=crazyflie rotation matrix
            cf2w_R = rpy_to_R(cf_att)

            # Make is=inclined surface rotation matrix
            Rz = jnp.array([[jnp.cos(azimuth), -jnp.sin(azimuth), 0],
                            [jnp.sin(azimuth), jnp.cos(azimuth), 0],
                            [0, 0, 1]])
            is2w_R = Rz

            # World to is=inclined surface
            w2is_H = jnp.eye(4)
            w2is_H = w2is_H.at[:3, :3].set(is2w_R.T)
            w2is_H = w2is_H.at[:3, 3].set(-is2w_R.T @ is_pos)

            # Transform cf position to is frame
            cf_pos_is = w2is_H @ jnp.concatenate([cf_pos, jnp.array([1.0])])
            cf_pos_is = cf_pos_is[:3]
            cf_att_is = R_to_rpy(is2w_R.T @ cf2w_R)

            # Force cf_yaw=0
            cf_att_is = cf_att_is.at[2].set(0.0)  # todo: UNCOMMENT

            # Transform cf velocity to is frame
            cf_vel_is = is2w_R.T @ cf_vel

            # Prepare observation
            obs = jnp.concatenate([cf_pos_is, cf_vel_is, cf_att_is, jnp.array([inclination])])

            # Assuming vel_plat represents the velocity vector in the world frame
            if vel_plat is not None:
                is_vel = is2w_R.T @ vel_plat.reshape(-1)[-3:]
                obs = jnp.concatenate([obs, is_vel])
        else:
            obs = jnp.concatenate([cf_pos, cf_vel, cf_att])
        if prev_action is not None:
            obs = jnp.concatenate([obs, prev_action])
        return obs

    def to_output(self, action: jax.Array) -> Dict:
        assert self.mapping is not None, "Mapping not provided"
        actions_mapped = {k: a for a, k in zip(action, self.mapping)}
        actions_mapped["pwm_ref"] = None
        actions_mapped["theta_ref"] = actions_mapped.get("theta_ref", 0.0)
        actions_mapped["phi_ref"] = actions_mapped.get("phi_ref", 0.0)
        actions_mapped["psi_ref"] = actions_mapped.get("psi_ref", 0.0)
        actions_mapped["z_ref"] = actions_mapped.get("z_ref", 0.0)
        return actions_mapped


@struct.dataclass
class AgentState(base.Base):
    prev_action: jax.typing.ArrayLike


@struct.dataclass
class AgentOutput(base.Base):
    action: jax.typing.ArrayLike  # between -1 and 1 --> [pwm/thrust/zref, phi_ref, theta_ref, psi_ref]


@struct.dataclass
class AttitudeControllerParams(base.Base):
    rate: Union[float, jax.typing.ArrayLike]  # Rate of the controller
    mapping: List[str] = struct.field(pytree_node=False)

    def to_output(self, action: jax.Array) -> AttitudeControllerOutput:
        actions_mapped = {k: a for a, k in zip(action, self.mapping)}
        actions_mapped["pwm_ref"] = actions_mapped.get("pwm_ref", 10001)
        actions_mapped["theta_ref"] = actions_mapped.get("theta_ref", 0.0)
        actions_mapped["phi_ref"] = actions_mapped.get("phi_ref", 0.0)
        actions_mapped["psi_ref"] = actions_mapped.get("psi_ref", 0.0)
        actions_mapped["z_ref"] = None
        return AttitudeControllerOutput(**actions_mapped)

    def to_command(self, state: Any, action: jax.Array, z: Union[float, jax.Array], vz: Union[float, jax.Array], z_plat: Union[float, jax.Array], att: jax.typing.ArrayLike) -> Tuple[Any, AttitudeControllerOutput]:
        # Scale action
        output = self.to_output(action)
        return state, output


@struct.dataclass
class SentinelState(base.Base):
    init_pos: jax.typing.ArrayLike
    init_vel: jax.typing.ArrayLike
    init_att: jax.typing.ArrayLike
    init_ang_vel: jax.typing.ArrayLike
    init_pos_plat: jax.typing.ArrayLike
    init_att_plat: jax.typing.ArrayLike
    init_vel_plat: jax.typing.ArrayLike


@struct.dataclass
class SentinelParams(base.Base):
    # Ctrl limits
    pwm_from_hover: Union[float, jax.typing.ArrayLike]
    pwm_range: jax.typing.ArrayLike
    phi_max: Union[float, jax.typing.ArrayLike]
    theta_max: Union[float, jax.typing.ArrayLike]
    psi_max: Union[float, jax.typing.ArrayLike]
    z_max: Union[float, jax.typing.ArrayLike]
    # Init states
    x_range: jax.typing.ArrayLike
    y_range: jax.typing.ArrayLike
    z_range: jax.typing.ArrayLike
    azimuth_max: Union[float, jax.typing.ArrayLike]
    polar_range: jax.typing.ArrayLike
    fixed_position: jax.typing.ArrayLike
    fixed_inclination: Union[float, jax.typing.ArrayLike]
    vel_land: Union[float, jax.typing.ArrayLike]
    gamma: Union[float, jax.typing.ArrayLike]
    noise: Union[bool, jax.typing.ArrayLike]
    ctrl_mapping: List[str] = struct.field(pytree_node=False)
    init_cf: str = struct.field(pytree_node=False)
    init_plat: str = struct.field(pytree_node=False)


@struct.dataclass
class PIDState(base.Base):
    integral: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class PIDParams(base.Base):
    # Attitude controller params
    rate: Union[float, jax.typing.ArrayLike]  # Rate of the controller
    # Crazyflie params
    mass: Union[float, jax.typing.ArrayLike]  # 0.03303
    pwm_constants: jax.typing.ArrayLike  # (world) [2.130295e-11, 1.032633e-6, 5.485e-4] # [a,b,c]
    pwm_from_hover: Union[float, jax.typing.ArrayLike]
    # PID
    kp: Union[float, jax.typing.ArrayLike]
    kd: Union[float, jax.typing.ArrayLike]
    ki: Union[float, jax.typing.ArrayLike]
    max_integral: Union[float, jax.typing.ArrayLike]  # Max integrator state
    # Action mapping
    mapping: List[str] = struct.field(pytree_node=False)

    @property
    def pwm_hover(self) -> Union[float, jax.Array]:
        return force_to_pwm(self.pwm_constants, self.force_hover)

    @property
    def force_hover(self) -> Union[float, jax.Array]:
        return self.mass * 9.81

    def to_output(self, action: jax.Array) -> AttitudeControllerOutput:
        actions_mapped = {k: a for a, k in zip(action, self.mapping)}
        actions_mapped["pwm_ref"] = None
        actions_mapped["theta_ref"] = actions_mapped.get("theta_ref", 0.0)
        actions_mapped["phi_ref"] = actions_mapped.get("phi_ref", 0.0)
        actions_mapped["psi_ref"] = actions_mapped.get("psi_ref", 0.0)
        actions_mapped["z_ref"] = actions_mapped.get("z_ref", 0.0)
        return AttitudeControllerOutput(**actions_mapped)

    def to_command(self, state: PIDState, action: jax.Array, z: Union[float, jax.Array], vz: Union[float, jax.Array], z_plat: Union[float, jax.Array], att: jax.typing.ArrayLike = None) -> Tuple[PIDState, AttitudeControllerOutput]:
        # subtract platform z, so that the controller is relative to the platform
        z = z - z_plat

        # Scale action
        output = self.to_output(action)

        # PID
        dt = 1/self.rate
        # error = jnp.clip(output.z_ref - z, -0.1, 0.1)
        error = output.z_ref - z
        proportional = self.kp * error
        # last_error = jnp.where(step_state.seq == 0, error, state.last_error)
        # derivative = self.kd * (error - last_error) / dt
        derivative = -self.kd * vz
        integral_unclipped = state.integral + self.ki * error * dt
        integral = jnp.clip(integral_unclipped, -self.max_integral, self.max_integral)

        z_force_from_hover = proportional + derivative + integral
        z_force = z_force_from_hover + self.mass * 9.81
        z_force = jnp.clip(z_force, 0.0, None)

        # Get angles
        if att is not None:
            phi, theta, _ = att
        else:
            phi, theta = output.phi_ref, output.theta_ref

        # Given the z_force component, the total force can be calculated
        # by finding the force that would be required to maintain the desired z_force given that the force is directed based
        # on the roll and pitch references.
        cphi_ctheta = jnp.clip(jnp.cos(phi) * jnp.cos(theta), 0.21, 1.0)
        force = z_force / cphi_ctheta
        pwm_unclipped = force_to_pwm(self.pwm_constants, force)

        # clip pwm
        max_pwm = self.pwm_hover + self.pwm_from_hover
        min_pwm = self.pwm_hover - self.pwm_from_hover
        pwm = jnp.clip(pwm_unclipped, min_pwm, max_pwm)

        # Update output
        new_output = output.replace(pwm_ref=pwm)

        # Update state
        new_state = state.replace(integral=integral)

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
            force_hover=self.force_hover,
            pwm_unclipped=pwm_unclipped,
            pwm_hover=self.pwm_hover,
        )
        return new_state, new_output


@struct.dataclass
class RewardParams:
    k1: float = 0.78  # Weights att_error
    k2: float = 0.54  # Weights vyz_error
    k3: float = 2.35  # Weights vx*theta
    k4: float = 2.74  # Weights act_att_error
    f1: float = 8.4   # Weights final att_error
    f2: float = 1.76  # Weights final vel_error
    fp: float = 56.5  # Weights final perfect reward
    p: float = 0.05


@struct.dataclass
class ZPIDState(base.Base):
    pidZ: PidObject
    pidVz: PidObject


@struct.dataclass
class ZPIDParams(base.Base):
    # Other
    UINT16_MAX: Union[int, jax.typing.ArrayLike]
    pwm_scale: Union[float, jax.typing.ArrayLike]
    pwm_base: Union[float, jax.typing.ArrayLike]
    pwm_range: Union[float, jax.typing.ArrayLike]
    vel_max_overhead: Union[float, jax.typing.ArrayLike]
    zvel_max: Union[float, jax.typing.ArrayLike]
    # PID
    pidZ: PidObject
    pidVz: PidObject
    # Action mapping
    mapping: List[str] = struct.field(pytree_node=False)

    def reset(self) -> ZPIDState:
        # Replace output limits
        z_outputLimit = jnp.maximum(0.5, self.zvel_max) * self.vel_max_overhead
        vz_outputLimit = self.UINT16_MAX / 2 / self.pwm_scale
        pidZ = self.pidZ.replace(outputLimit=z_outputLimit)
        pidVz = self.pidVz.replace(outputLimit=vz_outputLimit)
        # Reset
        pidZ = pidZ.pidReset()
        pidVz = pidVz.pidReset()
        return ZPIDState(pidZ=pidZ, pidVz=pidVz)

    def to_output(self, action: jax.Array) -> AttitudeControllerOutput:
        actions_mapped = {k: a for a, k in zip(action, self.mapping)}
        actions_mapped["pwm_ref"] = None
        actions_mapped["theta_ref"] = actions_mapped.get("theta_ref", 0.0)
        actions_mapped["phi_ref"] = actions_mapped.get("phi_ref", 0.0)
        actions_mapped["psi_ref"] = actions_mapped.get("psi_ref", 0.0)
        actions_mapped["z_ref"] = actions_mapped.get("z_ref", 0.0)
        return AttitudeControllerOutput(**actions_mapped)

    def to_command(self, state: ZPIDState, action: jax.Array, z: Union[float, jax.Array], vz: Union[float, jax.Array], z_plat: Union[float, jax.Array], att: jax.typing.ArrayLike = None) -> Tuple[ZPIDState, AttitudeControllerOutput]:
        # Get output from action (use mapping)
        output = self.to_output(action)

        # Get PID objects
        pidZ = state.pidZ
        pidVz = state.pidVz

        # subtract platform z, so that the controller is relative to the platform
        z = z - z_plat
        z_ref = output.z_ref

        # Run position controller
        pidZ = pidZ.pidUpdate(desired=z_ref, measured=z)
        vz_ref = pidZ.output

        # Run velocity controller
        pidVz = pidVz.pidUpdate(desired=vz_ref, measured=vz)
        pwmRaw = pidVz.output

        # Scale the thrust and add feed forward term
        pwm_unclipped = pwmRaw * self.pwm_scale + self.pwm_base
        pwm = jnp.clip(pwm_unclipped, self.pwm_range[0], self.pwm_range[1])

        # # Get angles
        # if att is not None:
        #     phi, theta, _ = att
        # else:
        #     phi, theta = output.phi_ref, output.theta_ref
        #
        # # Given the z_force component, the total force can be calculated
        # # by finding the force that would be required to maintain the desired z_force given that the force is directed based
        # # on the roll and pitch references.
        # cphi_ctheta = jnp.clip(jnp.cos(phi) * jnp.cos(theta), 0.21, 1.0)

        # Update output
        new_output = output.replace(pwm_ref=pwm)

        # Update state
        new_state = state.replace(pidZ=pidZ, pidVz=pidVz)

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
            error=pidVz.prevError,
            proportional=pidVz.outP,
            derivative=pidVz.outD,
            integral_unclipped=pidVz.outI,
            integral=pidVz.outI,
            z_force_from_hover=0.,
            z_force=0.,
            force=0.,
            force_hover=0.,
            pwm_unclipped=pwm_unclipped,
            pwm_hover=self.pwm_base,
        )
        return new_state, new_output


class Sentinel(BaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> SentinelParams:
        scale = 0.5  # TODO: MODIFY
        params = SentinelParams(
            # ctrl
            ctrl_mapping=["z_ref", "theta_ref", "phi_ref", "psi_ref"],
            pwm_from_hover=15000,
            pwm_range=jnp.array([20000, 60000]),
            phi_max=scale*onp.pi / 6,
            theta_max=scale*onp.pi / 6,
            psi_max=0.,  # No yaw (or onp.pi?)
            z_max=2.0,
            # init crazyflie
            init_cf="inclined_landing",  # inclined_landing, random, fixed
            fixed_position=jnp.array([0.0, 0.0, 2.0]),  # Above the platform
            x_range=jnp.array([-2.0, 2.0]),
            y_range=jnp.array([-2.0, 2.0]),
            z_range=jnp.array([-0.5, 2.0]),
            # init platform
            init_plat="random",  # random, fixed
            fixed_inclination=onp.pi / 7,  # Fixed inclination (25.7 degrees)
            azimuth_max=0.,  # todo: not yet working....
            polar_range=jnp.array([0., scale*onp.pi / 7]),  # Max inclination (25.7 degrees)
            # Goal
            vel_land=0.1,  # Landing velocity
            gamma=0.99,  # Discount factor
            # Domain randomization
            noise=True,  # Whether to add noise to the measurements & perform domain randomization.
        )
        return params

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> SentinelState:
        """Default state of the root."""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        rngs = jax.random.split(rng, num=6)
        params = self.get_step_state(graph_state).params if graph_state else self.init_params(rng[0])
        # Start position
        if params.init_cf == "random":
            init_x = jax.random.uniform(rngs[1], shape=(), minval=params.x_range[0], maxval=params.x_range[1])
            init_y = jax.random.uniform(rngs[2], shape=(), minval=params.y_range[0], maxval=params.y_range[1])
            init_z = jax.random.uniform(rngs[3], shape=(), minval=params.z_range[0], maxval=params.z_range[1])
            init_pos = jnp.array([init_x, init_y, init_z])
        elif params.init_cf == "fixed":
            init_pos = jnp.array([0.0, 0.0, 2.0])
        elif params.init_cf == "inclined_landing":
            init_x = jax.random.uniform(rngs[1], shape=(), minval=0.5, maxval=params.x_range[1])
            init_y = jax.random.uniform(rngs[2], shape=(), minval=-0.2, maxval=0.2)
            init_z = jax.random.uniform(rngs[3], shape=(), minval=-0.2, maxval=0.2)
            init_pos = jnp.array([init_x, init_y, init_z])
        else:
            raise ValueError(f"Unknown start position method: {params.init_cf}")

        # Inclination
        if params.init_plat == "random":
            polar = jax.random.uniform(rngs[4], shape=(), minval=params.polar_range[0], maxval=params.polar_range[1])
            azimuth = jax.random.uniform(rngs[5], shape=(), minval=-params.azimuth_max, maxval=params.azimuth_max)
        elif params.init_plat == "fixed":
            polar = params.fixed_inclination
            azimuth = 0.0
        else:
            raise ValueError(f"Unknown inclination method: {params.init_plat}")
        init_att_plat = spherical_to_rpy(polar, azimuth)

        return SentinelState(
            init_pos=init_pos,
            init_vel=jnp.array([0.0, 0.0, 0.0]),
            init_att=jnp.array([0.0, 0.0, 0.0]),
            init_ang_vel=jnp.array([0.0, 0.0, 0.0]),
            init_pos_plat=jnp.array([0.0, 0.0, 0.0]),
            init_att_plat=init_att_plat,
            init_vel_plat=jnp.array([0.0, 0.0, 0.0]),
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
        ss_sentinel = self.get_step_state(graph_state, "sentinel")
        dr = ss_sentinel.params.noise if ss_sentinel is not None else False
        return OdeParams(
            dr=dr,  # Whether to perform domain randomization
            mass_var=0.02,
            mass=0.03303,
            gain_constant=1.1094,
            time_constant=0.183806,
            state_space=onp.array([-15.4666, 1, 3.5616e-5, 7.2345e-8]),  # [A,B,C,D]
            thrust_gap=0.88,
            pwm_constants=onp.array([2.130295e-11, 1.032633e-6, 5.485e-4]),
            dragxy_constants=onp.array([9.1785e-7, 0.04076521, 380.8359]),
            dragz_constants=onp.array([10.311e-7, 0.04076521, 380.8359]),
            clip_pos=jnp.array([2.1, 2.1, 2.1]),
            clip_vel=jnp.array([20., 20., 20.]),
        )

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> OdeState:
        """Default state of the node."""
        ss_world = self.get_step_state(graph_state)
        params = ss_world.params if ss_world else self.init_params(rng, graph_state)
        # Determine mass
        rng, rng_mass = jax.random.split(rng)
        use_dr = params.dr
        dmass = use_dr*params.mass*params.mass_var*jax.random.uniform(rng_mass, shape=(), minval=-1, maxval=1)
        mass = params.mass + dmass
        # Determine initial state
        A, B, C, D = params.state_space
        init_thrust_state = B * params.pwm_hover / (-A)  # Assumes dthrust = 0.
        ss_sentinel = self.get_step_state(graph_state, "sentinel")
        if ss_sentinel is not None:
            state = OdeState(
                mass=mass,
                pos=ss_sentinel.state.init_pos,
                vel=ss_sentinel.state.init_vel,
                att=ss_sentinel.state.init_att,
                ang_vel=ss_sentinel.state.init_ang_vel,
                thrust_state=init_thrust_state,
                pos_plat=ss_sentinel.state.init_pos_plat,
                att_plat=ss_sentinel.state.init_att_plat,
                vel_plat=ss_sentinel.state.init_vel_plat,
            )
        else:
            state = OdeState(
                mass=mass,
                pos=jnp.array([0.0, 0.0, 2.0]),
                vel=jnp.array([0.0, 0.0, 0.0]),
                att=jnp.array([0.0, 0.0, 0.0]),
                ang_vel=jnp.array([0.0, 0.0, 0.0]),
                thrust_state=init_thrust_state,
                pos_plat=jnp.array([0.0, 0.0, 0.0]),
                att_plat=jnp.array([0.0, 0.0, 0.0]),
                vel_plat=jnp.array([0.0, 0.0, 0.0]),
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
                    pos_plat=world_state.pos_plat,
                    att_plat=world_state.att_plat,
                    vel_plat=world_state.vel_plat,
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
        # att = new_step_state.state.att.at[0].multiply(-1)  # todo: PHI SIGN: check if this is necessary
        att = new_step_state.state.att

        # Prepare output
        output = WorldOutput(
            pos=new_step_state.state.pos,
            vel=new_step_state.state.vel,
            att=att,
            ang_vel=new_step_state.state.ang_vel,
            thrust_state=new_step_state.state.thrust_state,
            pos_plat=new_step_state.state.pos_plat,
            att_plat=new_step_state.state.att_plat,
            vel_plat=new_step_state.state.vel_plat,
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
    def _old_ode_crazyflie(params: OdeParams, state: OdeState, u: AttitudeControllerOutput) -> OdeState:
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
        phi_ref = u.phi_ref  # todo: PHI SIGN: check if this is necessary
        theta_ref = u.theta_ref
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
        dmass = 0.0  # No mass change
        dpos_plat = jnp.array([0.0, 0.0, 0.0])
        datt_plat = jnp.array([0.0, 0.0, 0.0])
        dvel_plat = jnp.array([0.0, 0.0, 0.0])
        dstate = OdeState(mass=dmass, pos=dpos, vel=dvel, att=datt, ang_vel=dang_vel, thrust_state=dthrust_state,
                          pos_plat=dpos_plat, att_plat=datt_plat, vel_plat=dvel_plat)
        return dstate

    @staticmethod
    def _ode_crazyflie(params: OdeParams, state: OdeState, u: AttitudeControllerOutput) -> OdeState:
        # Unpack params
        mass = params.mass
        gain_c = params.gain_constant
        time_c = params.time_constant
        A, B, C, D = params.state_space
        pwm_constants = params.pwm_constants
        dragxy_c = params.dragxy_constants  # [9.1785e-7, 0.04076521, 380.8359] # Fa,x
        dragz_c = params.dragz_constants # [10.311e-7, 0.04076521, 380.8359] # Fa,z
        # Unpack state
        x, y, z = state.pos
        xdot, ydot, zdot = state.vel
        phi, theta, psi = state.att
        p, q, r = state.ang_vel
        thrust_state = state.thrust_state
        # Unpack action
        pwm = u.pwm_ref
        phi_ref = u.phi_ref
        theta_ref = u.theta_ref
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

        # Calculate forces
        force_thrust = 4 * (C * thrust_state + D * pwm)  # Thrust force
        force_thrust = onp.clip(force_thrust - force_offset, 0, None)  # Correct for offset

        # Calculate rotation matrix
        R = rpy_to_R(jnp.array([phi, theta, psi]))

        # Calculate drag matrix
        # pwm_drag = force_to_pwm(pwm_constants, force_thrust)  # Symbolic PWM to approximate rotor drag
        # dragxy = dragxy_c[0] * 4 * (dragxy_c[1] * pwm_drag + dragxy_c[2])  # Fa,x
        # dragz = dragz_c[0] * 4 * (dragz_c[1] * pwm_drag + dragz_c[2])  # Fa,z
        # drag_matrix = jnp.array([
        #     [dragxy, 0, 0],
        #     [0, dragxy, 0],
        #     [0, 0, dragz]
        # ])
        # force_drag = drag_matrix @ jnp.array([xdot, ydot, zdot]) # todo: should be vel in body frame...

        # Calculate dstate
        dpos = jnp.array([xdot, ydot, zdot])
        dvel = R @ jnp.array([0, 0, force_thrust / mass]) - jnp.array([0, 0, 9.81])
        datt = jnp.array([
            (gain_c * phi_ref - phi) / time_c,  # phi_dot
            (gain_c * theta_ref - theta) / time_c,  # theta_dot
            0.  # (gain_c * psi_ref - psi) / time_c  # psi_dot
        ])
        dang_vel = jnp.array([0.0, 0.0, 0.0])  # No angular velocity
        dthrust_state = A * thrust_state + B * pwm  # Thrust_state dot
        dmass = 0.0  # No mass change
        dpos_plat = jnp.array([0.0, 0.0, 0.0])
        datt_plat = jnp.array([0.0, 0.0, 0.0])
        dvel_plat = jnp.array([0.0, 0.0, 0.0])
        dstate = OdeState(mass=dmass, pos=dpos, vel=dvel, att=datt, ang_vel=dang_vel, thrust_state=dthrust_state,
                          pos_plat=dpos_plat, att_plat=datt_plat, vel_plat=dvel_plat)
        return dstate


class MoCap(BaseNode):
    def init_params(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> MoCapParams:
        ss_sentinel = self.get_step_state(graph_state, "sentinel")
        noise = ss_sentinel.params.noise if ss_sentinel is not None else False
        return MoCapParams(
            noise=noise,
            pos_std=onp.array([0.01, 0.01, 0.01], dtype=float),     # [x, y, z]
            vel_std=onp.array([0.02, 0.02, 0.02], dtype=float),        # [xdot, ydot, zdot]
            att_std=onp.array([0.01, 0.01, 0.01], dtype=float),     # [phi, theta, psi]
            ang_vel_std=onp.array([0.1, 0.1, 0.1], dtype=float),        # [p, q, r]
            pos_plat_std=onp.array([0.01, 0.01, 0.01], dtype=float),  # [x, y, z]
            att_plat_std=onp.array([0.01, 0.01, 0.01], dtype=float),  # [phi, theta, psi]
            vel_plat_std=onp.array([0.02, 0.02, 0.02], dtype=float),   # [xdot, ydot, zdot]
        )

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> MoCapOutput:
        """Default output of the node."""
        # Randomly define some initial sensor values
        ss_sentinel = self.get_step_state(graph_state, "sentinel")
        att_plat = ss_sentinel.state.init_att_plat if ss_sentinel is not None else jnp.array([0.0, onp.pi/7, 0.0])
        output = MoCapOutput(
            pos=jnp.array([0.0, 0.0, 2.0]),
            vel=jnp.array([0.0, 0.0, 0.0]),
            att=jnp.array([0.0, 0.0, 0.0]),
            ang_vel=jnp.array([0.0, 0.0, 0.0]),
            pos_plat=jnp.array([0.0, 0.0, 0.0]),
            att_plat=att_plat,
            vel_plat=jnp.array([0.0, 0.0, 0.0]),
        )
        return output

    def step(self, step_state: StepState) -> Tuple[StepState, MoCapOutput]:
        """Step the node."""
        world = step_state.inputs["world"][-1].data
        use_noise = step_state.params.noise
        params = step_state.params

        # Sample small amount of noise to pos, vel (std=0.05), pos(std=0.005)
        rngs = jax.random.split(step_state.rng, 8)
        new_rng = rngs[0]
        pos_noise = params.pos_std*jax.random.normal(rngs[1], world.pos.shape)
        vel_noise = params.vel_std*jax.random.normal(rngs[2], world.vel.shape)
        att_noise = params.att_std*jax.random.normal(rngs[3], world.att.shape)
        ang_vel_noise = params.ang_vel_std*jax.random.normal(rngs[4], world.ang_vel.shape)
        pos_plat_noise = params.pos_plat_std*jax.random.normal(rngs[5], world.pos_plat.shape)
        att_plat_noise = params.att_plat_std*jax.random.normal(rngs[6], world.att_plat.shape)
        vel_plat_noise = params.vel_plat_std*jax.random.normal(rngs[7], world.vel_plat.shape)

        # Prepare output
        output = MoCapOutput(
            pos=world.pos + use_noise*pos_noise,
            vel=world.vel + use_noise*vel_noise,
            att=world.att + use_noise*att_noise,
            ang_vel=world.ang_vel + use_noise*ang_vel_noise,
            pos_plat=world.pos_plat + use_noise*pos_plat_noise + jnp.array([0.0, 0.0, 0.0]),
            att_plat=world.att_plat + use_noise*att_plat_noise,
            vel_plat=world.vel_plat + use_noise*vel_plat_noise,
        )

        # Update state
        new_step_state = step_state.replace(rng=new_rng)

        return new_step_state, output


class AttitudeController(BaseNode):
    def init_params(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> AttitudeControllerParams:
        # Get sentinel params
        ss_sentinel = self.get_step_state(graph_state, "sentinel")
        mapping = ss_sentinel.params.ctrl_mapping if ss_sentinel is not None else ["pwm_ref", "theta_ref", "phi_ref", "psi_ref"]
        return AttitudeControllerParams(
            rate=self.rate,
            mapping=mapping,
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

        # Prepare output
        action = inputs["agent"][-1].data.action
        new_state, output = params.to_command(state, action, z=None, vz=None, z_plat=None, att=None)

        # Update state
        new_step_state = step_state.replace(state=new_state)
        return new_step_state, output


class PID(AttitudeController):
    def init_params(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> PIDParams:
        # Get sentinel params
        ss_sentinel = self.get_step_state(graph_state, "sentinel")
        mapping = ss_sentinel.params.ctrl_mapping if ss_sentinel is not None else ["z_ref", "theta_ref", "phi_ref"]
        pwm_from_hover = ss_sentinel.params.pwm_from_hover if ss_sentinel is not None else 15000
        # Get mass
        ss_world = self.get_step_state(graph_state, "world")
        mass = ss_world.params.mass if ss_world is not None else 0.03303
        pwm_constants = ss_world.params.pwm_constants if ss_world is not None else onp.array([2.130295e-11, 1.032633e-6, 5.485e-4])
        # PID
        return PIDParams(
            rate=self.rate,
            mapping=mapping,
            mass=mass,
            pwm_constants=pwm_constants,
            pwm_from_hover=pwm_from_hover,
            # kp=8.06,  # 0.447,
            # kd=1.91,  # 0.221
            # ki=0.263,  # 0.246
            # kp=1.0,  # 0.447,
            # kd=0.4,  # 0.221
            # ki=1.0,  # 0.246
            # kp=8.47,  # 0.447,
            # kd=2.283,  # 0.221
            # ki=0.668,  # 0.246
            # max_integral=2.01  # [N]
            kp=0.25,
            ki=0.25,
            kd=0.1,
            max_integral=0.1  # [N]
        )

    def init_state(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> PIDState:
        return PIDState(
            integral=0.0,
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
        z = inputs["mocap"][-1].data.pos[-1]
        vz = inputs["mocap"][-1].data.vel[-1]
        z_plat = inputs["mocap"][-1].data.pos_plat[-1]
        att = inputs["mocap"][-1].data.att
        new_state, output = params.to_command(state, action, z=z, vz=vz, z_plat=z_plat, att=att)

        # Update state
        new_step_state = step_state.replace(state=new_state)

        return new_step_state, output


class ZPID(AttitudeController):
    def init_params(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> ZPIDParams:
        # Get sentinel params
        ss_sentinel = self.get_step_state(graph_state, "sentinel")
        mapping = ss_sentinel.params.ctrl_mapping if ss_sentinel is not None else ["z_ref", "theta_ref", "phi_ref"]
        pwm_range = ss_sentinel.params.pwm_range if ss_sentinel is not None else [20000, 60000]
        # Get mass
        ss_world = self.get_step_state(graph_state, "world")
        mass = ss_world.params.mass if ss_world is not None else 0.03303
        pwm_constants = ss_world.params.pwm_constants if ss_world is not None else onp.array([2.130295e-11, 1.032633e-6, 5.485e-4])
        #
        UINT16_MAX = 65_535
        zvel_max = 1.0
        vel_max_overhead = 1.1
        pwm_scale = 1_000
        z_outputLimit = jnp.maximum(0.5, zvel_max) * vel_max_overhead
        vz_outputLimit = UINT16_MAX / 2 / pwm_scale
        pidZ = PidObject.pidInit(kp=2.0, ki=0.5, kd=0.0, outputLimit=z_outputLimit, iLimit=1., dt=1/self.rate, samplingRate=self.rate, cutoffFreq=20., enableDFilter=False)
        pidVz = PidObject.pidInit(kp=25., ki=15., kd=0.0, outputLimit=vz_outputLimit, iLimit=5000., dt=1/self.rate, samplingRate=self.rate, cutoffFreq=20., enableDFilter=False)
        params = ZPIDParams(
            UINT16_MAX=UINT16_MAX,
            pwm_scale=pwm_scale,
            pwm_base=force_to_pwm(pwm_constants, 9.81 * mass),
            pwm_range=pwm_range,
            vel_max_overhead=vel_max_overhead,
            zvel_max=zvel_max,
            # PID
            pidZ=pidZ,
            pidVz=pidVz,
            # Action mapping
            mapping=mapping,
        )
        return params

    def init_state(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> ZPIDState:
        ss = self.get_step_state(graph_state)
        params = ss.params if ss else self.init_params(rng, graph_state)
        state = params.reset()
        return state

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> AttitudeControllerOutput:
        ss_world = self.get_step_state(graph_state, "world")
        hover_pwm = ss_world.params.pwm_hover if ss_world is not None else 40000
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
        z = inputs["mocap"][-1].data.pos[-1]
        vz = inputs["mocap"][-1].data.vel[-1]
        z_plat = inputs["mocap"][-1].data.pos_plat[-1]
        att = inputs["mocap"][-1].data.att
        new_state, output = params.to_command(state, action, z=z, vz=vz, z_plat=z_plat, att=att)

        # Update state
        new_step_state = step_state.replace(state=new_state)

        return new_step_state, output


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
            mapping=None
        )

    def init_state(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> AgentState:
        ss = self.get_step_state(graph_state)
        ss = ss if ss else self.init_params(rng, graph_state)
        return AgentState(prev_action=jnp.zeros(ss.params.action_dim, dtype=jnp.float32))

    def step(self, step_state: base.StepState) -> Tuple[base.StepState, AgentOutput]:
        params = step_state.params
        obs = self.get_observation(step_state)
        if any([params.act_scaling is None, params.obs_scaling is None, params.model is None]):
            new_ss, output = super().step(step_state)
            return new_ss, output  # Random action if not initialized

        # Evaluate policy
        if params.stochastic:
            rng = step_state.rng
            rng, rng_policy = jax.random.split(rng)
            action = params.get_action(obs, rng_policy)
            new_state = AgentState(prev_action=action)
            new_ss = step_state.replace(rng=rng, state=new_state)
        else:
            action = params.get_action(obs)
            new_state = AgentState(prev_action=action)
            new_ss = step_state.replace(state=new_state)
        output = AgentOutput(action=action)
        return new_ss, output

    @staticmethod
    def get_observation(step_state: base.StepState) -> jax.Array:
        mocap = step_state.inputs["mocap"][-1].data
        prev_action = step_state.state.prev_action
        obs = step_state.params.get_observation(pos=mocap.pos, vel=mocap.vel, att=mocap.att,
                                                pos_plat=mocap.pos_plat, att_plat=mocap.att_plat,
                                                # prev_action=prev_action
                                                )
        return obs


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


def rpy_to_wxyz(v: jax.typing.ArrayLike) -> jax.Array:
    """
    Converts euler rotations in degrees to quaternion.
    this follows the Tait-Bryan intrinsic rotation formalism: x-y'-z''
    """
    c1, c2, c3 = jnp.cos(v / 2)
    s1, s2, s3 = jnp.sin(v / 2)
    w = c1 * c2 * c3 - s1 * s2 * s3
    x = s1 * c2 * c3 + c1 * s2 * s3
    y = c1 * s2 * c3 - s1 * c2 * s3
    z = c1 * c2 * s3 + s1 * s2 * c3
    return jnp.array([w, x, y, z])


def rpy_to_R(rpy, convention="xyz"):
    phi, theta, psi = rpy
    Rz = jnp.array([[jnp.cos(psi), -jnp.sin(psi), 0],
                    [jnp.sin(psi), jnp.cos(psi), 0],
                    [0, 0, 1]])
    Ry = jnp.array([[jnp.cos(theta), 0, jnp.sin(theta)],
                    [0, 1, 0],
                    [-jnp.sin(theta), 0, jnp.cos(theta)]])
    Rx = jnp.array([[1, 0, 0],
                    [0, jnp.cos(phi), -jnp.sin(phi)],
                    [0, jnp.sin(phi), jnp.cos(phi)]])
    # Define below which one is Tait-bryan and which one is Euler
    if convention == "xyz":
        R = Rx @ Ry @ Rz  # This uses Tait-Bryan angles (XYZ sequence)
    elif convention == "zyx":
        R = Rz @ Ry @ Rx  # This uses Tait-Bryan angles (ZYX sequence)
    else:
        raise ValueError(f"Unknown convention: {convention}")
    return R


def R_to_rpy(R: jax.typing.ArrayLike, convention="xyz") -> jax.typing.ArrayLike:
    p = jnp.arcsin(R[0, 2])
    def no_gimbal_lock(*_):
        r = jnp.arctan2(-R[1, 2] / jnp.cos(p), R[2, 2] / jnp.cos(p))
        y = jnp.arctan2(-R[0, 1] / jnp.cos(p), R[0, 0] / jnp.cos(p))
        return jnp.array([r, p, y])

    def gimbal_lock(*_):
        # When cos(p) is close to zero, gimbal lock occurs, and many solutions exist.
        # Here, we arbitrarily set roll to zero in this case.
        r = 0
        y = jnp.arctan2(R[1, 0], R[1, 1])
        return jnp.array([r, p, y])
    rpy = jax.lax.cond(jnp.abs(jnp.cos(p)) > 1e-6, no_gimbal_lock, gimbal_lock, None)
    return rpy


def spherical_to_R(polar, azimuth):
    Rz = jnp.array([[jnp.cos(azimuth), -jnp.sin(azimuth), 0],
                    [jnp.sin(azimuth), jnp.cos(azimuth), 0],
                    [0, 0, 1]])
    Ry = jnp.array([[jnp.cos(polar), 0, jnp.sin(polar)],
                    [0, 1, 0],
                    [-jnp.sin(polar), 0, jnp.cos(polar)]])
    R = Ry @ Rz
    return R


def R_to_spherical(R):
    polar = jnp.arccos(R[2, 2])
    azimuth = jnp.arctan2(R[1, 2], R[0, 2])
    return polar, azimuth


def rpy_to_spherical(rpy):
    R = rpy_to_R(rpy, convention="xyz")
    polar, azimuth = R_to_spherical(R)
    return polar, azimuth


def spherical_to_rpy(polar, azimuth):
    R = spherical_to_R(polar, azimuth)
    rpy = R_to_rpy(R)
    return rpy


def save(path, json_rollout):
    """Saves trajectory as an HTML text file."""
    from etils import epath

    path = epath.Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    path.write_text(json_rollout)


def render(rollout: Union[List[GraphState], GraphState], done: jax.Array = None, frame="world"):
    """Render the rollout as an HTML file."""
    if frame not in ["world"]:
        raise NotImplementedError(f"Frame {frame} not implemented")
    if not BRAX_INSTALLED:
        raise ImportError("Brax not installed. Install it with `pip install brax`")
    from brax.io import html

    # Extract rollout data
    world_output_rollout = rollout.nodes["mocap"].inputs["world"][:, -1].data

    # Determine fps
    max_ts = jnp.max(rollout.nodes["agent"].ts)
    max_seq = jnp.max(rollout.nodes["agent"].seq)
    dt = max_ts / max_seq

    # Initialize system
    sys = mjcf.load(CRAZYFLIE_PLATFORM_BRAX_XML)
    sys = sys.replace(dt=dt)

    def _set_pipeline_state(i):
        world_output = world_output_rollout[i]

        platform_qpos, platform_cor_qpos, cf_qpos = get_qpos(world_output.pos, world_output.att,
                                                             world_output.pos_plat, world_output.att_plat)
        if sys.init_q.shape[0] == 24:
            qpos = jnp.concatenate([platform_qpos, platform_cor_qpos, cf_qpos])
        elif sys.init_q.shape[0] == 16:
            qpos = jnp.concatenate([platform_cor_qpos, cf_qpos])
        else:
            raise ValueError(f"Unsupported qpos shape: {sys.init_q.shape}")

        # Set initial state
        # qpos = jnp.concatenate([platform, platform_cor, cf])
        qvel = jnp.zeros_like(qpos)
        x, xd = gen_pipeline.kinematics.forward(sys, qpos, qvel)
        pipeline_state = gen_pipeline.State.init(qpos, qvel, x, xd)  # pytype: disable=wrong-arg-types  # jax-ndarray
        # pipeline_state = gen_pipeline.init(sys, qpos, qvel)
        return pipeline_state

    jit_set_pipeline_state = jax.jit(_set_pipeline_state)
    pipeline_state_lst = [jit_set_pipeline_state(0)]
    for i in range(1, world_output_rollout.pos.shape[0]):
        if done is not None and done[i]:
            break
        pipeline_state_i = jit_set_pipeline_state(i)
        pipeline_state_lst.append(pipeline_state_i)
    rollout_json = html.render(sys, pipeline_state_lst)
    return rollout_json


def rollout(env: Union[rl.Environment, rl.BaseWrapper], rng: jax.Array):
    init_gs, init_obs, info = env.reset(rng)

    def _scan(_carry, _):
        _gs, _obs = _carry
        _ss = _gs.nodes["agent"]
        _action = _ss.params.get_action(_obs)
        next_gs, next_obs, reward, terminated, truncated, info = env.step(_gs, _action)
        done = jnp.logical_or(terminated, truncated)
        r = Rollout(next_gs, next_obs, _action, reward, terminated, truncated, done, info)
        return (next_gs, next_obs), r

    carry = (init_gs, init_obs)
    _, r = jax.lax.scan(_scan, carry, jnp.arange(env.max_steps))
    return r


class Environment(rl.Environment):
    def __len__(self, graph: Graph, step_states: Dict[str, base.StepState] = None, only_init: bool = False, starting_eps: int = 0, randomize_eps: bool = False, order: Tuple[str, ...] = None):
        super().__init__(graph, step_states, only_init, starting_eps, randomize_eps, order)

    def observation_space(self, graph_state: base.GraphState) -> rl.Box:
        cdata = self.get_observation(graph_state)
        low = jnp.full(cdata.shape, -1e6)
        high = jnp.full(cdata.shape, 1e6)
        return rl.Box(low, high, shape=cdata.shape, dtype=cdata.dtype)

    def action_space(self, graph_state: base.GraphState) -> rl.Box:
        ss_sentinel = self.get_step_state(graph_state, "sentinel")
        ss_agent = self.get_step_state(graph_state)
        high_mapping = dict(
            pwm_ref=ss_sentinel.params.pwm_from_hover,
            phi_ref=ss_sentinel.params.phi_max,
            theta_ref=ss_sentinel.params.theta_max,
            psi_ref=ss_sentinel.params.psi_max,
            z_ref=ss_sentinel.params.z_max
        )
        high = jnp.array([high_mapping[k] for a, k in zip(range(ss_agent.params.action_dim), ss_sentinel.params.ctrl_mapping)], dtype=float)
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
        terminated = False
        return terminated  # Not terminating prematurely

    def get_reward(self, graph_state: base.GraphState, action: jax.Array) -> Union[float, jax.Array]:
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

        # use jnp.where to add rewards
        reward = 0.0
        reward = reward - jnp.sqrt(x**2 + y**2 + z**2)
        reward = reward - 0.2*jnp.sqrt(vx**2 + vy**2 + vz**2)
        reward = reward - 0.1*jnp.abs(theta_ref)
        reward = reward - 0.1*jnp.abs(phi_ref)
        reward = reward - 0.1*jnp.abs(psi_ref)
        reward = reward - 0.1*jnp.abs(z_ref) if z_ref is not None else reward
        reward = reward - 0.1*jnp.abs(z_ref - z) if z_ref is not None else reward
        return reward

    def get_info(self, graph_state: base.GraphState, action: jax.Array = None) -> Dict[str, Any]:
        """Override this method if you want to add additional info."""
        return {}

    def get_output(self, graph_state: base.GraphState, action: jax.Array) -> AgentOutput:
        return AgentOutput(action=action)

    def update_step_state(self, graph_state: base.GraphState, action: jax.Array = None) -> Tuple[base.GraphState, base.StepState]:
        """Override this method if you want to update the step state."""
        step_state = self.get_step_state(graph_state)
        new_state = step_state.state.replace(prev_action=action)
        new_step_state = step_state.replace(state=new_state)
        return graph_state, new_step_state


class ReferenceTracking(Environment):

    def get_reward_old(self, graph_state: base.GraphState, action: jax.Array) -> Union[float, jax.Array]:
        # Get current state
        ss = self.get_step_state(graph_state)
        last_mocap: MoCapOutput = ss.inputs["mocap"][-1].data
        pos = last_mocap.pos
        phi, theta, psi = last_mocap.att
        x, y, z = pos
        vx, vy, vz = last_mocap.vel

        # Get denormalized action
        p_att: PIDParams = self.get_step_state(graph_state, "attitude").params
        output = p_att.to_output(action)
        z_ref = output.z_ref
        phi_ref = output.phi_ref
        theta_ref = output.theta_ref
        psi_ref = output.psi_ref

        # Reduce difference with current reference.
        dz_ref = z_ref - z if z_ref is not None else 0.0
        dphi_ref = phi_ref - phi
        dtheta_ref = theta_ref - theta
        dpsi_ref = psi_ref - psi

        # Get prev actions
        # prev_action = ss.state.prev_action
        # prev_output = p_att.to_output(prev_action)
        # prev_z_ref = prev_output.z_ref
        # prev_phi_ref = prev_output.phi_ref
        # prev_theta_ref = prev_output.theta_ref
        # prev_psi_ref = prev_output.psi_ref
        # dz_ref = z_ref - prev_z_ref if z_ref is not None else 0.0
        # dphi_ref = phi_ref - prev_phi_ref
        # dtheta_ref = theta_ref - prev_theta_ref
        # dpsi_ref = psi_ref - prev_psi_ref

        pos_error = jnp.linalg.norm(jnp.array([x, y, z]))
        # C = 0.1/jnp.clip(pos_error, 0.01)
        C = 1.0

        # use jnp.where to add rewards
        cost = 0.0
        cost = cost + pos_error
        cost = cost + 0.2*jnp.sqrt(vx**2 + vy**2 + vz**2)
        cost = cost + 0.3*jnp.abs(theta_ref)
        cost = cost + 0.3*jnp.abs(phi_ref)
        cost = cost + 0.3*jnp.abs(psi_ref)
        cost = cost + 0.1*jnp.abs(z_ref)*C if z_ref is not None else cost
        cost = cost + 0.3*jnp.abs(dz_ref)*C*0
        # cost = cost + 0.5*jnp.abs(dphi_ref)*C
        # cost = cost + 0.5*jnp.abs(dtheta_ref)*C
        # cost = cost + 0.5*jnp.abs(dpsi_ref)*C

        # Get termination conditions
        ss_sentinel = self.get_step_state(graph_state, "sentinel")
        gamma = ss_sentinel.params.gamma
        terminated = self.get_terminated(graph_state)
        truncated = self.get_truncated(graph_state)
        done = jnp.logical_or(terminated, truncated)
        cost = cost * (1-done) + done * (1/(1-gamma)) * cost
        return -cost

    def get_info(self, graph_state: base.GraphState, action: jax.Array = None) -> Dict[str, Any]:
        return {
            # "new_rwd": 0.,
            "is_perfect": False,
            "pos_perfect": False,
            "att_perfect": False,
            "vel_perfect": False,
            "pos_error": 0.,
            "att_error": 0.,
            "vel_error": 0.,
        }

    def get_reward(self, graph_state: base.GraphState, action: jax.Array):
        # Get current state
        state: OdeState = self.get_step_state(graph_state, "world").state
        x, y, z = state.pos
        vx, vy, vz = state.vel
        phi, theta, psi = state.att

        # Get denormalized action
        p_att: PIDParams = self.get_step_state(graph_state, "attitude").params
        output = p_att.to_output(action)
        z_ref = output.z_ref
        phi_ref = output.phi_ref
        theta_ref = output.theta_ref
        psi_ref = output.psi_ref

        # Penalize delta actions
        ss = self.get_step_state(graph_state)
        prev_action = ss.state.prev_action
        prev_output = p_att.to_output(prev_action)
        dz_ref = z_ref - prev_output.z_ref
        dphi_ref = phi_ref - prev_output.phi_ref
        dtheta_ref = theta_ref - prev_output.theta_ref
        dpsi_ref = psi_ref - prev_output.psi_ref

        # Position cost
        goal_pos = jnp.array([0., 0., 1.])
        pos = jnp.array([x, y, z])
        error_position = jnp.linalg.norm(pos - goal_pos)

        # Velocity cost
        error_velocity = jnp.sqrt(vx**2 + vy**2 + vz**2)

        # Orientation cost
        error_angle = jnp.sqrt(phi**2 + theta**2 + psi**2)

        # Action cost
        error_angle_ref = jnp.sqrt(phi_ref**2 + theta_ref**2 + psi_ref**2)
        error_z_ref = (z_ref - goal_pos[2])**2
        error_action = (0.15 * error_z_ref + 0.6 * error_angle_ref)#/(jnp.maximum(error_position, 0.1))

        # Delta action cost
        error_dangle_ref = jnp.sqrt(dphi_ref**2 + dtheta_ref**2 + dpsi_ref**2)
        error_dz_ref = jnp.sqrt(dz_ref**2)
        error_daction = (0.2 * error_dz_ref + 0.6 * error_dangle_ref)  # Penalize delta action more.

        # Bounds cost
        out_of_bounds = False
        out_of_bounds = jnp.logical_or(out_of_bounds, jnp.abs(x) > 1.95)
        out_of_bounds = jnp.logical_or(out_of_bounds, jnp.abs(y) > 1.95)
        out_of_bounds = jnp.logical_or(out_of_bounds, jnp.abs(z) > 1.95)
        bounds_cost = 10 * out_of_bounds

        # Reward params
        rwd_params = RewardParams()
        k1, k2, k3, k4 = rwd_params.k1, rwd_params.k2, rwd_params.k3, rwd_params.k4
        f1, f2 = rwd_params.f1, rwd_params.f2
        fp = rwd_params.fp
        p = rwd_params.p*2

        # Total cost
        # todo: add error_daction
        # eps_cost = error_position + 0.2 * error_velocity + 0.1 * error_angle + error_action + bounds_cost
        eps_cost = error_position + 0.2 * error_velocity + 0.1 * error_angle + error_action + error_daction + bounds_cost
        final_cost = error_position + f1 * error_angle + f2 * error_velocity

        # Is perfect
        pos_perfect = (error_position < (p * 1.5))
        att_perfect = (error_angle < (p * 1))
        vel_perfect = (error_velocity < (p * 5))
        angle_ref_perfect = (error_angle_ref < (p * 1))
        z_ref_perfect = (error_z_ref < (p * 0.5))
        is_perfect = pos_perfect * att_perfect * vel_perfect * angle_ref_perfect * z_ref_perfect
        perfect_cost = -fp * is_perfect

        # Get termination conditions
        ss_sentinel = self.get_step_state(graph_state, "sentinel")
        gamma = ss_sentinel.params.gamma
        terminated = False # pos_perfect * att_perfect * vel_perfect * angle_ref_perfect * z_ref_perfect
        truncated = self.get_truncated(graph_state)
        done = jnp.logical_or(terminated, truncated)
        cost = eps_cost + \
               0 * done * ((1 - terminated) * (1 / (1 - gamma)) + terminated) * final_cost + \
               0 * done * terminated * perfect_cost

        # Info
        info = {
            "is_perfect": is_perfect,
            "pos_perfect": pos_perfect,
            "att_perfect": att_perfect,
            "vel_perfect": vel_perfect,
            "pos_error": error_position,
            "att_error": error_angle,
            "vel_error": error_velocity,
        }
        return -cost, truncated, terminated, info

    def step(self, graph_state: base.GraphState, action: jax.Array):
        """
        Step the environment.
        Can be overridden to provide custom step behavior.

        :param graph_state: The current graph state.
        :param action: The action to take.
        :return: Tuple of (graph_state, observation, reward, terminated, truncated, info)
        """
        # Convert action to output
        output = self.get_output(graph_state, action)
        step_state = self.get_step_state(graph_state)
        # Step the graph
        gs_pre, _ = self.graph.step(graph_state, step_state, output)
        # Get reward, done flags, and some info
        reward, truncated, terminated, info_rwd = self.get_reward(gs_pre, action)
        # Update step_state
        gs_post, _ = self.update_step_state(gs_pre, action)
        # Get observation
        obs = self.get_observation(gs_post)
        # Get info
        info = self.get_info(gs_post, action)
        info.update(info_rwd)
        return gs_post, obs, reward, terminated, truncated, info


class ReferenceTrackingTerminate(Environment):

    def get_info(self, graph_state: base.GraphState, action: jax.Array = None) -> Dict[str, Any]:
        return {
            # "new_rwd": 0.,
            "is_perfect": False,
            "pos_perfect": False,
            "att_perfect": False,
            "vel_perfect": False,
            "pos_error": 0.,
            "att_error": 0.,
            "vel_error": 0.,
        }

    def get_reward(self, graph_state: base.GraphState, action: jax.Array):
        # Get current state
        state: OdeState = self.get_step_state(graph_state, "world").state
        x, y, z = state.pos
        vx, vy, vz = state.vel
        phi, theta, psi = state.att

        # Get denormalized action
        p_att: PIDParams = self.get_step_state(graph_state, "attitude").params
        output = p_att.to_output(action)
        z_ref = output.z_ref
        phi_ref = output.phi_ref
        theta_ref = output.theta_ref
        psi_ref = output.psi_ref

        # Position cost
        goal_pos = jnp.array([0., 0., 1.])
        pos = jnp.array([x, y, z])
        error_position = jnp.linalg.norm(pos - goal_pos)

        # Velocity cost
        error_velocity = jnp.sqrt(vx**2 + vy**2 + vz**2)

        # Orientation cost
        error_angle = jnp.sqrt(phi**2 + theta**2 + psi**2)

        # Action cost
        error_angle_ref = jnp.sqrt(phi_ref**2 + theta_ref**2 + psi_ref**2)
        error_z_ref = (z_ref - goal_pos[2])**2
        error_action = 0.15 * error_z_ref + 0.6 * error_angle_ref#/(jnp.maximum(error_position, 0.1))

        # Bounds cost
        out_of_bounds = False
        out_of_bounds = jnp.logical_or(out_of_bounds, jnp.abs(x) > 1.95)
        out_of_bounds = jnp.logical_or(out_of_bounds, jnp.abs(y) > 1.95)
        out_of_bounds = jnp.logical_or(out_of_bounds, jnp.abs(z) > 1.95)
        bounds_reward = 10 * out_of_bounds

        # Reward params
        # k1: float = 0.78  # Weights att_error
        # k2: float = 0.54  # Weights vyz_error
        # k3: float = 2.35  # Weights vx*theta
        # k4: float = 2.74  # Weights act_att_error
        # f1: float = 8.4  # Weights final att_error
        # f2: float = 1.76  # Weights final vel_error
        # fp: float = 56.5  # Weights final perfect reward
        # p: float = 0.05
        rwd_params = RewardParams()
        k1, k2, k3, k4 = rwd_params.k1, rwd_params.k2, rwd_params.k3, rwd_params.k4
        f1, f2 = rwd_params.f1, rwd_params.f2
        fp = rwd_params.fp
        p = rwd_params.p*2

        # Total cost
        # eps_reward = -1 * error_position - 0.2 * error_velocity - 0.1 * error_angle - error_action - bounds_reward
        eps_cost = error_position + k1*error_angle + k2*error_velocity + k4*error_action #+ bounds_reward
        final_cost = error_position + f1*error_angle + f2*error_velocity

        # Is perfect
        pos_perfect = (error_position < (p * 1.5))
        att_perfect = (error_angle < (p * 1))
        vel_perfect = (error_velocity < (p * 5))
        angle_ref_perfect = (error_angle_ref < (p * 1))
        z_ref_perfect = (error_z_ref < (p * 0.5))
        is_perfect = pos_perfect * att_perfect * vel_perfect * angle_ref_perfect * z_ref_perfect
        perfect_cost = -fp * is_perfect

        # Get termination conditions
        ss_sentinel = self.get_step_state(graph_state, "sentinel")
        gamma = ss_sentinel.params.gamma
        terminated = pos_perfect * att_perfect * vel_perfect * angle_ref_perfect * z_ref_perfect
        truncated = self.get_truncated(graph_state)
        done = jnp.logical_or(terminated, truncated)
        cost = eps_cost + \
               done * ((1 - terminated) * (1 / (1 - gamma)) + terminated) * final_cost + \
               done * terminated * perfect_cost

        # Info
        info = {
            "is_perfect": is_perfect,
            "pos_perfect": pos_perfect,
            "att_perfect": att_perfect,
            "vel_perfect": vel_perfect,
            "pos_error": error_position,
            "att_error": error_angle,
            "vel_error": error_velocity,
        }
        return -cost, truncated, terminated, info

    def step(self, graph_state: base.GraphState, action: jax.Array):
        """
        Step the environment.
        Can be overridden to provide custom step behavior.

        :param graph_state: The current graph state.
        :param action: The action to take.
        :return: Tuple of (graph_state, observation, reward, terminated, truncated, info)
        """
        # Convert action to output
        output = self.get_output(graph_state, action)
        step_state = self.get_step_state(graph_state)
        # Step the graph
        gs_pre, _ = self.graph.step(graph_state, step_state, output)
        # Get reward, done flags, and some info
        reward, truncated, terminated, info_rwd = self.get_reward(gs_pre, action)
        # Update step_state
        gs_post, _ = self.update_step_state(gs_pre, action)
        # Get observation
        obs = self.get_observation(gs_post)
        # Get info
        info = self.get_info(gs_post, action)
        info.update(info_rwd)
        return gs_post, obs, reward, terminated, truncated, info


class InclinedLanding(Environment):
    def __init__(self, *args, xml_path: str = None, reward_params: RewardParams = None, **kwargs):
        super().__init__(*args, **kwargs)
        if not MUJOCO_INSTALLED:
            raise ImportError("Mujoco not installed. Install it with `pip install mujoco` or `pip install mujoco-mjx`")
        self._rwd_params = RewardParams() if reward_params is None else reward_params
        # self._xml_path = CRAZYFLIE_PLATFORM_MJX_LW_XML if xml_path is None else xml_path
        self._xml_path = CRAZYFLIE_PLATFORM_MJX_LW_XML if xml_path is None else xml_path
        self._mj_m = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_d = mujoco.MjData(self._mj_m)
        self._mjx_m = mjx.device_put(self._mj_m)
        self._mjx_d = mjx.device_put(self._mj_d)

    def get_terminated(self, graph_state: base.GraphState) -> Union[bool, jax.Array]:
        state: OdeState = self.get_step_state(graph_state, "world").state
        dmin = contact_distance(self._mjx_m, self._mjx_d, state.pos, state.att, state.pos_plat, state.att_plat)
        terminated = dmin < 0.0
        # platform_qpos, platform_cor_qpos, cf_qpos = get_qpos(state.pos, state.att, state.pos_plat, state.att_plat)
        # if self._mjx_d.qpos.shape[0] == 24:
        #     qpos = jnp.concatenate([platform_qpos, platform_cor_qpos, cf_qpos])
        # elif self._mjx_d.qpos.shape[0] == 16:
        #     qpos = jnp.concatenate([platform_cor_qpos, cf_qpos])
        # else:
        #     print(f"invalid shape")
        # mjx_d = self._mjx_d.replace(qpos=qpos)
        # mjx_d = mjx.forward(self._mjx_m, mjx_d)
        # terminated = mjx_d.contact.dist.min() < 0.0
        # ss = self.get_step_state(graph_state)
        # last_mocap: MoCapOutput = ss.inputs["mocap"][-1].data
        # x, _, _ = last_mocap.pos
        # terminated = x < 0.
        return terminated

    def get_reward(self, graph_state: base.GraphState, action: jax.Array):
        # Get denormalized action
        p_att: PIDParams = self.get_step_state(graph_state, "attitude").params
        output = p_att.to_output(action)
        z_ref = output.z_ref
        theta_ref = output.theta_ref
        phi_ref = output.phi_ref
        psi_ref = output.psi_ref
        att_ref = jnp.array([phi_ref, theta_ref, psi_ref])

        # Get current state
        world_state: OdeState = self.get_step_state(graph_state, "world").state
        pos_target = jnp.array([0., 0., 0.])  # Final position target

        # Get rotation matrices
        R_cf2w_ref = rpy_to_R(att_ref)
        R_cf2w = rpy_to_R(world_state.att)
        R_is2w = rpy_to_R(world_state.att_plat)
        z_cf_ref = R_cf2w_ref[:, 2]
        z_cf = R_cf2w[:, 2]
        z_is = R_is2w[:, 2]  # Final attitude target

        # Calculate attitude error
        att_error = jnp.arccos(jnp.clip(jnp.dot(z_cf, z_is), -1, 1))  # Minimize angle between two z-axis vectors
        act_att_error = jnp.arccos(jnp.clip(jnp.dot(z_cf_ref, z_is), -1, 1))  # Minimize angle between two z-axis vectors

        # Calculate components of the landing velocity
        ss_sentinel = self.get_step_state(graph_state, "sentinel")
        vel_land_ref = -z_is * ss_sentinel.params.vel_land  # target is landing velocity in negative direction of platform z-axis
        vel_land_error = jnp.linalg.norm(vel_land_ref-world_state.vel)
        z_cf_xy = jnp.array([z_cf[0], z_cf[1], 0]) / jnp.linalg.norm(jnp.array([z_cf[0], z_cf[1], 0]))  # Project z-axis to xy-plane
        vel_underact = 0.5*jnp.clip(jnp.dot(z_cf_xy, world_state.vel), None, 0)   # Promote underactuated motion (i.e. velocity in negative z-axis)

        # @struct.dataclass
        # class RewardParams:
        #     k1: float = 0.78  # Weights att_error
        #     k2: float = 0.54  # Weights vyz_error
        #     k3: float = 2.35  # Weights vx*theta
        #     k4: float = 2.74  # Weights act_att_error
        #     f1: float = 8.4  # Weights final att_error
        #     f2: float = 1.76  # Weights final vel_error
        #     fp: float = 56.5  # Weights final perfect reward
        #     p: float = 0.05

        # running cost
        k1, k2, k3, k4 = self._rwd_params.k1, self._rwd_params.k2, self._rwd_params.k3, self._rwd_params.k4
        f1, f2 = self._rwd_params.f1, self._rwd_params.f2
        fp = self._rwd_params.fp
        p = self._rwd_params.p
        pos_error = jnp.linalg.norm(pos_target - world_state.pos)
        vxyz_error = jnp.linalg.norm(world_state.vel - jnp.dot(z_is, world_state.vel) * z_is)
        act_z_error = z_ref ** 2 # todo: * 0
        pos_perfect = (pos_error < (p * 1.5))#*0.66))
        att_perfect = (att_error < (p * 1))#*3))
        vel_perfect = (vel_land_error < (p * 5))#*2))
        is_perfect = pos_perfect * att_perfect * vel_perfect
        cost_eps = pos_error + k1*att_error + k2*vxyz_error + k3*vel_underact + k1*act_att_error + k4*act_z_error
        cost_final = pos_error + f1*att_error + f2*vel_land_error
        cost_perfect = -fp * is_perfect  #*2

        # Get termination conditions
        gamma = ss_sentinel.params.gamma
        terminated = self.get_terminated(graph_state)
        truncated = self.get_truncated(graph_state)
        done = jnp.logical_or(terminated, truncated)
        cost = cost_eps + done * ((1-terminated) * (1/(1-gamma)) + terminated) * cost_final + done * terminated * cost_perfect

        info = {
            "is_perfect": is_perfect,
            "pos_perfect": pos_perfect,
            "att_perfect": att_perfect,
            "vel_perfect": vel_perfect,
            "pos_error": pos_error,
            "att_error": att_error,
            "vel_error": vel_land_error,
        }

        return -cost, truncated, terminated, info

    def get_info(self, graph_state: base.GraphState, action: jax.Array = None) -> Dict[str, Any]:
        return {
            # "new_rwd": 0.,
            "is_perfect": False,
            "pos_perfect": False,
            "att_perfect": False,
            "vel_perfect": False,
            "pos_error": 0.,
            "att_error": 0.,
            "vel_error": 0.,
        }

    def step(self, graph_state: base.GraphState, action: jax.Array):
        """
        Step the environment.
        Can be overridden to provide custom step behavior.

        :param graph_state: The current graph state.
        :param action: The action to take.
        :return: Tuple of (graph_state, observation, reward, terminated, truncated, info)
        """
        # Convert action to output
        output = self.get_output(graph_state, action)
        step_state = self.get_step_state(graph_state)
        # Step the graph
        gs_pre, _ = self.graph.step(graph_state, step_state, output)
        # Get reward, done flags, and some info
        reward, truncated, terminated, info_rwd = self.get_reward(gs_pre, action)
        # Update step_state
        gs_post, _ = self.update_step_state(gs_pre, action)
        # Get observation
        obs = self.get_observation(gs_post)
        # Get info
        info = self.get_info(gs_post, action)
        info.update(info_rwd)
        return gs_post, obs, reward, terminated, truncated, info


def contact_distance(mjx_model, mjx_data, pos, att, pos_plat, att_plat):
    platform_qpos, platform_cor_qpos, cf_qpos = get_qpos(pos, att, pos_plat, att_plat)
    if mjx_data.qpos.shape[0] == 24:
        qpos = jnp.concatenate([platform_qpos, platform_cor_qpos, cf_qpos])
    elif mjx_data.qpos.shape[0] == 16:
        qpos = jnp.concatenate([platform_cor_qpos, cf_qpos])
    else:
        raise ValueError(f"Unsupported qpos shape: {mjx_data.qpos.shape}")
    mjx_d = mjx_data.replace(qpos=qpos)
    mjx_d = mjx.forward(mjx_model, mjx_d)
    dmin = mjx_d.contact.dist.min()
    return dmin


def get_qpos(pos, att, pos_plat, att_plat):
    cf_quat = rpy_to_wxyz(att)
    cf_qpos = jnp.concatenate([pos, cf_quat, jnp.array([0])])

    plat_quat = rpy_to_wxyz(att_plat)
    pitch_plat = jnp.array([0])
    platform_qpos = jnp.concatenate([pos_plat, plat_quat, pitch_plat])

    # Set corrected platform state
    polar_cor, azimuth_cor = rpy_to_spherical(att_plat)
    att_cor = jnp.array([0., 0, azimuth_cor])
    quat_cor = rpy_to_wxyz(att_cor)
    pitch_cor = jnp.array([polar_cor])
    platform_cor_qpos = jnp.concatenate([pos_plat, quat_cor, pitch_cor])
    return platform_qpos, platform_cor_qpos, cf_qpos


if __name__ == "__main__":
    """Test the crazyflie pipeline."""
    # pwm_constants = jnp.array([2.130295e-11, 1.032633e-6, 5.485e-4])
    # state_space = jnp.array([-15.4666, 1, 3.5616e-5, 7.2345e-8])
    # A, B, C, D = state_space
    # mass = 0.03303
    # static_force = 9.81 * mass
    # static_pwm = force_to_pwm(pwm_constants, static_force)
    # # Calculate thrust_state in steady_state
    # thrust_state = B/(-A) * static_pwm
    # dynamic_force = 4 * (C * thrust_state + D * static_pwm)  # Thrust force
    # dynamic_pwm = force_to_pwm(pwm_constants, dynamic_force)
    # print(f"static_force: {static_force}, dynamic_force: {dynamic_force}")
    # print(f"Static PWM: {static_pwm}, Dynamic PWM: {dynamic_pwm}")

    # Initialize system
    import brax.generalized.pipeline as gen_pipeline
    from brax.io import mjcf
    from brax.io import html
    xml_path = "/home/r2ci/supergraph/supergraph/compiler/crazyflie/cf2_brax.xml"
    sys = mjcf.load(xml_path)
    sys = sys.replace(dt=1/10)

    # Set platform state
    num_samples = 50
    pos = jnp.array([0.5, 0., 0.])  # Set the platform at 0.1m in the x-axis
    pos_offset = jnp.array([0., 0., 0.])  # Offset of cf position in local frame (mocap vs bottom of cf)
    att = jnp.array([0., 0., 0.])
    pos_plat = jnp.zeros((num_samples, 3), dtype=float)

    # This is the angles we have.
    polar = jnp.linspace(onp.pi/7, onp.pi / 7, num_samples)
    azimuth = jnp.linspace(0.02, 0.98*onp.pi, num_samples)  # When pitch==0, yaw should not be computable (yaw is arbitrary)

    R = jax.vmap(spherical_to_R)(polar, azimuth)
    rpy = jax.vmap(R_to_rpy)(R)
    polar_recon, azimuth_recon = jax.vmap(rpy_to_spherical)(rpy)
    print("Direct reconstruction: ", (polar_recon - polar).mean(), (azimuth_recon - azimuth).mean())

    # todo: double check that roll does not affect the global pitch and yaw!!
    # todo: we still need to modify.

    # todo: convert global pitch & yaw to rpy attitudes
    #       - visualize the rpy attitudes and verify their rotation corresponds to fixed pitch and yaw around z
    # todo: convert the rpy attitudes back to global pitch and yaw
    #       - If there is no global pitch, global yaw should be arbitrary --> verify that this is the case (fallback to global yaw = 0)
    # todo: verify that the global pitch and yaw are the same as the original pitch and yaw when pitch is non-zero
    #       - obs: global yaw can be used to convert crazyflie pose to platform frame (avoid zero global pitch)
    #       - obs: global pitch can be used as the inclination observation of the platform
    # todo: verify that adding non-zero yaw to rpy attitudes does not affect the global pitch and yaw
    # pos_plat = pos_plat.at[:, 0].set(jnp.linspace(0., -1., num_samples))
    # pos_plat = pos_plat.at[:, 1].set(jnp.linspace(0., -1., num_samples))
    att_plat = jnp.zeros((num_samples, 3), dtype=float)
    att_plat = att_plat.at[:, 0].set(jnp.linspace(0., onp.pi / 2, num_samples))
    att_plat = att_plat.at[:, 1].set(jnp.linspace(0.02, 0.02, num_samples))
    att_plat = att_plat.at[:, 2].set(jnp.linspace(0., 0, num_samples))

    def _in_platform_frame(pos, att, pos_plat, att_plat):
        vel = jnp.zeros_like(pos)
        vel_plat = jnp.zeros_like(pos_plat)
        obs = PPOAgentParams.get_observation(pos, vel, att, pos_plat, vel_plat, att_plat, pos_offset=pos_offset)

        pos_is = obs[:3]
        att_is = obs[6:9]
        polar = obs[9]
        pos_plat_is = pos_plat - pos_plat
        att_plat_is = jnp.array([0, 0, 0])
        # vel_is = obs[3:6]
        # pitch_plat = obs[9]  # inclination
        # vel_plat_is = obs[10:13]

        # Set crazyflie state
        quat_is = rpy_to_wxyz(att_is)
        cf = jnp.concatenate([pos_is, quat_is, jnp.array([0.])])

        # Set platform state
        quat_plat = rpy_to_wxyz(att_plat_is)
        inclination = jnp.array([polar])
        platform = jnp.concatenate([pos_plat_is, quat_plat, inclination])

        # Set platform state
        pos_cor = pos_plat_is
        quat_cor = rpy_to_wxyz(att_plat_is)
        inclination_cor = jnp.array([polar])
        platform_cor = jnp.concatenate([pos_cor, quat_cor, inclination_cor])

        # Set initial state
        if sys.init_q.shape[0] == 24:
            qpos = jnp.concatenate([platform, platform_cor, cf])
        elif sys.init_q.shape[0] == 16:
            qpos = jnp.concatenate([platform_cor, cf])
        else:
            raise ValueError(f"Unsupported qpos shape: {sys.init_q.shape}")
        qvel = jnp.zeros_like(qpos)
        x, xd = gen_pipeline.kinematics.forward(sys, qpos, qvel)
        pipeline_state = gen_pipeline.State.init(qpos, qvel, x, xd)  # pytype: disable=wrong-arg-types  # jax-ndarray
        # pipeline_state = gen_pipeline.init(sys, qpos, qvel)
        return pipeline_state


    def in_world_frame(pos, att, pos_plat, att_plat):
        # Set crazyflie state
        quat = rpy_to_wxyz(att)
        cf = jnp.concatenate([pos, quat, jnp.array([0.])])

        # Set platform state
        quat_plat = rpy_to_wxyz(att_plat)
        pitch_plat = jnp.array([0])  # jnp.array([att_plat[1]])
        platform = jnp.concatenate([pos_plat, quat_plat, pitch_plat])

        # Set corrected platform state
        polar_cor, azimuth_cor = rpy_to_spherical(att_plat)
        att_cor = jnp.array([0., 0, azimuth_cor])
        quat_cor = rpy_to_wxyz(att_cor)
        pitch_cor = jnp.array([polar_cor])
        platform_cor = jnp.concatenate([pos_plat, quat_cor, pitch_cor])

        # Set initial state
        if sys.init_q.shape[0] == 24:
            qpos = jnp.concatenate([platform, platform_cor, cf])
        elif sys.init_q.shape[0] == 16:
            qpos = jnp.concatenate([platform_cor, cf])
        else:
            raise ValueError(f"Unsupported qpos shape: {sys.init_q.shape}")
        qvel = jnp.zeros_like(qpos)
        x, xd = gen_pipeline.kinematics.forward(sys, qpos, qvel)
        pipeline_state = gen_pipeline.State.init(qpos, qvel, x, xd)  # pytype: disable=wrong-arg-types  # jax-ndarray
        # pipeline_state = gen_pipeline.init(sys, qpos, qvel)
        return pipeline_state


    # In platform
    jit_in_platform_frame = jax.jit(_in_platform_frame)
    in_platform_states = []
    for i in range(pos_plat.shape[0]):
        in_platform_i = jit_in_platform_frame(pos, att, pos_plat[i], att_plat[i])
        in_platform_states.append(in_platform_i)
    rollout_json = html.render(sys, in_platform_states)
    save("./in_platform.html", rollout_json)
    print("./in_platform.html saved!")

    # IN world frame
    jit_in_world_frame = jax.jit(in_world_frame)
    in_world_states = []
    for i in range(pos_plat.shape[0]):
        in_world_i = jit_in_world_frame(pos, att, pos_plat[i], att_plat[i])
        in_world_states.append(in_world_i)
    rollout_json = html.render(sys, in_world_states)
    save("./in_world.html", rollout_json)
    print("./in_world.html saved!")
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
    extra = jnp.array([0.])
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
