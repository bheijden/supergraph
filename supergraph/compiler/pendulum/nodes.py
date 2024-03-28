from typing import Tuple, Union, List
from math import ceil
import jax
import jax.numpy as jnp
import numpy as onp
from flax import struct

try:
    from brax.generalized import pipeline as gen_pipeline
    from brax.io import mjcf

    BRAX_INSTALLED = True
except ModuleNotFoundError:
    print("Brax not installed. Install it with `pip install brax`")
    BRAX_INSTALLED = False

from supergraph.compiler.base import GraphState, StepState
from supergraph.compiler.node import BaseNode


@struct.dataclass
class OdeParams:
    """Pendulum state definition"""

    max_speed: Union[float, jax.typing.ArrayLike]
    J: Union[float, jax.typing.ArrayLike]
    mass: Union[float, jax.typing.ArrayLike]
    length: Union[float, jax.typing.ArrayLike]
    b: Union[float, jax.typing.ArrayLike]
    K: Union[float, jax.typing.ArrayLike]
    R: Union[float, jax.typing.ArrayLike]
    c: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class OdeState:
    """Pendulum state definition"""

    th: Union[float, jax.typing.ArrayLike]
    thdot: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class BraxParams:
    """Pendulum param definition"""

    max_speed: Union[float, jax.typing.ArrayLike]
    friction_loss: Union[float, jax.typing.ArrayLike]
    sys: gen_pipeline.System


@struct.dataclass
class BraxState:
    """Pendulum state definition"""

    pipeline_state: gen_pipeline.State


@struct.dataclass
class WorldOutput:
    """World output definition"""

    th: Union[float, jax.typing.ArrayLike]
    thdot: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class SensorOutput:
    th: Union[float, jax.typing.ArrayLike]
    thdot: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class ActuatorOutput:
    """Pendulum actuator output"""

    action: jax.typing.ArrayLike  # Torque to apply to the pendulum


class OdeWorld(BaseNode):
    def __init__(self, *args, dt_substeps: float = 1 / 100, **kwargs):
        super().__init__(*args, **kwargs)
        dt = 1 / self.rate
        self.substeps = ceil(dt / dt_substeps)
        self.dt_substeps = dt / self.substeps

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> OdeParams:
        """Default params of the node."""
        return OdeParams(
            max_speed=40.0,
            J=0.00019745720783248544,  # 0.000159931461600856,
            mass=0.053909555077552795,  # 0.0508581731919534,
            length=0.0471346490085125,  # 0.0415233722862552,
            b=1.3641421901411377e-05,  # 1.43298488358436e-05,
            K=0.046251337975263596,  # 0.0333391179016334,
            R=8.3718843460083,  # 7.73125142447252,
            c=0.0006091465475037694,  # 0.000975041213361349,
        )

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> OdeState:
        """Default state of the node."""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        init_th = jax.random.uniform(rng, shape=(), minval=-onp.pi, maxval=onp.pi)
        init_thdot = jax.random.uniform(rng, shape=(), minval=-2.0, maxval=2.0)
        return OdeState(th=init_th, thdot=init_thdot)

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> WorldOutput:
        """Default output of the node."""
        # Grab output from state
        ss_world = self.get_step_state(graph_state)
        world_state = ss_world.state if ss_world else self.init_state(rng, graph_state)
        return WorldOutput(th=world_state.th, thdot=world_state.thdot)

    def step(self, step_state: StepState) -> Tuple[StepState, WorldOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Get action
        u = inputs["actuator"].data.action[-1][0]
        x = jnp.array([state.th, state.thdot])
        next_x = x

        # Calculate next state
        for _ in range(self.substeps):
            next_x = self._runge_kutta4(self._ode_disk_pendulum, self.dt_substeps, params, next_x, u)

        # Update state
        next_th, next_thdot = next_x
        next_thdot = jnp.clip(next_thdot, -params.max_speed, params.max_speed)
        new_state = state.replace(th=next_th, thdot=next_thdot)
        new_step_state = step_state.replace(state=new_state)

        # Prepare output
        output = WorldOutput(th=next_th, thdot=next_thdot)
        # print(f"{self.name.ljust(14)} | x: {x} | u: {u} -> next_x: {next_x}")
        return new_step_state, output

    @staticmethod
    def _runge_kutta4(ode, dt, params, x, u):
        k1 = ode(params, x, u)
        k2 = ode(params, x + 0.5 * dt * k1, u)
        k3 = ode(params, x + 0.5 * dt * k2, u)
        k4 = ode(params, x + dt * k3, u)
        return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    @staticmethod
    def _ode_disk_pendulum(params: OdeParams, x, u):
        g, J, m, l, b, K, R, c = 9.81, params.J, params.mass, params.length, params.b, params.K, params.R, params.c
        activation = jnp.sign(x[1])
        ddx = (u * K / R + m * g * l * jnp.sin(x[0]) - b * x[1] - x[1] * K * K / R - c * activation) / J
        return jnp.array([x[1], ddx])


class BraxWorld(BaseNode):
    def __init__(self, *args, dt_substeps: float = 1 / 100, **kwargs):
        super().__init__(*args, **kwargs)
        assert BRAX_INSTALLED, "Brax not installed. Install it with `pip install brax`"
        self.sys = mjcf.loads(DISK_PENDULUM_XML)
        dt = 1 / self.rate
        self.substeps = ceil(dt / dt_substeps)
        self.dt_substeps = dt / self.substeps

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> BraxParams:
        """Default params of the node."""
        # Realistic parameters for the disk pendulum
        damping = 0.00015877
        armature = 6.4940527e-06
        gear = 0.00428677
        mass_weight = 0.05076142
        radius_weight = 0.05121992
        offset = 0.04161447
        friction_loss = 0.00097525

        # Appropriately replace parameters for the disk pendulum
        itransform = self.sys.link.inertia.transform.replace(pos=jnp.array([[0.0, offset, 0.0]]))
        i = self.sys.link.inertia.i.at[0, 0, 0].set(
            0.5 * mass_weight * radius_weight**2
        )  # inertia of cylinder in local frame.
        inertia = self.sys.link.inertia.replace(transform=itransform, mass=jnp.array([mass_weight]), i=i)
        link = self.sys.link.replace(inertia=inertia)
        actuator = self.sys.actuator.replace(gear=jnp.array([gear]))
        dof = self.sys.dof.replace(armature=jnp.array([armature]), damping=jnp.array([damping]))
        new_sys = self.sys.replace(link=link, actuator=actuator, dof=dof, dt=self.dt_substeps)
        return BraxParams(max_speed=40.0, friction_loss=friction_loss, sys=new_sys)

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> BraxState:
        """Default state of the node."""
        if rng is None:
            rng = jax.random.PRNGKey(0)

        # Sample initial state
        init_th = jax.random.uniform(rng, shape=(), minval=-onp.pi, maxval=onp.pi)
        init_thdot = jax.random.uniform(rng, shape=(), minval=-2.0, maxval=2.0)

        # Set the initial state of the disk pendulum
        step_state = self.get_step_state(graph_state)
        params = step_state.params if step_state else self.init_params(rng, graph_state)
        qpos = params.sys.init_q.at[0].set(init_th)
        qvel = jnp.array([init_thdot])
        pipeline_state = gen_pipeline.init(params.sys, qpos, qvel)
        return BraxState(pipeline_state=pipeline_state)

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> WorldOutput:
        """Default output of the node."""
        # Grab output from state
        ss_world = self.get_step_state(graph_state)
        state = ss_world.state if ss_world else self.init_state(rng, graph_state)
        return WorldOutput(th=state.pipeline_state.q[0], thdot=state.pipeline_state.qd[0])

    def step(self, step_state: StepState) -> Tuple[StepState, WorldOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Get action
        action = inputs["actuator"].data.action[-1]

        # Brax does not have static friction implemented
        thdot = state.pipeline_state.qd[0]
        activation = jnp.sign(thdot)
        friction = params.friction_loss * activation / params.sys.actuator.gear[0]
        action_friction = action - friction

        # Run the pipeline for the number of substeps
        def f(state, _):
            return (
                gen_pipeline.step(self.sys, state, action_friction),
                None,
            )

        new_pipeline_state = jax.lax.scan(f, state.pipeline_state, (), self.substeps)[0]

        # Update state
        new_state = state.replace(pipeline_state=new_pipeline_state)
        new_step_state = step_state.replace(state=new_state)

        # Prepare output
        output = WorldOutput(th=new_pipeline_state.q[0], thdot=new_pipeline_state.qd[0])

        return new_step_state, output


class Sensor(BaseNode):
    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> SensorOutput:
        """Default output of the node."""
        # Randomly define some initial sensor values
        th = jnp.pi
        thdot = 0.0
        return SensorOutput(th=th, thdot=thdot)

    def step(self, step_state: StepState) -> Tuple[StepState, SensorOutput]:
        """Step the node."""
        world = step_state.inputs["world"][-1].data

        # Prepare output
        output = SensorOutput(th=world.th, thdot=world.thdot)

        # Update state (NOOP)
        new_step_state = step_state

        return new_step_state, output


class Actuator(BaseNode):
    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> ActuatorOutput:
        """Default output of the node."""
        return ActuatorOutput(action=jnp.array([0.0], dtype=jnp.float32))

    def step(self, step_state: StepState) -> Tuple[StepState, ActuatorOutput]:
        """Step the node."""

        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Update state
        new_step_state = step_state

        # Prepare output
        # key = "controller" if "controller" in inputs else "agent"
        controller_output = next(iter(inputs.values()))[-1].data
        output = ActuatorOutput(action=controller_output.action)
        return new_step_state, output


class RandomAgent(BaseNode):
    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> ActuatorOutput:
        """Default output of the node."""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        action = jax.random.uniform(rng, shape=(1,), minval=-2.0, maxval=2.0)
        return ActuatorOutput(action=action)

    def step(self, step_state: StepState) -> Tuple[StepState, ActuatorOutput]:
        """Step the node."""
        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Prepare output
        rng, rng_net = jax.random.split(step_state.rng)
        action = jax.random.uniform(rng_net, shape=(1,), minval=-2.0, maxval=2.0)
        output = ActuatorOutput(action=action)

        # Update state
        new_step_state = step_state.replace(rng=rng)

        return new_step_state, output


DISK_PENDULUM_XML = """
<mujoco model="disk_pendulum">
    <compiler inertiafromgeom="auto" angle="radian" coordinate="local" eulerseq="xyz" autolimits="true"/>
    <option gravity="0 0 -9.81" timestep="0.01" iterations="10"/>
    <custom>
        <numeric data="10" name="constraint_ang_damping"/> <!-- positional & spring -->
        <numeric data="1" name="spring_inertia_scale"/>  <!-- positional & spring -->
        <numeric data="0" name="ang_damping"/>  <!-- positional & spring -->
        <numeric data="0" name="spring_mass_scale"/>  <!-- positional & spring -->
        <numeric data="0.5" name="joint_scale_pos"/> <!-- positional -->
        <numeric data="0.1" name="joint_scale_ang"/> <!-- positional -->
        <numeric data="3000" name="constraint_stiffness"/>  <!-- spring -->
        <numeric data="10000" name="constraint_limit_stiffness"/>  <!-- spring -->
        <numeric data="50" name="constraint_vel_damping"/>  <!-- spring -->
        <numeric data="10" name="solver_maxls"/>  <!-- generalized -->
    </custom>

    <asset>
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
    </asset>

    <default>
        <geom contype="0" friction="1 0.1 0.1" material="geom"/>
    </default>

    <worldbody>
        <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
        <geom name="table" type="plane" pos="0 0.0 -0.1" size="1 1 0.1" contype="8" conaffinity="11" condim="3"/>
        <body name="disk" pos="0.0 0.0 0.0" euler="1.5708 0.0 0.0">
            <joint name="hinge_joint" type="hinge" axis="0 0 1" range="-180 180" armature="0.00022993" damping="0.0001" limited="false"/>
            <geom name="disk_geom" type="cylinder" size="0.06 0.001" contype="0" conaffinity="0" condim="3" mass="0.0"/>
            <geom name="mass_geom" type="cylinder" size="0.02 0.005" contype="0" conaffinity="0"  condim="3" rgba="0.04 0.04 0.04 1"
                  pos="0.0 0.04 0." mass="0.05085817"/>
        </body>
    </worldbody>

    <actuator>
        <motor joint="hinge_joint" ctrllimited="false" ctrlrange="-3.0 3.0"  gear="0.01"/>
    </actuator>
</mujoco>
"""


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
    th_rollout = rollout.nodes["agent"].inputs["sensor"][:, -1].data.th
    thdot_rollout = rollout.nodes["agent"].inputs["sensor"][:, -1].data.thdot

    # Determine fps
    dt = rollout.nodes["agent"].ts[-1] / rollout.nodes["agent"].ts.shape[-1]

    # Initialize system
    sys = mjcf.loads(DISK_PENDULUM_XML)
    sys = sys.replace(dt=dt)

    def _set_pipeline_state(th, thdot):
        qpos = sys.init_q.at[0].set(th)
        qvel = jnp.array([thdot])
        pipeline_state = gen_pipeline.init(sys, qpos, qvel)
        return pipeline_state

    pipeline_state_rollout = jax.vmap(_set_pipeline_state)(th_rollout, thdot_rollout)
    pipeline_state_lst = []
    for i in range(th_rollout.shape[0]):
        pipeline_state_i = jax.tree_util.tree_map(lambda x: x[i], pipeline_state_rollout)
        pipeline_state_lst.append(pipeline_state_i)
    rollout_json = html.render(sys, pipeline_state_lst)
    return rollout_json
