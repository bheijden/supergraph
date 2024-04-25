from typing import Callable, Any
import jax
import jax.experimental.host_callback as host_callback
import jax.numpy as jnp
import numpy as onp

import mujoco
import mujoco.viewer
from mujoco import mjx

import supergraph.compiler.crazyflie as cf


class MujocoViewer:
    def __init__(self, xml_path: str, show_left_ui: bool = False, show_right_ui: bool = False):
        # Ui settings
        self._show_left_ui = show_left_ui
        self._show_right_ui = show_right_ui

        # Load system
        self._xml_path = xml_path
        self._mj_m = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_d = mujoco.MjData(self._mj_m)
        self._mjx_m = mjx.device_put(self._mj_m)
        self._mjx_d = mjx.device_put(self._mj_d)
        self._viewer = None
        self.open()

    def update(self, qpos: jax.Array):
        mjx_d = self._mjx_d.replace(qpos=qpos)
        mjx_d = mjx.forward(self._mjx_m, mjx_d)
        _ = host_callback.call(self._sync_viewer, mjx_d, result_shape=jax.ShapeDtypeStruct((), onp.float32))
        return mjx_d

    def close(self):
        self._viewer.close()
        self._viewer = None

    def open(self):
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self._mj_m, self._mj_d, key_callback=self._key_callback,
                                                        show_left_ui=self._show_left_ui, show_right_ui=self._show_right_ui)
            self._paused = False

    def _sync_viewer(self, mjx_d):
        self.open()
        mjx.device_get_into(self._mj_d, mjx_d)
        if not self._paused:
            self._viewer.sync()
        return 1.

    def _key_callback(self, keycode):
        if chr(keycode) == ' ':
            self._paused = not self._paused

    # def _get_qpos(self, viewer_output: ViewerOutput) -> jax.Array:
    #     boxyaw = jnp.array([viewer_output.boxsensor.wrapped_yaw])
    #     boxpos = viewer_output.boxsensor.boxpos
    #     goalpos = viewer_output.supervisor.goalpos
    #     goalyaw = viewer_output.supervisor.goalyaw
    #     jpos = viewer_output.armsensor.jpos
    #     qpos = jnp.concatenate([boxpos, boxyaw, goalpos, jpos, jnp.array([0])])
    #     # jax_print("qpos={qpos}", qpos=qpos)
    #     return qpos


if __name__ == "__main__":
    xml_path = "/home/r2ci/supergraph/supergraph/compiler/crazyflie/cf2_lightweight.xml"
    show_left_ui = False
    show_right_ui = False
    viewer = MujocoViewer(xml_path, show_left_ui, show_right_ui)

    # Call the viewer update function with the qpos
    def get_qpos(pos, att, pos_plat, att_plat):
        cf_quat = cf.nodes.rpy_to_wxyz(att)
        cf_qpos = jnp.concatenate([pos, cf_quat, jnp.array([0])])

        plat_quat = cf.nodes.rpy_to_wxyz(att_plat)
        pitch_plat = jnp.array([0])
        platform_qpos = jnp.concatenate([pos_plat, plat_quat, pitch_plat])

        # Set corrected platform state
        polar_cor, azimuth_cor = cf.nodes.rpy_to_spherical(att_plat)
        att_cor = jnp.array([0., 0, azimuth_cor])
        quat_cor = cf.nodes.rpy_to_wxyz(att_cor)
        pitch_cor = jnp.array([polar_cor])
        platform_cor = jnp.concatenate([pos_plat, quat_cor, pitch_cor])
        # qpos = jnp.concatenate([platform_qpos, platform_cor, cf_qpos])
        qpos = jnp.concatenate([platform_cor, cf_qpos])
        return qpos

    def update_viewer(pos, att, pos_plat, att_plat):
        qpos = get_qpos(pos, att, pos_plat, att_plat)
        viewer.update(qpos)

    # Data
    num_samples = 50
    pos_offset = jnp.array([0., 0., 0.])  # Offset of cf position in local frame (mocap vs bottom of cf)
    pos = jnp.zeros((num_samples, 3), dtype=float)
    pos = pos.at[:, 2].set(jnp.linspace(0.35, -0.00))
    att = jnp.zeros((num_samples, 3), dtype=float)
    pos_plat = jnp.zeros((num_samples, 3), dtype=float)
    # pos_plat = pos_plat.at[:, 0].set(jnp.linspace(0., -1., num_samples))
    # pos_plat = pos_plat.at[:, 1].set(jnp.linspace(0., -1., num_samples))
    att_plat = jnp.zeros((num_samples, 3), dtype=float)
    # att_plat = att_plat.at[:, 0].set(jnp.linspace(0., onp.pi / 5))
    # att_plat = att_plat.at[:, 1].set(jnp.linspace(0., onp.pi / 4, num_samples))
    # att_plat = att_plat.at[:, 2].set(jnp.linspace(0., onp.pi, num_samples))

    jit_update_viewer = jax.jit(update_viewer)
    import time
    dt = 1/20
    i = 0
    start = time.time()
    while True:
        jit_update_viewer(pos[i], att[i], pos_plat[i], att_plat[i])
        if i == (num_samples - 1):
            print("Resetting")
            time.sleep(5.0)
        i = (i + 1) % num_samples
        end = time.time()
        elapsed = end - start
        sleep_time = max(0., dt - elapsed)
        time.sleep(sleep_time)
        start = end + sleep_time
