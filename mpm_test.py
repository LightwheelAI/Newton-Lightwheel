# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################

import sys

import numpy as np
import torch
import warp as wp

import newton
import newton.examples
import newton.utils
from newton.solvers import SolverImplicitMPM
from pathlib import Path
from pxr import Usd, UsdGeom
from import_usd_costume import parse_usd_particles, parse_usd_scene, transform_points, parse_world_matrix

lab_to_mujoco = [9, 3, 6, 0, 10, 4, 7, 1, 11, 5, 8, 2]
mujoco_to_lab = [3, 7, 11, 1, 5, 9, 2, 6, 10, 0, 4, 8]


@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion along the last dimension of q and v.    Args:
    q: The quaternion in (x, y, z, w). Shape is (..., 4).
    v: The vector in (x, y, z). Shape is (..., 3).    Returns:
    The rotated vector in (x, y, z). Shape is (..., 3).
    """
    q_w = q[..., 3]  # w component is at index 3 for XYZW format
    q_vec = q[..., :3]  # xyz components are at indices 0, 1, 2
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    # for two-dimensional tensors, bmm is faster than einsum
    if q_vec.dim() == 2:
        c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    else:
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
    return a - b + c


def compute_obs(actions, state: newton.State, joint_pos_initial, indices, gravity_vec, command):
    q = wp.to_torch(state.joint_q)
    qd = wp.to_torch(state.joint_qd)
    root_quat_w = q[3:7].unsqueeze(0)
    root_lin_vel_w = qd[3:6].unsqueeze(0)
    root_ang_vel_w = qd[:3].unsqueeze(0)
    joint_pos_current = q[7:].unsqueeze(0)
    joint_vel_current = qd[6:].unsqueeze(0)
    vel_b = quat_rotate_inverse(root_quat_w, root_lin_vel_w)
    a_vel_b = quat_rotate_inverse(root_quat_w, root_ang_vel_w)
    grav = quat_rotate_inverse(root_quat_w, gravity_vec)
    joint_pos_rel = joint_pos_current - joint_pos_initial
    joint_vel_rel = joint_vel_current
    rearranged_joint_pos_rel = torch.index_select(joint_pos_rel, 1, indices)
    rearranged_joint_vel_rel = torch.index_select(joint_vel_rel, 1, indices)
    obs = torch.cat([vel_b, a_vel_b, grav, command, rearranged_joint_pos_rel, rearranged_joint_vel_rel, actions], dim=1)
    return obs


@wp.kernel
def update_collider_mesh(
    src_points: wp.array(dtype=wp.vec3),
    src_shape: wp.array(dtype=int),
    res_mesh: wp.uint64,
    shape_transforms: wp.array(dtype=wp.transform),
    shape_body_id: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    dt: float,
):
    v = wp.tid()
    res = wp.mesh_get(res_mesh)

    shape_id = src_shape[v]
    p = wp.transform_point(shape_transforms[shape_id], src_points[v])

    X_wb = body_q[shape_body_id[shape_id]]

    cur_p = res.points[v] + dt * res.velocities[v]
    next_p = wp.transform_point(X_wb, p)
    res.velocities[v] = (next_p - cur_p) / dt
    res.points[v] = cur_p


class Example:
    def __init__(
        self,
        viewer,
        voxel_size=0.05,
        particles_per_cell=3,
        tolerance=1.0e-5,
        sand_friction=0.48,
        # dynamic_grid=True,
    ):
        # setup simulation parameters first
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        self.device = wp.get_device()

        # import the robot model
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 1.2

        asset_path = newton.utils.download_asset("anybotics_anymal_c")
        builder.add_urdf(
            str(asset_path / "urdf" / "anymal.urdf"),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )
        # builder.add_ground_plane()

        # setup robot joint properties
        builder.joint_q[:3] = [0.0, 0.0, 0.62]
        builder.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]
        builder.joint_q[7:] = [
            0.0,
            -0.4,
            0.8,
            0.0,
            -0.4,
            0.8,
            0.0,
            0.4,
            -0.8,
            0.0,
            0.4,
            -0.8,
        ]
        for i in range(len(builder.joint_dof_mode)):
            builder.joint_dof_mode[i] = newton.JointMode.TARGET_POSITION
        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = 150
            builder.joint_target_kd[i] = 5

        stage = Usd.Stage.Open(str((Path(__file__).parent /"scene.usda").resolve()))


        

        # particle_offset = wp.vec3(0.5, 2, 0.5)
        world_offset = wp.vec3(0.0, 4.0, 0.35)
        # world_offset = wp.vec3(0.0, 0.0, 0.0)
        particle_offset = wp.vec3(0.0, 0.0, 0.42)
        global_scale = 1.0
        # parse particles
        max_fraction = 1.0
        sand_particles = parse_usd_particles(builder, "sand", stage, global_scale=1.0, global_translate=world_offset + particle_offset)
        mud_particles = parse_usd_particles(builder, "mud", stage, global_scale=1.0, global_translate=world_offset + particle_offset)
        snow_particles = parse_usd_particles(builder, "snow", stage, global_scale=1.0, global_translate=world_offset + particle_offset)
        # breakpoint()

        sand_yield_pressure = np.array(sand_particles["mpm_material"]["particle_material_yieldPressure"], dtype=float)
        sand_yield_stress = np.array(sand_particles["mpm_material"]["particle_material_yieldStress"], dtype=float)
        sand_tensile_yield_ratio = np.array(sand_particles["mpm_material"]["particle_material_tensileYieldRatio"], dtype=float)
        sand_friction_array = np.array(sand_particles["mpm_material"]["particle_material_friction"], dtype=float)
        sand_hardening = np.array(sand_particles["mpm_material"]["particle_material_hardening"], dtype=float)
        sand_color = np.full((sand_particles["pt_count"],3), [0.7, 0.6, 0.4])
        # test = wp.array(sand_color, dtype=wp.vec3)
        # breakpoint()
        # sand_color = wp.full(shape=self.model.particle_count, value=wp.vec3(0.75, 0.75, 0.8), device=self.device)

        mud_yield_pressure = np.array(mud_particles["mpm_material"]["particle_material_yieldPressure"], dtype=float)
        mud_yield_stress = np.array(mud_particles["mpm_material"]["particle_material_yieldStress"], dtype=float)
        mud_tensile_yield_ratio = np.array(mud_particles["mpm_material"]["particle_material_tensileYieldRatio"], dtype=float)
        mud_friction_array = np.array(mud_particles["mpm_material"]["particle_material_friction"], dtype=float)
        mud_hardening = np.array(mud_particles["mpm_material"]["particle_material_hardening"], dtype=float)
        mud_color = np.full((mud_particles["pt_count"],3), [0.4, 0.25, 0.25])

        snow_yield_pressure = np.array(snow_particles["mpm_material"]["particle_material_yieldPressure"], dtype=float)
        snow_yield_stress = np.array(snow_particles["mpm_material"]["particle_material_yieldStress"], dtype=float)
        snow_tensile_yield_ratio = np.array(snow_particles["mpm_material"]["particle_material_tensileYieldRatio"], dtype=float)
        snow_friction_array = np.array(snow_particles["mpm_material"]["particle_material_friction"], dtype=float)
        snow_hardening = np.array(snow_particles["mpm_material"]["particle_material_hardening"], dtype=float)
        snow_color = np.full((snow_particles["pt_count"],3), [0.75, 0.75, 0.8])


        yp = np.concatenate([sand_yield_pressure, mud_yield_pressure, snow_yield_pressure], axis = 0)
        ys = np.concatenate([sand_yield_stress, mud_yield_stress, snow_yield_stress], axis = 0)
        tyr = np.concatenate([sand_tensile_yield_ratio, mud_tensile_yield_ratio, snow_tensile_yield_ratio], axis = 0)
        frictions = np.concatenate([sand_friction_array, mud_friction_array, snow_friction_array], axis = 0)
        hardening = np.concatenate([sand_hardening, mud_hardening, snow_hardening], axis = 0)
        colors = np.concatenate([sand_color, mud_color, snow_color], axis = 0)
        # parse scene 
        scene = parse_usd_scene(builder, stage, global_scale=global_scale, global_translate=world_offset, use_static_collider=True, skip_mesh_approximation=False)

        # finalize model
        self.model = builder.finalize()
        self.model.particle_mu = sand_friction
        self.model.particle_ke = 1.0e15

        # setup mpm solver
        mpm_options = SolverImplicitMPM.Options()
        mpm_options.voxel_size = voxel_size
        mpm_options.tolerance = tolerance
        mpm_options.transfer_scheme = "pic"
        mpm_options.grid_type = "sparse"
        mpm_options.strain_basis = "P0"
        mpm_options.max_iterations = 50

        # global defaults
        mpm_options.hardening = 0.0
        mpm_options.critical_fraction = 0.0
        mpm_options.air_drag = 1.0


        # mpm model
        mpm_model = SolverImplicitMPM.Model(self.model, mpm_options)

        mpm_model.material_parameters.yield_pressure = wp.array(yp, dtype=float)
        mpm_model.material_parameters.yield_stress = wp.array(ys, dtype=float)
        mpm_model.material_parameters.tensile_yield_ratio = wp.array(tyr, dtype=float)
        mpm_model.material_parameters.friction = wp.array(frictions, dtype=float)
        mpm_model.material_parameters.hardening = wp.array(hardening, dtype=float)
        # breakpoint()

        mpm_model.notify_particle_material_changed()

        self.model.particle_colors = wp.array(colors, dtype=wp.vec3)

        # Select and merge meshes for robot/sand collisions
        collider_body_idx = [idx for idx, key in enumerate(builder.body_key) if "SHANK" in key]
        # collider_shape_ids = np.concatenate(
        #     [[m for m in self.model.body_shapes[b] if self.model.shape_source[m]] for b in collider_body_idx]
        # )
        
        collider_body_shape_ids = [[m for m in self.model.body_shapes[b] if self.model.shape_source[m]] for b in collider_body_idx]
        print("collider_body_shape_ids:", collider_body_shape_ids)
        
        collider_scene_shape_ids = [[
           idx for idx, key in enumerate(builder.shape_key) if "SHAPE_" in key
        ]]
        print("collider_scene_shape_ids:", collider_scene_shape_ids)

        collider_shape_ids = np.concatenate(collider_body_shape_ids + collider_scene_shape_ids)
        robot_collider_shape_ids = np.array(collider_body_shape_ids).flatten().tolist()
        # breakpoint()

        collider_points, collider_indices, collider_v_shape_ids = _merge_meshes(
            [self.model.shape_source[m].vertices for m in collider_shape_ids],
            [self.model.shape_source[m].indices for m in collider_shape_ids],
            [self.model.shape_scale.numpy()[m] for m in collider_shape_ids],
            collider_shape_ids,
        )

        rob_collider_points, rob_collider_indices, rob_collider_v_shape_ids = _merge_meshes(
            [self.model.shape_source[m].vertices for m in robot_collider_shape_ids],
            [self.model.shape_source[m].indices for m in robot_collider_shape_ids],
            [self.model.shape_scale.numpy()[m] for m in robot_collider_shape_ids],
            robot_collider_shape_ids,
        )

        self.collider_mesh = wp.Mesh(wp.clone(collider_points), collider_indices, wp.zeros_like(collider_points))
        self.collider_rest_points = collider_points
        self.collider_shape_ids = wp.array(collider_v_shape_ids, dtype=int)

        self.robot_collider_mesh = wp.Mesh(wp.clone(rob_collider_points), rob_collider_indices, wp.zeros_like(rob_collider_points))
        self.robot_collider_rest_points = rob_collider_points
        self.robot_collider_shape_ids = wp.array(rob_collider_v_shape_ids, dtype=int)

        collider_array = []
        collider_friction_array = []
        collider_adhesion_array = []

        # for i in scene :
        #     current_object = scene[i]
        #     current_points = wp.array(current_object["points"], dtype=wp.vec3)
        #     current_indices = wp.array(current_object["faceVertexIndices"], dtype=int)
        #     current_mesh = wp.Mesh(current_points, current_indices, wp.zeros_like(current_points))
        #     collider_array.append(current_mesh)
        #     collider_friction_array.append(0.5)
        #     collider_adhesion_array.append(0.0e6)

        # collider_array.append(self.robot_collider_mesh)
        # collider_friction_array.append(0.5)
        # collider_adhesion_array.append(0.0e6)
        # setup collider with locomotion terrain collision mesh from houdini
        # new_stage = Usd.Stage.Open(str((Path(__file__).parent / "scene.usda").resolve()))

        prim = stage.GetPrimAtPath("/World/Locomotion_Terrain_collision_mesh/Mesh")
        collision_mesh = UsdGeom.Mesh(prim)

        collision_mesh_points = np.array(collision_mesh.GetPointsAttr().Get(), dtype=float)
        world_mat = parse_world_matrix(prim)
        world_points = transform_points(world_mat, collision_mesh_points, global_scale, world_offset)
        collision_mesh_points = wp.array(world_points, dtype = wp.vec3)
        collision_mesh_indices = wp.array(collision_mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        collision_wp_mesh = wp.Mesh(collision_mesh_points, collision_mesh_indices, wp.zeros_like(collision_mesh_points))

        collider_array.append(collision_wp_mesh)
        collider_friction_array.append(0.5)
        collider_adhesion_array.append(0.0e6)

        collider_array.append(self.robot_collider_mesh)
        collider_friction_array.append(0.5)
        collider_adhesion_array.append(0.0e6)


        mpm_model.setup_collider(
            colliders=collider_array, 
            collider_friction=collider_friction_array, 
            collider_adhesion=collider_adhesion_array,
            ground_height = -10.0
        )
        
        # def __output_collider_mesh_to_json(mesh: wp.Mesh, file_path: str) :
        #     import json
        #     pts = mesh._points
        #     idx = mesh.indices

        #     pts = pts.list()
        #     idx = idx.list()

        #     data = {}
        #     data["vertices"] = []
        #     data["indices"] = []
        #     for i in pts :
        #         data["vertices"].append(i[0])
        #         data["vertices"].append(i[1])
        #         data["vertices"].append(i[2])
        #     for i in idx :
        #         data["indices"].append(int(i))

        #     with open(file_path, "w", encoding = "utf-8") as f :
        #         json.dump(data, f, indent=4, ensure_ascii=False)

        # __output_collider_mesh_to_json(collision_wp_mesh, "C:/Users/legen/geo/collider_mesh.json")
        # __output_collider_mesh_to_json(self.collider_mesh, "C:/Users/legen/geo/robot_collider_mesh.json")
        # breakpoint()


        self.solver = newton.solvers.SolverMuJoCo(self.model)
        self.mpm_solver = SolverImplicitMPM(mpm_model, mpm_options)
        

        # simulation state
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.mpm_solver.enrich_state(self.state_0)
        self.mpm_solver.enrich_state(self.state_1)

        self.mpm_solver.project_outside(self.state_0, self.state_0, dt=0.0, max_dist=5.0)

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self._update_collider_mesh(self.state_0)
        self._update_robot_collider_mesh(self.state_0)

        # Setup control policy
        self.control = self.model.control()

        q0 = wp.to_torch(self.state_0.joint_q)
        self.torch_device = q0.device
        self.joint_pos_initial = q0[7:].unsqueeze(0).detach().clone()
        self.act = torch.zeros(1, 12, device=self.torch_device, dtype=torch.float32)
        self.rearranged_act = torch.zeros(1, 12, device=self.torch_device, dtype=torch.float32)

        # Download the policy from the newton-assets repository
        policy_path = str(asset_path / "rl_policies" / "anymal_walking_policy_physx.pt")
        self.policy = torch.jit.load(policy_path, map_location=self.torch_device)

        # Pre-compute tensors that don't change during simulation
        self.lab_to_mujoco_indices = torch.tensor(
            [lab_to_mujoco[i] for i in range(len(lab_to_mujoco))], device=self.torch_device
        )
        self.mujoco_to_lab_indices = torch.tensor(
            [mujoco_to_lab[i] for i in range(len(mujoco_to_lab))], device=self.torch_device
        )
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        self.command = torch.zeros((1, 3), device=self.torch_device, dtype=torch.float32)
        self.command[0, 0] = 1

        self._reset_key_prev = False
        self._auto_forward = False

        # set model on viewer and setup capture
        self.viewer.set_model(self.model)
        self.viewer.camera.pos = wp.vec3(0.06, -5.57, 4.21)
        self.viewer.camera.pitch = -29.5
        self.viewer.camera.yaw = -266.8
        self.viewer.show_particles = True
        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate_robot()
            self.graph = capture.graph

    def apply_control(self):
        obs = compute_obs(
            self.act,
            self.state_0,
            self.joint_pos_initial,
            self.lab_to_mujoco_indices,
            self.gravity_vec,
            self.command,
        )
        with torch.no_grad():
            self.act = self.policy(obs)
            self.rearranged_act = torch.gather(self.act, 1, self.mujoco_to_lab_indices.unsqueeze(0))
            a = self.joint_pos_initial + 0.5 * self.rearranged_act
            a_with_zeros = torch.cat([torch.zeros(6, device=self.torch_device, dtype=torch.float32), a.squeeze(0)])
            a_wp = wp.from_torch(a_with_zeros, dtype=wp.float32, requires_grad=False)
            # copy action targets to control buffer
            wp.copy(self.control.joint_target, a_wp)

    def simulate_robot(self):
        # robot substeps
        # self.contacts = self.model.collide(self.state_0, rigid_contact_margin=0.1)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, contacts=None, dt=self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate_sand(self):
        # sand step (in-place on frame dt)
        self._update_collider_mesh(self.state_0)
        self._update_robot_collider_mesh(self.state_0)
        self.mpm_solver.step(self.state_0, self.state_0, contacts=None, control=None, dt=self.frame_dt)

    def step(self):
        # compute control before graph/step
        if hasattr(self.viewer, "is_key_down"):
            fwd = 1.0 if self.viewer.is_key_down("i") else (-1.0 if self.viewer.is_key_down("k") else 0.0)
            lat = 0.5 if self.viewer.is_key_down("j") else (-0.5 if self.viewer.is_key_down("l") else 0.0)
            rot = 1.0 if self.viewer.is_key_down("u") else (-1.0 if self.viewer.is_key_down("o") else 0.0)

            if fwd or lat or rot:
                # disable forward motion
                self._auto_forward = False

            self.command[0, 0] = float(fwd)
            self.command[0, 1] = float(lat)
            self.command[0, 2] = float(rot)
        if self._auto_forward:
            self.command[0, 0] = 1
        self.apply_control()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate_robot()

        # MPM solver step is not graph-capturable yet
        self.simulate_sand()

        self.sim_time += self.frame_dt

    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def _update_collider_mesh(self, state):
        wp.launch(
            update_collider_mesh,
            dim=self.collider_rest_points.shape[0],
            inputs=[
                self.collider_rest_points,
                self.collider_shape_ids,
                self.collider_mesh.id,
                self.model.shape_transform,
                self.model.shape_body,
                state.body_q,
                self.frame_dt,
            ],
        )
        self.collider_mesh.refit()

    def _update_robot_collider_mesh(self, state):
        wp.launch(
            update_collider_mesh,
            dim=self.robot_collider_rest_points.shape[0],
            inputs=[
                self.robot_collider_rest_points,
                self.robot_collider_shape_ids,
                self.robot_collider_mesh.id,
                self.model.shape_transform,
                self.model.shape_body,
                state.body_q,
                self.frame_dt,
            ],
        )
        self.robot_collider_mesh.refit()


def _spawn_particles(builder: newton.ModelBuilder, res, bounds_lo, bounds_hi, density):
    Nx = res[0]
    Ny = res[1]
    Nz = res[2]

    px = np.linspace(bounds_lo[0], bounds_hi[0], Nx + 1)
    py = np.linspace(bounds_lo[1], bounds_hi[1], Ny + 1)
    pz = np.linspace(bounds_lo[2], bounds_hi[2], Nz + 1)

    points = np.stack(np.meshgrid(px, py, pz)).reshape(3, -1).T

    cell_size = (bounds_hi - bounds_lo) / res
    cell_volume = np.prod(cell_size)

    radius = np.max(cell_size) * 0.5
    mass = np.prod(cell_volume) * density

    rng = np.random.default_rng()
    points += 2.0 * radius * (rng.random(points.shape) - 0.5)
    vel = np.zeros_like(points)

    builder.particle_q = points
    builder.particle_qd = vel

    builder.particle_mass = np.full(points.shape[0], mass)
    builder.particle_radius = np.full(points.shape[0], radius)
    builder.particle_flags = np.ones(points.shape[0], dtype=int)


def _merge_meshes(
    points: list[np.array],
    indices: list[np.array],
    scales: list[np.array],
    shape_ids: list[int],
):
    pt_count = np.array([len(pts) for pts in points])
    offsets = np.cumsum(pt_count) - pt_count

    mesh_id = np.repeat(np.arange(len(points), dtype=int), repeats=pt_count)

    merged_points = np.vstack([pts * scale for pts, scale in zip(points, scales, strict=False)])

    merged_indices = np.concatenate([idx + offsets[k] for k, idx in enumerate(indices)])

    return (
        wp.array(merged_points, dtype=wp.vec3),
        wp.array(merged_indices, dtype=int),
        wp.array(np.array(shape_ids)[mesh_id], dtype=int),
    )

if __name__ == "__main__":
    import argparse

    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()
    parser.add_argument("--voxel-size", "-dx", type=float, default=0.03)
    parser.add_argument("--particles-per-cell", "-ppc", type=float, default=3.0)
    parser.add_argument("--sand-friction", "-mu", type=float, default=0.48)
    parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-5)
    # parser.add_argument("--dynamic-grid", action=argparse.BooleanOptionalAction, default=True)

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # This example requires a GPU device
    if wp.get_device().is_cpu:
        print("Error: This example requires a GPU device.")
        sys.exit(1)

    # Create example and load policy
    example = Example(
        viewer,
        voxel_size=args.voxel_size,
        particles_per_cell=args.particles_per_cell,
        tolerance=args.tolerance,
        sand_friction=args.sand_friction,
        # dynamic_grid=args.dynamic_grid,
    )

    # Run via unified example runner
    newton.examples.run(example)

