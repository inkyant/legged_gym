

from isaacgym.torch_utils import quat_rotate_inverse
from legged_gym.envs.b1.b1_config import B1RoughCfg
from legged_gym.envs.base.legged_robot import LeggedRobot
import torch


class B1(LeggedRobot):
    cfg : B1RoughCfg
    def __init__(self, cfg: B1RoughCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.dof_pos_history = torch.zeros(self.num_envs, self.num_dof * 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.before_last_dof_vel = torch.zeros_like(self.dof_vel)


    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.before_last_dof_vel[:] = self.last_dof_vel[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        self.dof_pos_history[:, :2*self.num_dof] = self.dof_pos_history[:, self.num_dof:]
        self.dof_pos_history[:, 2*self.num_dof:] = self.dof_pos[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def compute_observations(self):
        """ Computes observations for the b1
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.last_actions,
                                    (self.dof_pos_history - self.default_dof_pos.repeat(1, 3)) * self.obs_scales.dof_pos,
                                    self.last_dof_vel * self.obs_scales.dof_vel,
                                    self.before_last_dof_vel * self.obs_scales.dof_vel
                                    ), dim=-1)
        
        # body velocity 6
        # body orientation 3
        # command 3
        # joint position 12
        # joint velocity 12
        # joint target history (2 time steps) 24
        # joint position history (3 time steps) 36
        # joint velocity history (2 time steps) 24
    
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.before_last_dof_vel[env_ids] = 0.
        self.dof_pos_history[env_ids] = 0.