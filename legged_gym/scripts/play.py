# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import imageio
from isaacgym import gymapi
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

import numpy as np
from PIL import Image, ImageDraw, ImageFont

def add_text_to_image(image_array, text):
    """Add text to a numpy image array and return the updated numpy array."""

    image = Image.fromarray(image_array)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('Arial.ttf', size=35)

    position = (10, 10)
    padding = 5

    bbox = draw.textbbox(position, text, font=font)
    box_x0, box_y0, box_x1, box_y1 = bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding

    # Draw the background box
    draw.rectangle([box_x0, box_y0, box_x1, box_y1], fill=(0, 0, 0))

    draw.text(position, text, (255, 255, 255), font=font)
    return np.array(image)

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.env.headless_graphics = True
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join('/opt/isaacgym/output_files/dog_walk', args.exptid, 'export')
        checkpoint = train_cfg.runner.checkpoint if train_cfg.runner.checkpoint != -1 else 0
        export_policy_as_jit(ppo_runner.alg.actor_critic, path, f'policy_jit_{checkpoint}.pt')
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards

    img_idx = 0
    
    max_frames = 1000

    # actors for camera playback
    actor_idxs = [0, 10, 20, 30, 40]
    # cameras = []

    for actor_idx in actor_idxs:
        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(-5,0,5)
        local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(45.0))
        cam = env.gym.create_camera_sensor(env.envs[actor_idx], gymapi.CameraProperties())
        env.gym.attach_camera_to_body(cam, env.envs[actor_idx], env.actor_handles[actor_idx], local_transform, gymapi.FOLLOW_POSITION)
        # cameras.append(cam)

    if RECORD_FRAMES:
        env.graphics_device_id = env.sim_device_id
        # ffmpeg -f image2 -framerate 20 -i frames/%d.png -c:v libx264 -crf 22 export/video.mp4
        frames_path = os.path.join('/opt/isaacgym/output_files/dog_walk', args.exptid, 'frames')
        os.makedirs(frames_path, exist_ok=True)

    frames_per_actor = max_frames // len(actor_idxs)

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:

                env.gym.fetch_results(env.sim, True)
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)

                # all camera sensors are always rendered, so might as well make video by stitching 
                # the frames together.
                for i, actor_idx in enumerate(actor_idxs):
                    # for some reason it works without the camera handle if you just put 0
                    image = env.gym.get_camera_image(env.sim, env.envs[actor_idx], 0, gymapi.IMAGE_COLOR)
                    image = image.reshape(image.shape[0], -1, 4)[..., :3]
                    text = f"Command:\n   Vel X: {env.commands[actor_idx, 0]:.2f}\n   Vel Y: {env.commands[actor_idx, 1]:.2f}\n   Angular Vel: {env.commands[actor_idx, 2]  * 180 / np.pi:.2f}\n   Heading: {env.commands[actor_idx, 3] * 180 / np.pi:.2f} deg \
                             \nVel X Error: {env.commands[actor_idx, 0] - env.base_lin_vel[actor_idx, 0]:.2f}\nVel Y Error: {(env.commands[actor_idx, 1] - env.base_lin_vel[actor_idx, 1]):.2f}\nAngular Vel Error: {env.commands[actor_idx, 2] - env.base_ang_vel[actor_idx, 2]:.2f}"
                    image = add_text_to_image(image, text)
                    filename = os.path.join(frames_path, f"{img_idx + i*frames_per_actor}.png")
                    imageio.imwrite(filename, image)

                print(f"\rsaved {img_idx*100 / frames_per_actor:.2f}% of frames", end='')
                img_idx += 1 
                if img_idx >= frames_per_actor:
                    print()
                    break

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            # logger.plot_states()
            pass
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = True
    args = get_args()
    play(args)
