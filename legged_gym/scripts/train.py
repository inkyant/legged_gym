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

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import wandb
import shutil

def train(args):

    print("\nStarting training with exptid:", args.exptid)

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, log_root="/opt/isaacgym/output_files/dog_walk/")
    
    final_output_dir = f'/opt/isaacgym/output_files/dog_walk/{args.exptid}/final_export/'
    os.makedirs(final_output_dir, exist_ok=True)

    shutil.copyfile('/opt/isaacgym/legged_gym/legged_gym/envs/b1/b1_config.py', final_output_dir + 'b1_config.py')
    print("="*8,"\nREWARD SCALES:\n", env.reward_scales, "\n")
    
    wandb.init(project="b1-gym", entity="apfurman", config=env.reward_scales)
    
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True, wandb_project=args.exptid)
    shutil.copyfile(f'/opt/isaacgym/output_files/dog_walk/{args.exptid}/models/model_{train_cfg.runner.max_iterations}.pt', final_output_dir + f'model_{train_cfg.runner.max_iterations}.pt')

if __name__ == '__main__':
    args = get_args()
    train(args)
