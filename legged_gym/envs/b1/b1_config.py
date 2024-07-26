
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

import numpy as np

class B1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.6] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': -0.0 ,  # [rad]
            'RR_hip_joint': -0.0,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 80.}  # [N*m/rad]
        damping = {'joint': 2}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/b1/urdf/b1.urdf'
        name = "b1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        # dof limit is lower for hips
        soft_dof_pos_limit = np.array([
            [0.7, 0.7], [0.9, 0.9], [0.9, 0.9],
            [0.7, 0.7], [0.9, 0.9], [0.9, 0.9],
            [0.7, 0.7], [0.9, 0.9], [0.9, 0.9],
            [0.7, 0.7], [0.9, 0.9], [0.9, 0.9],
        ])
        base_height_target = 0.5
        class scales( LeggedRobotCfg.rewards.scales ):
            # lin_vel_z, ang_vel_xy, orientation, base_height, torques, dof_vel, dof_acc, 
            # action_rate, collision, termination, dof_pos_limits, dof_vel_limits, 
            # torque_limits, tracking_lin_vel, tracking_ang_vel, feet_air_time, stumble, 
            # stand_still, feet_contact_forces

            # DEFAULTS:
            # {'action_rate': -0.01, 
            # 'ang_vel_xy': -0.05, 
            # 'collision': -1.0, 
            # 'dof_acc': -2.5e-07,
            # 'feet_air_time': 1.0, 
            # 'lin_vel_z': -2.0,
            # 'torques': -1e-05, 
            # 'tracking_ang_vel': 0.5, 
            # 'tracking_lin_vel': 1.0
            # }
            dof_pos_limits = -1.0
            pass

class B1RoughCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        max_iterations = 6000 # number of policy updates
        save_interval = 100 # check for potential saves every this many iterations
        run_name = ''
        experiment_name = 'rough_b1'

  