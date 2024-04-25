import isaacgym
import torch
import glob
import pickle as pkl
import lcm
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.go1_gym_deploy.utils.deployment_runner import DeploymentRunner
from legged_gym.go1_gym_deploy.envs.lcm_agent import LCMAgent
from legged_gym.go1_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from legged_gym.go1_gym_deploy.utils.command_profile import *

from legged_gym.go1_gym_deploy.envs.history_wrapper import HistoryWrapper

from legged_gym.utils.helpers import  export_policy_as_jit_actor,export_policy_as_jit_encoder
import os
import pathlib
from legged_gym.envs.go1.go1_config import Go1RoughCfg
from legged_gym.rsl_rl.modules.actor_critic import ActorCritic
lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

def load_and_run_policy(label, experiment_name, max_vel=1.0, max_yaw_vel=1.0,args='--task=go1'):
    cfg=Go1RoughCfg
    #print(cfg)
    se = StateEstimator(lc)

    control_dt = 0.02
    command_profile = RCControllerProfile(dt=control_dt, state_estimator=se, x_scale=max_vel, y_scale=0.6, yaw_scale=max_yaw_vel)

    hardware_agent = LCMAgent(cfg, se, command_profile)
    se.spin()

    hardware_agent = HistoryWrapper(hardware_agent)

    policy = load_policy()

    # load runner
    root = f"{pathlib.Path(__file__).parent.resolve()}/../../logs/"
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    deployment_runner = DeploymentRunner(experiment_name=experiment_name, se=None,
                                         log_root=f"{root}/{experiment_name}")
    deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    deployment_runner.add_policy(policy)
    deployment_runner.add_command_profile(command_profile)

    if len(sys.argv) >= 2:
        max_steps = int(sys.argv[1])
    else:
        max_steps = 10000000
    print(f'max steps {max_steps}')

    deployment_runner.run(max_steps=max_steps, logging=True)

def load_policy():
 
    actor_critic = ActorCritic(45,241,5,12)
    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs','rough_go1', 'exported', 'policies')
    actor=export_policy_as_jit_actor(actor_critic, path)
    encoder=export_policy_as_jit_encoder(actor_critic, path)

     
    def policy(obs, info):
        i = 0
        mu,log_var = encoder.forward(obs["obs_history"].to('cpu'))
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        latent = mu + eps * std
        action = actor.forward(torch.cat((obs["obs"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


if __name__ == '__main__':
    label = "gait-conditioned-agility/pretrain-v0/train"

    experiment_name = "example_experiment"
    #args = get_args()
    load_and_run_policy(label, experiment_name=experiment_name, max_vel=3.5, max_yaw_vel=5.0,args='--task=go1')
