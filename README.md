# DreamWaQ paper implementation (Forked from legged_gym)

This repository is a fork of the original [legged_gym](https://github.com/leggedrobotics/legged_gym) repository, providing the implementation of the [DreamWaQ](https://arxiv.org/abs/2301.10602) paper. Below are the specific changes made in this fork:

## Changes Made
### Beta VAE implementation
- Implemented the Beta VAE as per the paper within the 'rsl_rl' folder. The modifications involve updating the 'actor_critic.py' file for the encoder and decoder implementation, and incorporating the VAE losses in the 'ppo.py' file.

### Observation History Wrapper
- Added a history wrapper to incorporate observation history.

### Added Go1 Robot Config
- As we had a Unitree Go1 robot to deploy on, added a Go1 Robot Config.

### Deploy Policy using Walk These Ways Repo
- Utilized the [Walk These Ways](https://github.com/Improbable-AI/walk-these-ways) repository for deploying the policy.

## TODO
- Implement the Adaboot part of the DreamWaQ paper.
- Specify the beta value in the Beta VAE implementation; currently set to 1.0.

## Installation
This repository builds upon [Legged Gym](https://github.com/leggedrobotics/legged_gym). Follow their installation instructions to set up the required environment. Additional dependencies specific to this implementation will be detailed in the `requirements.txt` file.
