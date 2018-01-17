# EX2 Exloration with Exemplar Models for Deep Reinforcement Learning
This repository contains the code to run the experiments from this [paper](https://arxiv.org/abs/1703.01260).

## Installation
Dependecies
* [rllab](https://github.com/openai/rllab)
* Mujoco v1.31 
* OpenAI Gym using commit hash 518f4b
* [gym-doom](https://github.com/ppaquette/gym-doom)

Once rllab is installed run the following command in the root folder of rllab:
```bash
git submodule add -f git@github.com:jcoreyes/ex2.git
```

For doom, you need to copy ex2/ens/dooms/assets/my_way_home2.wad to
<doom_py-package-location>/scenarios/my_way_home2.wad.
.
## EC2 Setup
Except for doom, most experiments are meant to be run on EC2 but can also be run locally.
Follow these [instructions](https://rllab.readthedocs.io/en/latest/user/cluster.html) to setup
EC2. Furthermore the information in misc/ec2_info will need to configured according to your AWS security settings.

## Running experiments
Then you can run ex2 on the appropiate environment via `python ex2/exps/pointmass.py`. Change the mode variable in the script to 'ec2' to run on EC2.



