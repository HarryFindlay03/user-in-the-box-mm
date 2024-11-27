import sys
import os
import numpy as np

from uitb import Simulator


# create folder structure
save_path = 'mm_testing/dataset/'

if not os.path.isdir(save_path):
    os.makedirs(save_path)


# build pointing simulator
config_file = 'uitb/configs/mobl_arms_index_pointing.yaml'
simulator_folder = Simulator.build(config_file)

simulator = Simulator.get(simulator_folder=simulator_folder)

simulator.reset()

n_steps = 64000

# datasets
vision_frames = np.zeros((n_steps, 80, 120))
proprioception_frames = np.zeros((n_steps, 44))

for i in range(n_steps):
    if(i % (n_steps//10) == 0):
        print('#', end='', flush=True)

    # take random action
    simulator.step(simulator.action_space.sample())

    # get modality observations
    obs_dict = simulator.get_observation()
    vision_obs = obs_dict['vision'][0]
    proprioception_obs = obs_dict['proprioception']

    # test
    print(vision_obs)
    quit()

    # saving modality observations
    vision_frames[i] = vision_obs
    proprioception_frames[i] = proprioception_obs


#clean up simulator
print('\n')
simulator.close()

# saving
np.save(save_path + 'vision_frames', vision_frames)
np.save(save_path + 'proprioception_frames', proprioception_frames)