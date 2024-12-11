import sys
import os
import argparse
import numpy as np

from stable_baselines3 import PPO

from uitb import Simulator

if __name__ == "__main__":
    # parsing
    parser = argparse.ArgumentParser(description="Collecting Observations")
    parser.add_argument("--num_episodes", type=int, default=100, help="Average steps per episode = 130")
    
    args = parser.parse_args()


    # create folder structure
    save_path = 'mm_testing/dataset/'

    if not os.path.isdir(save_path):
        os.makedirs(save_path)


    # build pointing simulator
    config_file = 'uitb/configs/mobl_arms_index_pointing.yaml'
    simulator_folder = Simulator.build(config_file)

    simulator = Simulator.get(simulator_folder=simulator_folder)

    simulator.reset()

    # loading saved policy - must ensure training has taken place
    # model_path = 'simulators/mobl_arms_index_pointing_original/checkpoints/model_95000000_steps.zip' # from Aleksi repo
    model_path = 'simulators/mobl_arms_index_pointing/checkpoints/model_100000000_steps.zip' # reproduced

    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    # datasets
    vision_frames = []
    proprioception_frames = []

    # save observations on trained policy
    num_episodes = args.num_episodes
    steps = 0

    for episode_indx in range(num_episodes):
        print(f"Episode: {episode_indx+1} / {num_episodes}")

        # reset environment
        obs, _ = simulator.reset()
        terminated = False
        truncated = False
        reward = 0

        # loop until episode end
        while not terminated and not truncated:
            steps += 1

            action, _ = model.predict(obs, deterministic=False)

            obs, _, terminated, truncated, _ = simulator.step(action)

            # record and save observations
            vision_frames.append(obs['vision'][0])
            proprioception_frames.append(obs['proprioception'])


    #clean up simulator
    print(f'\nNumber of Steps: {steps}\n')
    simulator.close()

    # saving
    np.save(save_path + 'vision_frames', vision_frames)
    np.save(save_path + 'proprioception_frames', proprioception_frames)