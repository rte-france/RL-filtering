# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
# This file is part of RL-filtering, a project for bid filtering in European balancing platforms.

import numpy as np

from sb3_contrib import TQC
from stable_baselines3.common.noise import NormalActionNoise
from models.FilteringEnv import FilteringEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import torch
import tensorflow as tf

torch.set_num_threads(2)

class EvalAtInitToo(EvalCallback):
    # eval callback before training too
    def _on_training_start(self):
        ncalls = self.n_calls
        self.n_calls = 0
        res = self._on_step()
        self.n_calls = ncalls
        return res


if __name__ == "__main__":
    max_vol = 200 #maximum redispatch volume

    # Create environment
    env = FilteringEnv_testDataParams(ts_path="objects/H-1_Y2_ATC_fullyear/", #env with training data
                                      path_data="data",
                                      atc="_30052023_Y2.csv",
                                      bid_file="all_bids_grouped_Y2.csv",
                                      not_grouped_file="all_bids_Y2.csv", max_vol=max_vol, get_var=False)

    LR = "00003"
    NNshape = "4"
    NNsize = "40"
    gamma = "0.9"
    tb_log_name = str(max_vol) #name of tensorboard file
    tb_log = "log_test" #name of tensorboard folder

    checkpoint_callback = CheckpointCallback( #intermediate save of model
        save_freq=10000,
        save_path="saved_ex/",
        name_prefix="TQC_LR_" + LR + "_NNsize_" + NNsize + "volLim" + str(max_vol), #name of file saved
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    file_writer = tf.summary.create_file_writer(tb_log + "/var_" + tb_log_name) #to get variance in tensorboard
    file_writer.set_as_default()

    # Instantiate the agent
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    eval_callback = EvalCallback(
        FilteringEnv_testDataParams(ts_path="ieee_96_marie/raw_data/timeseries/",  # env with testing data
                                    path_data="data",
                                    max_vol=max_vol, get_var=True),
        n_eval_episodes=52,
        eval_freq=25000, #evaluate over 52 weeks every 25000 training points
        deterministic=True,
    )

    eval_callback_init = EvalAtInitToo(
        FilteringEnv_testDataParams(ts_path="ieee_96_marie/raw_data/timeseries/",
                                    path_data="data", max_vol=max_vol, get_var=True),
        n_eval_episodes=52,
        eval_freq=25000,
        deterministic=True, )


    model = TQC("MlpPolicy", env, learning_rate=float("0." + LR), action_noise=action_noise, verbose=1,
                tensorboard_log=tb_log, train_freq=1, gamma=float(gamma), tau=0.02, top_quantiles_to_drop_per_net=5,
                use_sde=True, gradient_steps=1, batch_size=256, buffer_size=100000,
                policy_kwargs=dict(log_std_init=-3, activation_fn=torch.nn.ReLU, net_arch=[int(NNsize)] * int(NNshape)))

    # Train the agent and display a progress bar
    model.learn(total_timesteps=int(400000), tb_log_name=tb_log_name, progress_bar=True, log_interval=1,
                callback=[eval_callback, checkpoint_callback])

    # Save the agent when training is finished
    model.save("saved_ex/TQC_LR" + LR + "_NNsize_" + NNsize + "volLim" + str(max_vol))
