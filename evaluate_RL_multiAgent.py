from ray.tune.registry import register_env
from models.FilteringEnv_multiAgent import FilteringEnv_multiAgent

from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.policy.policy import PolicySpec
import numpy as np


path_to_checkpoint = "saved_ex/SAC_MA_GreedyDe_00003_maxVol300/checkpoint_013750"
max_vol = 300

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return agent_id

def env_creator(env_config):
    filt_env = FilteringEnv_multiAgent(ts_path=env_config["ts_path"],
                                       path_data=env_config["path_data"],
                                       atc=env_config["atc"],
                                       bid_file=env_config["bid_file"],
                                       grouped_areas=env_config["grouped_areas"],
                                       bid_correspondance=env_config["bid_correspondance"],
                                       not_grouped_file=env_config["not_grouped_file"],
                                       alpha=env_config["alpha"],
                                       areas=env_config["areas"],
                                       borders_list=env_config["borders_list"],
                                       max_border_capacity=env_config["max_border_capacity"],
                                       price_normalization=env_config["price_normalization"],
                                       max_vol=env_config["max_vol"], get_var=env_config["get_var"],
                                       greedy_TSO= env_config["greedy_TSO"],
                                       local_welfare_TSO=env_config["local_welfare_TSO"],
                                       full_greedy_TSO=env_config["full_greedy_TSO"]
                                       )
    return filt_env

def evaluate(model_fr,model_de,env):
    sum_reward = {}

    for i in range(52):
        obs, info = env.reset()
        terminateds = {"fr":False,"de":False}
        truncateds = {"fr":False,"de":False}
        while not terminateds["fr"] and not truncateds["fr"]:
            action_fr = model_fr.compute_actions_from_input_dict(input_dict={"obs":np.array([obs["fr"]])}, explore=False)
            action_de = model_de.compute_actions_from_input_dict(input_dict={"obs":np.array([obs["de"]])}, explore=False)
            action = {"fr":action_fr[0][0], "de":action_de[0][0]}
            obs, reward, terminateds, truncateds, info = env.step(action)
            sum_reward = {k: sum_reward.get(k, 0) + reward.get(k, 0) for k in set(reward)}
        print("sum reward", sum_reward)
    print("total reward", sum_reward)

if __name__ == '__main__':
    env_test_config = {"name_g2op_env": "ieee_96_marie",
                  "ts_path": "./ieee_96_marie/raw_data/timeseries/",
                  "path_data": "data",
                  "atc": "_02022023_noPmin.csv",
                  "bid_file": "all_bids_grouped_de_fr.csv",
                  "not_grouped_file": "all_bids.csv",
                  "alpha": 1.,  # tradeoff between market and tso in social welfare
                  "grouped_areas": ["fr", "de"],
                  "bid_correspondance": "bid_correspondance_fr_de.csv",
                  "areas": {'be', 'de', 'fr'},
                  "borders_list": {'be_de', 'be_fr', 'de_fr'},
                  "max_border_capacity": {'be_de': 1175, 'be_fr': 500, 'de_fr': 500},
                  "price_normalization": 340,
                  "max_vol": max_vol,
                  "get_var": False,
                  "greedy_TSO": [],
                  "local_welfare_TSO":[],
                  "full_greedy_TSO":["de"]
                  }

    ray_ma_env = FilteringEnv_multiAgent(ts_path="./ieee_96_marie/raw_data/timeseries/")
    LR = "00003"
    NNsize = 100

    register_env("my_env_test", env_creator)
    test_config = (SACConfig()
        .rollouts(num_rollout_workers=1)  # 1)
        .resources(num_cpus_per_worker=1, num_learner_workers=1)
        .environment(env="my_env_test", env_config=env_test_config, auto_wrap_old_gym_envs=False,
                     disable_env_checking=True)
        .framework("torch")
        .training(lr=float("0." + LR), train_batch_size=256, gamma=0.9
                  , replay_buffer_config={"buffer_size": 100000},  # only for SAC
                  optimization_config={'actor_learning_rate': float("0." + LR),
                                       'critic_learning_rate': float("0." + LR),
                                       'entropy_learning_rate': float("0." + LR)})  # only for SAC
        .multi_agent(
        policies={
            "fr": PolicySpec(
                action_space=ray_ma_env.action_space["fr"],
                observation_space=ray_ma_env.observation_space["fr"]
            ),
            "de": PolicySpec(
                action_space=ray_ma_env.action_space["de"],
                observation_space=ray_ma_env.observation_space["de"]
            )
        },
        policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
    )
    )

    test_config["rollout_fragment_length"] = 5
    test_config["explore"]=False
    test_config["model"]["fcnet_hiddens"] = [int(NNsize)]*2
    test_config["q_model_config"]["fcnet_hiddens"] = [int(NNsize)]*2
    test_config["policy_model_config"]["fcnet_hiddens"] = [int(NNsize)]*2


    my_new_sac = test_config.build()

    # Restore the old (checkpointed) state.
    my_new_sac.restore(path_to_checkpoint)

    #print the config
    print(my_new_sac.get_config().items())
    test_env = env_creator(env_test_config)

    evaluate(my_new_sac.get_policy("fr"),my_new_sac.get_policy("de"), test_env)
