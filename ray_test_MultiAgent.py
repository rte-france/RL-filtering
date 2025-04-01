from models.FilteringEnv_multiAgent import FilteringEnv_multiAgent
from ray.rllib.policy.policy import PolicySpec, Policy
from ray.tune.logger import UnifiedLogger, TBXLogger
import ray
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.registry import register_env
from ray import tune
import shutil
import torch
import os

torch.set_num_threads(2)

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
                                       greedy_TSO=env_config["greedy_TSO"],
                                       TSO_compensation_penalized= env_config["TSO_compensation_penalized"],
                                       local_welfare_TSO=env_config["local_welfare_TSO"],
                                       full_greedy_TSO=env_config["full_greedy_TSO"])
    return filt_env

def custom_log_creator(custom_path):

    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        return TBXLogger(config, custom_path)

    return logger_creator

if __name__ == "__main__":

    # same parameters for train and evaluation except training is done on year2 data
    env_config = {"ts_path": "data/H-1_Y2_ATC_fullyear/",
                  "path_data": "data",
                  "atc": "_30052023_Y2.csv",
                  "bid_file": "all_bids_grouped_de_fr_Y2.csv",
                  "not_grouped_file": "all_bids_Y2.csv",
                  "alpha": 1.,  # tradeoff between market and tso in social welfare
                  "grouped_areas": ["fr", "de"],
                  "bid_correspondance": "bid_correspondance_fr_de.csv",
                  "areas": {'be', 'de', 'fr'},
                  "borders_list": {'be_de', 'be_fr', 'de_fr'},
                  "max_border_capacity": {'be_de': 1175, 'be_fr': 500, 'de_fr': 500},
                  "price_normalization": 340,
                  "max_vol": 300,
                  "get_var": False,
                  "greedy_TSO": [],
                  "TSO_compensation_penalized": [],
                  "local_welfare_TSO": ["de"],
                  "full_greedy_TSO":[]}

    env_eval_config = {"name_g2op_env": "ieee_96_marie",
                       "ts_path": "ieee_96_marie/raw_data/timeseries/",
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
                       "max_vol": 300,
                       "get_var": False,
                       "greedy_TSO": [],
                       "TSO_compensation_penalized": [],
                       "local_welfare_TSO": ["de"],
                       "full_greedy_TSO":[]}

    ray_ma_env = FilteringEnv_multiAgent(ts_path=env_config["ts_path"])

    ray.init(local_mode=True, num_cpus=2)  # , num_gpus=0, num_workers=1)
    # info = ray.init(ignore_reinit_error=True, num_cpus= 3, num_gpus= 0)#, include_dashboard=True, dashboard_port=8265, dashboard_host="127.0.0.1")
    # print("Dashboard URL: http://{}".format(info.address_info["webui_url"]))

    # parameters tested in the run
    N_ITER = 15000  # Number of training runs
    LR = "00003"
    NNsize = 100
    env_name = "MA_LocalDe_"+LR+"_maxVol300_5"
    register_env(env_name, env_creator)

    checkpoint_root = "logs/test_ray_MA/SAC_"+env_name
    tensorboard_root = "tensorboard/Test_Variance_Local/TSO_param/" + env_name
    # Where checkpoints are written:
    shutil.rmtree(checkpoint_root, ignore_errors=True, onerror=None)

    # Where some data will be written and used by Tensorboard below:
    #ray_results = './test_ray_MA_tb/'
    #shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    #info = ray.init(ignore_reinit_error=True, num_cpus=2)
    #print("Dashboard URL: " + info.dashboard_url)

    # Configs (see ray's doc for more information)
    #SELECT_ENV = MAEnvWrapper  # Specifies the OpenAI Gym environment for Cart Pole



    config = (SACConfig()
        .rollouts(num_rollout_workers=1)  # 1)
        .resources(num_cpus_per_worker=1, num_learner_workers=1)
        .environment(env=env_name, env_config=env_config, auto_wrap_old_gym_envs=False,
                     disable_env_checking=True)
        #.framework("torch")
        .training(lr=float("0." + LR), train_batch_size=256, gamma=0.9
                  , replay_buffer_config={"buffer_size": 100000},  # only for SAC
                  optimization_config={'actor_learning_rate': float("0." + LR),
                                       'critic_learning_rate': float("0." + LR),
                                       'entropy_learning_rate': float("0." + LR)})  # only for SAC

        # SET FOR EVAL CALLBACK
        .evaluation(evaluation_interval=500, evaluation_duration=52,
                    evaluation_num_workers=1, evaluation_parallel_to_training=False,
                    evaluation_config={
                        "explore": False,
                        "env_config": env_eval_config
                    })
        .debugging(logger_creator=custom_log_creator(os.path.expanduser(tensorboard_root)))
        .multi_agent(
        policies={
            "fr": PolicySpec(
                action_space=ray_ma_env.action_space["fr"],
                observation_space=ray_ma_env.observation_space["fr"],
            ),
            "de": PolicySpec(
                action_space=ray_ma_env.action_space["de"],
                observation_space=ray_ma_env.observation_space["de"]
            )
        },
        policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
                 )
    )
    config["log_level"] = "INFO"#"WARN"  # Suppress too many messages, but try "INFO" to see what can be printed.
    #config.store_buffer_in_checkpoints = False

    # config["num_workers"] = 1#config.__setattr__("num_workers",1)
    config["num_envs_per_worker"] = 1
    # config["num_rollout_workers"]=0
    config["num_cpus_per_worker"] = 1
    config["num_gpus_per_worker"] = 0
    config["rollout_fragment_length"] = 5
    config["metrics_num_episodes_for_smoothing"] = 20
    config["model"]["fcnet_hiddens"] = [int(NNsize)]*2
    config["q_model_config"]["fcnet_hiddens"] = [int(NNsize)]*2
    config["policy_model_config"]["fcnet_hiddens"] = [int(NNsize)]*2
    config["target_network_update_freq"] = 1

    # Trainer
    #tune.run("SAC", stop={"training_iteration":N_ITER},config = config)

    algo = config.build()
    #print(algo.get_config().items())

    #path_to_checkpoint = "/local/ssd/marigiro/logs/test_ray_MA/SAC_MA_BenevFrDe_0001_maxVol300_long_old/long_checkpoint_004000"
    #algo.restore(path_to_checkpoint)

    results = []
    episode_data = []
    episode_json = []

    
    for n in range(N_ITER):
        result = algo.train()
        print(n, result.items())

        if n%250 == 249:
            file_name = algo.save(checkpoint_root)