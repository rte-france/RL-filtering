from models.FilteringEnv_multiAgent import FilteringEnv_multiAgent

class DoNothingMultiAgent:

    def __init__(self, gym_env):
        self.action_space = gym_env.action_space
        self._nb_bid_max_fr = gym_env._nb_bid_max_fr
        self._nb_bid_max_de = gym_env._nb_bid_max_de

    def act(self):

        delta = {"fr":[0]*self._nb_bid_max_fr, "de":[0]*self._nb_bid_max_de}
        return delta

def evaluate(env, agent):
    sum_reward = {}

    for i in range(52):
        obs = env.reset()
        terminateds = {"fr": False, "de": False}
        truncateds = {"fr": False, "de": False}
        while not terminateds["fr"] and not truncateds["fr"]:
            action = agent.act()
            obs, reward, terminateds, truncateds, info = env.step(action)
            sum_reward = {k: sum_reward.get(k, 0) + reward.get(k, 0) for k in set(reward)}
        print(sum_reward)
    print("total reward", sum_reward["de"])


if __name__=='__main__':

    env = FilteringEnv_multiAgent(#ts_path ="ieee_96_marie/raw_data/timeseries/",
                                           #"D:/Users/girodmar/Documents/Codes/filtrage_RL/objects/create_timeseries/H-1_Y2_ATC_fullyear/",
                                      #atc="_30052023_Y2.csv",
                                      #nb_bid_max=26,
                                      grouped_areas=["fr","de"],
                                      #bid_correspondance="bid_correspondance_fr_de.csv",
                                      #not_grouped_file="all_bids_Y2.csv",
                                      max_vol=300)


    agent = DoNothingMultiAgent(env)
    evaluate(env, agent)