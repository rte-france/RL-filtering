# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
# This file is part of RL-filtering, a project for bid filtering in European balancing platforms.

import numpy as np
import operator
from models.SecurityAnalysis import security_analysis
from models.FilteringEnv import FilteringEnv

class BaselineAgent:
    # agent that runs Baseline filtering
    def __init__(self, gym_env):
        self.action_space = gym_env.action_space
        self._grid2op_env = gym_env._grid2op_env.copy()
        self._thermal_limits = gym_env.thermal_limits
        self.contingencies = gym_env.contingencies

    def _n_1_sa(self, obs, bid, nb_lines, price):

        #check if there is a congestion in N state
        for i in range(nb_lines):
            if obs.p_or[i] > self._thermal_limits[i]:
                bid[0] = price  # infinite price for bids that create congestion
                break

        ## N-1 security analysis
        for index, c in enumerate(self.contingencies):
            if bid[0] == price: #the bid has already been removed in N state
                break
            if index == 0: #first contingency -> starts from a closed network
                set_lines = self._grid2op_env.action_space({"change_line_status": [c]}) #open contingency
                obs, *_ = self._grid2op_env.step(set_lines) #grid2op opens lines and runs a powerflow
                for i in range(nb_lines):
                    if obs.p_or[i] > self._thermal_limits[i]:
                        bid[0] = price #infinite price if the bid causes congestion
                        break

            else:
                close_lines = self._grid2op_env.action_space({"set_line_status": [[self.contingencies[index - 1], 1]]})
                self._grid2op_env.step(close_lines) #first close previous contingency
                open_lines = self._grid2op_env.action_space({"set_line_status": [[c, -1]]})
                obs, *_ = self._grid2op_env.step(open_lines) #now open this contingency
                for i in range(nb_lines):
                    if obs.p_or[i] > self._thermal_limits[i]:
                        bid[0] = price
                        break


    def act(self, obs, reward, done):
        # step 1: read obs
        self._grid2op_env.reset()
        grid = self._grid2op_env.backend._grid
        nb_lines = len(grid.get_lines()) + len(grid.get_trafos())
        nb_gens = len(grid.get_generators())
        nb_loads = len(grid.get_loads())
        p_or = obs[:nb_lines]
        gen_p = obs[nb_lines:nb_lines + nb_gens]
        gen_pmax = obs[nb_lines+ nb_gens:nb_lines + 2*nb_gens]
        load_p = obs[nb_lines + 2*nb_gens:nb_lines + 2*nb_gens + nb_loads]

        nb_bids = int(len(obs[nb_lines + 2*nb_gens + nb_loads:])/4)
        bids_price = obs[nb_lines + 2*nb_gens + nb_loads:nb_lines + 2*nb_gens + nb_loads + nb_bids]
        bids_qmax = obs[nb_lines + 2*nb_gens + nb_loads + nb_bids:nb_lines + 2*nb_gens + nb_loads + 2*nb_bids]
        bids_isSell = obs[nb_lines + 2*nb_gens + nb_loads + 2*nb_bids:nb_lines + 2*nb_gens + nb_loads + 3*nb_bids]
        bids_gens = obs[nb_lines + 2*nb_gens + nb_loads + 3*nb_bids:]

        # format bids, separate upward/downward and sort them in merit order
        bids_index = [i for i in range(nb_bids)]
        bids = list(zip(bids_price,bids_qmax,bids_isSell,bids_gens, bids_index))
        bids_sorted = sorted(bids, key=operator.itemgetter(2, 0))
        bids_sorted = [list(elem) for elem in bids_sorted]
        bid_sorted_up = [x for x in bids_sorted if x[2] == 1]
        bid_sorted_down = [x for x in bids_sorted if x[2] == -1]


        #step 2 : update grid2op
        gen_p = np.array(gen_p)
        load_p = np.array(load_p)

        #run simple powerflow
        grid2op_act = self._grid2op_env.action_space({"injection": {"load_p": load_p,
                                                                    "prod_p": gen_p}})
        obs, *_ = self._grid2op_env.step(grid2op_act)

        # run security analysis on network to have a secure base case and update grid2op network
        gen_p_sa = security_analysis(self._grid2op_env, self._thermal_limits, gen_pmax, gen_p, [],
                                     self.contingencies, max_vol = 10000 ,mode="basecase")
        gen_p_sa = np.array(gen_p_sa)
        grid2op_act = self._grid2op_env.action_space({"injection": {"load_p": load_p,
                                                                    "prod_p": gen_p_sa}})
        obs, *_ = self._grid2op_env.step(grid2op_act)

        # step 3: test upward and downward bids separately in merit order
        gen_with_bid_up = gen_p_sa
        gen_with_bid_down = gen_p_sa
        for bid in bid_sorted_up: #pile on each bid in merit order
            gen_with_bid_up[int(bid[3])] += bid[1] #add bid volume to corresponding gen

            #distribute imbalance linked to added bid on all loads and update grid2op
            delta_slack = sum(gen_with_bid_up) - sum(load_p)
            load_bid = load_p.copy()

            for load_id in range(len(load_p)):
                load_bid[load_id] += delta_slack * load_p[load_id] / sum(load_p)
            grid2op_act_sa = self._grid2op_env.action_space({"injection": {"load_p": load_bid,
                                                                    "prod_p": gen_with_bid_up}})
            obs, reward, done, info = self._grid2op_env.step(grid2op_act_sa)

            up_price = 9999 #~infinite price
            self._n_1_sa(obs, bid, nb_lines, up_price) #check if the network is secure with added bid


        for bid in reversed(bid_sorted_down):
            gen_with_bid_down[int(bid[3])] -= bid[1]

            # distribute slack
            delta_slack = sum(gen_with_bid_down) - sum(load_p)
            load_bid = load_p.copy()
            for load_id in range(len(load_p)):
                load_bid[load_id] += delta_slack * load_p[load_id] / sum(load_p)

            grid2op_act_sa = self._grid2op_env.action_space({"injection": {"load_p": load_bid,
                                                                    "prod_p": gen_with_bid_down}})
            obs, *_ = self._grid2op_env.step(grid2op_act_sa)

            down_price= -9999
            self._n_1_sa(obs, bid, nb_lines, down_price) # check if network is secure


        unsorted_bids = bid_sorted_up + bid_sorted_down
        return_bids = sorted(unsorted_bids, key=operator.itemgetter(4)) #sort bids in index order to match environment
        return_price = [bid[0] for bid in return_bids]
        return return_price


if __name__ == "__main__":

    env = FilteringEnv(bid_file="all_bids.csv", #testing data set bids - testing network by default
                                      max_vol=200, #max allowed redispatch volume
                                      bids_grouped=False)
    agent = BaselineAgent(env)

    done = False
    sum_reward = 0
    reward = 0

    #evaluation on 52 weeks
    for i in range(52):
        obs = env.reset()
        done = False
        while not done:
            action = agent.act(obs, reward, done)
            obs, reward, done, info = env.step(action)
            sum_reward += reward
        print("reward", reward)
    print("total reward", sum_reward)
