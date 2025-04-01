import gymnasium

from models.Clearing import Clearing
from models.SecurityAnalysis_multiAgent import security_analysis
import pandas as pd
from datetime import datetime,timedelta
from objects.Bid import Bid
from objects.Zone import Zone
from objects.Border import Border

import grid2op
from lightsim2grid import LightSimBackend
from grid2op.Chronics import ChangeNothing
from grid2op.Action import CompleteAction
from gymnasium.spaces import Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv as MAEnv

import csv
import collections
import numpy as np
import statistics
import tensorflow as tf

import time

START_YEAR = datetime.strptime("01/01/2020 00:00:00", '%d/%m/%Y %H:%M:%S') #do not change
START_DATE = datetime.strptime("01/01/2020 00:00:00", '%d/%m/%Y %H:%M:%S') #start date for current study
END_DATE = datetime.strptime("30/12/2020 23:00:00", '%d/%m/%Y %H:%M:%S')#datetime.strptime("31/12/2020 23:00:00", '%d/%m/%Y %H:%M:%S')

import warnings
warnings.filterwarnings("ignore")

class FilteringEnv_multiAgent(MAEnv):
    def __init__(self,
                 nb_bid_max_fr=42,
                 nb_bid_max_de=26,
                 name_g2op_env="ieee_96_marie",
                 ts_path= "ieee_96_marie/raw_data/timeseries/",
                 #ts_path="D:/Users/girodmar/Documents/Codes/filtrage_RL/objects/create_timeseries/H-1_Y2_ATC_fullyear/",
                 path_data="data",
                 atc="_02022023_noPmin.csv",
                 bid_file="all_bids_grouped_de_fr.csv",
                 not_grouped_file="all_bids.csv",
                 alpha=1.,  # tradeoff between market and tso in social welfare
                 grouped_areas=["fr", "de"],
                 bid_correspondance="bid_correspondance_fr_de.csv",
                 areas={'be', 'de', 'fr'},
                 borders_list={'be_de', 'be_fr', 'de_fr'},
                 max_border_capacity={'be_de': 1175, 'be_fr': 500, 'de_fr': 500},
                 price_normalization=340,
                 max_vol=200,
                 get_var=False,
                 TSO_compensation_penalized=[],
                 greedy_TSO = [],
                 local_welfare_TSO=[],
                 full_greedy_TSO=[]):

        self.bid_list = pd.read_csv(path_data+'/'+bid_file, sep=";")
        self.atc_Fmax = pd.read_csv(path_data+'/Fmax'+atc, sep=";", index_col= 0)
        self.atc_Fmin = pd.read_csv(path_data+'/Fmin'+atc, sep=";", index_col= 0)
        thermal_limits_pd = pd.read_csv(path_data+'/thermal_limits.csv')
        self.thermal_limits = thermal_limits_pd.rate_a.values.tolist()
        self.not_grouped = pd.read_csv(path_data+'/'+not_grouped_file)

        # TODO lire ca avec pandas !
        with open(path_data+'/gen_correspondance.csv', mode='r') as f:
            reader = csv.reader(f, delimiter=";")
            self.gen_correspondance = {rows[1]: int(rows[0]) for rows in reader}
        with open(path_data+'/load_correspondance.csv', mode='r') as f:
            reader = csv.reader(f, delimiter=";")
            self.load_correspondance = {rows[1]: int(rows[0]) for rows in reader}
        with open(path_data+'/'+bid_correspondance, mode='r') as f:
            reader = csv.reader(f, delimiter=";")
            self.bid_correspondance = {int(rows[0]): rows[1] for rows in reader}
        self.no_redispatch_gen = pd.read_csv(path_data+"/no_redispatch_gens.csv")["no_redispatch_gen"].values.tolist()

        self.not_grouped = pd.read_csv(path_data+'/'+not_grouped_file, sep=";")
        gens = pd.DataFrame(columns=self.gen_correspondance.keys(), index = self.not_grouped["CurrentStep"].unique())
        self.no_bid_gens = pd.DataFrame(columns=self.gen_correspondance.keys())
        for index, row in gens.iterrows():
            new_row = pd.Series(gens.columns.isin(self.not_grouped[self.not_grouped["CurrentStep"]==index].GenName).astype(int),gens.columns)
            self.no_bid_gens.loc[int(index)] = new_row
        self.no_bid_gens.rename(columns=self.gen_correspondance, inplace=True)


        self._alpha = alpha
        self.prod_ts = pd.read_csv(ts_path+"prod_df.csv", sep=";")
        self.pmax_ts = pd.read_csv(ts_path+"pmax_df.csv", sep=";")
        self.load_ts = pd.read_csv(ts_path+"load_df.csv", sep=";")
        self.max_load = self.load_ts.select_dtypes(include=[np.number]).to_numpy().max()

        # grid2op part
        self._name_g2op_env = name_g2op_env
        self._grid2op_env = grid2op.make(self._name_g2op_env,
                                         backend=LightSimBackend(),
                                         chronics_class=ChangeNothing,
                                         action_class=CompleteAction,
                                         )
        param = self._grid2op_env.parameters
        param.ENV_DC = True
        self._grid2op_env.change_parameters(param)
        self._grid2op_env.deactivate_forecast()
        _ = self._grid2op_env.reset()

        #area info
        self.areas = areas
        self.borders_list = borders_list
        self.max_border_capacity = max_border_capacity
        self.zones = []
        self.borders = []
        
        # action and observation space
        self.reward_range = (-1, 1)
        self._nb_bid_max_fr = nb_bid_max_fr
        self._nb_bid_max_de = nb_bid_max_de
        self.price_normalization = price_normalization
        self.reward_norm = self._grid2op_env.backend.gen_pmax.sum()*2*180/100 + 0.5

        action_space_fr = Box(low=-1, high=1, shape=(self._nb_bid_max_fr, ))
        action_space_de = Box(low=-1, high=1, shape=(self._nb_bid_max_de,))
        self.action_space = {"fr":action_space_fr, "de":action_space_de}
        obs_space_fr = Box(low=-1.2, high=1.2,
                           shape=(self._grid2op_env.n_line +  # 43 + #size of subset
                                  # self._grid2op_env.n_gen +
                                  # self._grid2op_env.n_load +
                                  #2 * len(areas) +  # atc
                                  # 2 + # for hour and month
                                  1 * self._nb_bid_max_fr,), dtype=np.float32)
        obs_space_de = Box(low=-1.2, high=1.2,
                           shape=(self._grid2op_env.n_line +  # 43 + #size of subset
                                  # self._grid2op_env.n_gen +
                                  # self._grid2op_env.n_load +
                                  #2 * len(areas) +  # atc
                                  # 2 + # for hour and month
                                  1 * self._nb_bid_max_de,), dtype=np.float32)
        self._obs_space_in_preferred_format = True
        self.observation_space = {"fr": obs_space_fr,"de": obs_space_de}

        #self.cumu_reward = 0
        self.cumu_step = 0

        self.all_bids = [] #bids for whole scenario
        self.no_filter_bids = [] #bids for one timestep for other areas
        self.filter_bids = {} #bids for timestep for filter area
        self.grouped_areas = grouped_areas

        self.TSO_compensation_penalized = TSO_compensation_penalized
        self.greedy_TSO = greedy_TSO
        self.local_welfare_TSO = local_welfare_TSO
        self.full_greedy_TSO = full_greedy_TSO
        #self.filter_area = filter_area
        self.demand = []
        self.gen_pmax = {}
        self.gen_p = None
        self.contingencies = [10, 18, 35, 103, 102, 91, 92]#intercos + 91 +92 #[25, 85, 24, 54, 20, 102, 91, 35, 18, 10, 26, 93, 92, 60, 59, 58, 103]
        self.max_vol = max_vol
        self.get_var = get_var
        if self.get_var:
            self.reward_list = []

        for gen in self._grid2op_env.backend._grid.get_generators():
            self.gen_pmax[gen.id] = self._grid2op_env.backend.gen_pmax[gen.id]

        self.date = START_DATE
        self.end_date = self.date #+ timedelta(hours=-1)
        self.current_step = (self.date - START_YEAR).days*24 + (self.date - START_YEAR).seconds//3600 +1
        """
        self.data_for_violin = pd.DataFrame(columns=["Step", "MarketWelfare", "SACosts", "Greedy_cm_costs_DE",
                                                     "Greedy_cm_costs_FR","Local_market_welfare_DE","Local_market_welfare_FR",
                                                     "Local_market_welfare_BE","Balancing_costs_DE","Balancing_costs_FR",
                                                     "Balancing_costs_BE","TSO_comp_DE", "TSO_comp_FR"])
        """
        self.data_for_violin = pd.DataFrame(columns=["Step","TSO_comp_DE", "TSO_comp_FR"])
    def _take_action(self, action):
        for zone in self.grouped_areas:
            for i in range(len(action[zone])):
                delta = round(action[zone][i]*self.price_normalization,2)
                self.filter_bids[zone][i].price += delta #add delta to price
                #print(self.filter_bids[i].price)


    def _instantiate_objects(self):

        self.zones = []
        self.borders = []
        for area in self.areas:
            self.zones.append(Zone(area))

        for border in self.borders_list:
            id = border
            upstream = id[:2]
            downstream = id[3:]
            max_capacity = self.max_border_capacity[id]
            self.borders.append(Border(id, upstream, downstream, max_capacity))

        for zone in self.zones:
            for border in self.borders:
                if border.upstream == zone.id:
                    sign = 1
                    zone.borders.append({"border_id": border.id, "border_sign": sign})
                if border.downstream == zone.id:
                    sign = -1
                    zone.borders.append({"border_id": border.id, "border_sign": sign})

    def _select_bids_for_timestep(self):
        self.filter_bids = {area: [] for area in self.grouped_areas}
        self.no_filter_bids = []
        self.demand = []
        for bid in self.all_bids:
            if bid.step == self.current_step:
            #if datetime.strptime(bid.start_date, '%d/%m/%Y %H:%M:%S') == self.date:
                if bid.gen_name != bid.zone:
                    if bid.zone in self.grouped_areas and bid.gen_name != "202_STEAM": #in self.filter_area:
                        self.filter_bids[bid.zone].append(bid)
                    else:
                        self.no_filter_bids.append(bid)
                else:
                    self.demand.append(bid)
        for zone in self.zones:
            zone.bids = []
            zone.demand = []
            filter_bid_list = list(self.filter_bids.values())
            for bid in filter_bid_list[0]+filter_bid_list[1]+self.no_filter_bids:
                if bid.zone == zone.id:
                    zone.bids.append(bid)
            for demand in self.demand:
                if demand.zone == zone.id:
                    zone.demand.append(demand)

    def _date_to_string(self,date):
        t = str(date.year) + '{:02d}'.format(date.month) + '{:02d}'.format(
            date.day) + "_" + '{:02d}'.format(date.hour) + "00"
        return t

    def _read_atc(self):
        t = self._date_to_string(self.date)

        atc_max_list = self.atc_Fmax[self.atc_Fmax["time"] == t]
        atc_min_list = self.atc_Fmin[self.atc_Fmin.time == t]
        for border in self.borders:
            border.max_flow = atc_max_list[border.id].values[0]
            border.min_flow = -atc_min_list[border.id].values[0]

    def _read_power(self):
        #t= self._date_to_string(self.date)

        #read data
        pmax_data = self.pmax_ts[self.pmax_ts["step"] == self.current_step]
        pmax_data.rename(columns=self.gen_correspondance, inplace=True)
        pmax_dict = pmax_data.drop(["date","step"], axis=1).to_dict('records')[0]

        prod_data = self.prod_ts[self.prod_ts["step"] == self.current_step]
        prod_data.rename(columns=self.gen_correspondance, inplace=True)
        prod_dict = prod_data.drop(["date","step"], axis=1).to_dict('records')[0]

        load_data = self.load_ts[self.load_ts["step"] == self.current_step]
        load_data.rename(columns=self.load_correspondance, inplace=True)
        load_dict_int = load_data.drop(["date","step"], axis=1).to_dict('records')[0]

        #format
        load_dict = {k: v for k, v in load_dict_int.items() if v > 0}
        load_od = collections.OrderedDict(sorted(load_dict.items()))
        load_od_list = list(load_od.values())
        load_p = np.array(load_od_list)

        gen_od = collections.OrderedDict(sorted(prod_dict.items()))
        gen_od_list = list(gen_od.values())
        gen_p = np.array(gen_od_list)
        return gen_p, load_p, pmax_dict

    def _get_obs_agent(self, grid2op_obs, bids):
        max_bid = 897.0

        bids_fr = bids["fr"]
        bids_de = bids["de"]
        bids_fr.sort(key=lambda x: x.id)
        bids_de.sort(key=lambda x: x.id)

        bid_qmax_fr = [np.float32(bid.qmax/max_bid) for bid in bids_fr]
        bid_qmax_de = [np.float32(bid.qmax / max_bid) for bid in bids_de]

        obs_fr = np.concatenate((grid2op_obs.p_or/(2*np.array(self.thermal_limits)), bid_qmax_fr), axis=0)
        obs_de = np.concatenate((grid2op_obs.p_or /(2*np.array(self.thermal_limits)), bid_qmax_de), axis=0)

        return {"fr": obs_fr, "de": obs_de}

    def step(self, action):
        #print(self.date)
        self._take_action(action) #action est un dict
        self.cumu_step += 1
        #run market clearing
        compute_price = False
        if len(self.local_welfare_TSO)>0 or len(self.full_greedy_TSO)>0:
            compute_price = True
        market_welfare, accepted_powers, market_price = Clearing(self.zones, self.borders, compute_price=compute_price)

        #update production in grid2op_env
        no_bid_gens_t = self.no_bid_gens.loc[[self.current_step]]
        # print(no_bid_gens_t.loc[:,(no_bid_gens_t == 0).any(axis=0)].columns)
        no_order_gens = list(no_bid_gens_t.loc[:, (no_bid_gens_t == 0).any(axis=0)].columns)
        no_redispatch_gen_t = list(set(self.no_redispatch_gen + no_order_gens))
        gen_delta = [0] * len(self._grid2op_env.backend._grid.get_generators())
        if len(self.full_greedy_TSO)>0:
            TSO_comp = {zone: 0 for zone in self.full_greedy_TSO}
            real_TSO_comp = {zone: 0 for zone in self.full_greedy_TSO}
            balancing_costs = {zone: 0 for zone in self.full_greedy_TSO}
        if len(self.local_welfare_TSO)>0:
            local_market_welfare = {zone: 0 for zone in self.local_welfare_TSO}

        #dispatch clearing results
        for zone in self.zones:
            for bid in zone.bids:
                if bid.qmax>0:
                    accepted_power = accepted_powers[zone.id][bid.id]
                    bid.update_power(bid.isSell*accepted_power)

                    if bid.zone in self.full_greedy_TSO:
                        if bid.isSell * (bid.initial_price - bid.price) > 0.1:  # paradoxically accepted
                            TSO_comp[bid.zone] = TSO_comp.get(bid.zone) + bid.accepted_power* (bid.initial_price - bid.price) # Market welfare area
                            if bid.isSell * (bid.initial_price - market_price[zone.id]) > 0:
                                real_TSO_comp[bid.zone] = real_TSO_comp.get(bid.zone) + bid.accepted_power * (
                                        bid.initial_price - market_price[zone.id])
                    if len(self.local_welfare_TSO) > 0 and zone.id in self.local_welfare_TSO and accepted_power>0:
                        local_market_welfare[zone.id] = local_market_welfare.get(zone.id)- \
                                                        bid.accepted_power*(bid.initial_price - market_price[zone.id])
                        #print(bid.id, bid.accepted_power, bid.initial_price)
                        #print(bid.id, bid.accepted_power, bid.initial_price, market_price[zone.id], local_market_welfare[zone.id])
                    """
                    if accepted_power >0:
                        new_violin_row = pd.Series(
                            [self.current_step, bid.accepted_power, bid.zone, bid.gen_name, bid.isSell]
                            , index=["Step", "AcceptedPower", "Zone", "Gen", "isSell"])
                        self.data_for_violin = self.data_for_violin.append(new_violin_row, ignore_index=True)
                    """
                    if zone.id in self.grouped_areas: #de-aggregate bids for security analysis
                        remaining_power = accepted_power
                        for k,v in self.bid_correspondance.items():
                            if v == bid.gen_name and k not in no_order_gens:
                                #no_order_gens[k] = 0
                                if remaining_power >0:
                                    if bid.isSell==1:
                                        distrib_power = min(self.gen_pmax[k] - self.gen_p[k], remaining_power)
                                    else:
                                        distrib_power = min(self.gen_p[k], remaining_power)
                                    remaining_power -= distrib_power
                                    gen_delta[k] += bid.isSell*distrib_power

                    else:
                        delta_pow = min(bid.accepted_power,  self.gen_pmax[bid.gen_id] - self.gen_p[bid.gen_id]) #pour résoudre des problèmes d'arrondis
                        gen_delta[bid.gen_id] += delta_pow #bid.accepted_power #somme en cas d'offre a la hausse + baisse
                #print(bid.id, bid.price, bid.initial_price, bid.qmax, bid.accepted_power, bid.isSell)
            #print(zone.demand[0].id,300,300,zone.demand[0].qmax,accepted_powers[zone.id][zone.demand[0].id],zone.demand[0].isSell)
            if len(self.local_welfare_TSO) > 0 and zone.id in self.local_welfare_TSO and len(zone.demand) > 0:
                #print(zone.demand[0].id,accepted_powers[zone.id][zone.demand[0].id], zone.demand[0].initial_price)
                local_market_welfare[zone.id] = local_market_welfare.get(zone.id) - accepted_powers[zone.id][
                    zone.demand[0].id] * zone.demand[0].isSell * (zone.demand[0].initial_price - market_price[zone.id])
            if len(self.full_greedy_TSO) > 0 and zone.id in self.full_greedy_TSO and len(zone.demand) > 0:
                balancing_costs[zone.id] = - accepted_powers[zone.id][zone.demand[0].id] * zone.demand[0].isSell * market_price[zone.id]
        # print(price["de"], price["fr"], price["be"])
        # print("TSO compensation", TSO_comp)
        #print(local_market_welfare)
        delta_array = np.array(gen_delta)
        gen_p_market = np.add(np.array(self.gen_p),delta_array) #somme prod initiale et res marche

        #run security analysis
        """
        sa_welfare, greedy_cm_costs = security_analysis(self._grid2op_env, [x * 1.4 for x in self.thermal_limits], self.gen_pmax, gen_p_market,
                                       no_redispatch_gen_t, self.contingencies, self.max_vol,
                                       self.gen_correspondance, self.load_correspondance, greedyTSO=self.greedy_TSO)
        """
        sa_welfare = 0 #when computing price
        greedy_cm_costs = {"be":0,"de":0,"fr":0}
        ##PRINT RESULTS
        """
        new_violin_row = pd.Series(
            [self.current_step, market_welfare, sa_welfare, greedy_cm_costs["de"],greedy_cm_costs["fr"],
             local_market_welfare["de"], local_market_welfare["fr"], local_market_welfare["be"],
             balancing_costs["de"],balancing_costs["fr"],balancing_costs["be"],TSO_comp["de"], TSO_comp["fr"]]
            , index=["Step", "MarketWelfare", "SACosts", "Greedy_cm_costs_DE","Greedy_cm_costs_FR","Local_market_welfare_DE",
                     "Local_market_welfare_FR", "Local_market_welfare_BE","Balancing_costs_DE","Balancing_costs_FR",
                     "Balancing_costs_BE","TSO_comp_DE", "TSO_comp_FR"])
        self.data_for_violin = self.data_for_violin.append(new_violin_row, ignore_index=True)
        """
        """
        new_violin_row = pd.Series(
            [self.current_step, market_price["de"], market_price["fr"], market_price["be"]]
            , index=["Step", "Price DE", "Price FR", "Price BE"])
        self.data_for_violin = self.data_for_violin.append(new_violin_row, ignore_index=True)
        #print(market_welfare, sa_welfare, greedy_cm_costs["de"],greedy_cm_costs["fr"],
        #      local_market_welfare["de"], local_market_welfare["fr"], local_market_welfare["be"])
        #print(balancing_costs["de"],balancing_costs["fr"],balancing_costs["be"],TSO_comp["de"], TSO_comp["fr"])
        """
        #new_violin_row = pd.Series([self.current_step, real_TSO_comp["de"], real_TSO_comp["fr"]], index = ["Step","TSO_comp_DE", "TSO_comp_FR"])
        #self.data_for_violin = self.data_for_violin.append(new_violin_row, ignore_index=True)

        reward_total = market_welfare - sa_welfare
        #print("reward", reward_total)
        reward_fr = reward_total
        reward_de = reward_total
        for zone in self.greedy_TSO:
            if zone == "fr":
                reward_fr = market_welfare - greedy_cm_costs["fr"]
            if zone == "de":
                reward_de = market_welfare - greedy_cm_costs["de"]
        for zone in self.full_greedy_TSO:
            if zone == 'fr':
                reward_fr = - ( greedy_cm_costs["fr"]+ TSO_comp["fr"] + balancing_costs["fr"])
            if zone == 'de':
                reward_de = - ( greedy_cm_costs["de"]+ TSO_comp["de"] + balancing_costs["de"])

        for zone in self.local_welfare_TSO:
            if zone == 'fr':
                reward_fr = local_market_welfare["fr"] - greedy_cm_costs["fr"]
            if zone == 'de':
                reward_de = local_market_welfare["de"] - greedy_cm_costs["de"]

        reward_fr = reward_fr/self.reward_norm + 0.6 #0.6 pour recentrer
        reward_de = reward_de / self.reward_norm + 0.6
        rewards = {"fr":reward_fr, "de":reward_de}
        #print(rewards)
        #self.cumu_reward += reward

        #prepare obs for next step
        self.date = self.date + timedelta(hours=1)
        self.current_step += 1

        self._select_bids_for_timestep() #updates self.filter_bids+self.no_filter_bids
        self._read_atc()

        self.gen_p, load_p, pmax_dict = self._read_power()
        self.gen_pmax.update(pmax_dict)

        delta_slack = sum(load_p) - sum(self.gen_p)
        sum_pmax = sum(self.gen_pmax.values())

        gen_slack = self.gen_p.copy()
        for gen_id in range(len(self.gen_p)):
            gen_slack[gen_id] += delta_slack * self.gen_pmax[gen_id] / sum_pmax

        # mettre a jour le grid2op_env
        grid2op_act = self._grid2op_env.action_space({"injection": {"load_p": load_p,
                                                                    "prod_p": gen_slack}})
        grid2op_obs, _, _,_ = self._grid2op_env.step(grid2op_act)
        #print(grid2op_act._dict_inj)
        
        obs = self._get_obs_agent(grid2op_obs, self.filter_bids)
        terminated = self.date == self.end_date
        terminateds = {"fr":terminated,"de":terminated}
        terminateds['__all__'] = terminated
        truncated = False

        if terminated:
            self.data_for_violin.to_csv("2024_02_05_MABenevFrDe_Comp_eval.csv")

        #if terminated and self.get_var:
            #self.reward_list.append(reward)
            #tf.summary.scalar('cumu_reward', data=self.cumu_reward, step=self.cumu_step)

        if self.date == END_DATE:
            truncated = True
            #self.data_for_violin.to_csv("2024_01_17_MABenevFrDe_test2_eval.csv")
            self.date = START_DATE
            self.end_date = self.date
            self.current_step = (self.date - START_YEAR).days*24 + (self.date - START_YEAR).seconds//3600 + 1
            #self.cumu_reward = 0
            if self.get_var:
                self.reward_list = []
        truncateds = {"fr":truncated, "de": truncated}
        truncateds['__all__'] = truncated

        return obs, rewards, terminateds, truncateds, {}

    def reset(self, *,seed=None, options=None):
        self.end_date =self.date + timedelta(days=7) #one checkpoint per week #end_
        end_step = self.current_step + 7*24
        self._instantiate_objects()

        #load all bids
        self.all_bids = []
        bid_list_sub = self.bid_list[(self.bid_list["CurrentStep"] >= self.current_step) &
                                     (self.bid_list["CurrentStep"] <= end_step)]
        for index, bid in bid_list_sub.iterrows():
            price = bid["Price"]
            qmax = bid["Qmax"]
            isSell = bid["isSell"]
            gen_name = bid["GenName"]
            zone = bid["MarketArea"]
            if gen_name != zone and zone not in self.grouped_areas: #filter_bids were aggregated
                gen_id = self.gen_correspondance[gen_name]
                node = self._grid2op_env.backend._grid.get_generators()[gen_id].bus_id
            elif gen_name == zone:
                gen_id = None
                node = None
            else:
                gen_id = None
                gen = list(self.bid_correspondance.keys())[list(self.bid_correspondance.values()).index(gen_name)]
                node = self._grid2op_env.backend._grid.get_generators()[gen].bus_id

            start_date = bid["StartDate"]
            step = bid["CurrentStep"]
            start_datetime = datetime.strptime(start_date, '%d/%m/%Y %H:%M:%S')
            id = gen_name + "_" + str(int(isSell)) + "_" + self._date_to_string(start_datetime)
            self.all_bids.append(Bid(id, price, qmax, isSell, gen_id, gen_name, zone, start_date, step,node))

        self._select_bids_for_timestep() #creates initial bid list in self.bids
        self._read_atc()

        #read new powers
        self.gen_p, load_p, pmax_dict = self._read_power()
        self.gen_pmax.update(pmax_dict)
        #reset grid2op env
        observation_grid2op = self._grid2op_env.reset()

        delta_slack = sum(load_p) - sum(self.gen_p)
        sum_pmax = sum(self.gen_pmax.values())

        gen_slack = self.gen_p.copy()
        for gen_id in range(len(self.gen_p)):
            gen_slack[gen_id] += delta_slack*self.gen_pmax[gen_id]/sum_pmax

        # mettre a jour le grid2op_env
        grid2op_act = self._grid2op_env.action_space({"injection": {"load_p": load_p,
                                                                    "prod_p": gen_slack}})
        grid2op_obs, _, _,_ = self._grid2op_env.step(grid2op_act)
        return (self._get_obs_agent(grid2op_obs, self.filter_bids), {})