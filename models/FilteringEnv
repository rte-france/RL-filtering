# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
# This file is part of RL-filtering, a project for bid filtering in European balancing platforms.

import gym

from models.Clearing import Clearing
from models.SecurityAnalysis import security_analysis
import pandas as pd
from datetime import datetime,timedelta
from objects.Bid import Bid
from objects.Zone import Zone
from objects.Border import Border

import grid2op
from lightsim2grid import LightSimBackend
from grid2op.Chronics import ChangeNothing
from grid2op.Action import CompleteAction
from gym.spaces import Box

import csv
import collections
import numpy as np


START_YEAR = datetime.strptime("01/01/2020 00:00:00", '%d/%m/%Y %H:%M:%S') #do not change
START_DATE = datetime.strptime("01/01/2020 00:00:00", '%d/%m/%Y %H:%M:%S')
END_DATE = datetime.strptime("30/12/2020 23:00:00", '%d/%m/%Y %H:%M:%S')

import warnings
warnings.filterwarnings("ignore")

class FilteringEnv_testDataParams(gym.Env):
    def __init__(self,
                 nb_bid_max=42,
                 name_g2op_env="ieee_96_marie",
                 ts_path="C:/Users/girodmar/data_grid2op/ieee_96_marie/raw_data/timeseries/",
                 path_data="data",
                 atc = "_02022023_noPmin.csv",
                 bid_file = "all_bids_grouped.csv",
                 not_grouped_file = "all_bids.csv",
                 filter_area="fr",
                 areas = {'be', 'de', 'fr'},
                 borders_list = {'be_de', 'be_fr', 'de_fr'},
                 max_border_capacity= {'be_de': 1175, 'be_fr': 500 , 'de_fr':500},
                 price_normalization = 340,
                 max_vol=400,
                 bids_grouped = True
                 ):

        #read data
        self.bids_grouped = bids_grouped #bids are grouped by similar gens in same zone for Proposed filtering
                                         # to reduce action space
        self.bid_list = pd.read_csv(path_data+'/'+bid_file, sep=";")
        self.atc_Fmax = pd.read_csv(path_data+'/Fmax'+atc, sep=";", index_col= 0)
        self.atc_Fmin = pd.read_csv(path_data+'/Fmin'+atc, sep=";", index_col= 0)
        thermal_limits_pd = pd.read_csv(path_data+'/thermal_limits.csv')
        self.thermal_limits = thermal_limits_pd.rate_a.values.tolist()
        if self.bids_grouped:
            self.not_grouped = pd.read_csv(path_data+'/'+not_grouped_file)

        with open(path_data+'/gen_correspondance.csv', mode='r') as f:
            reader = csv.reader(f, delimiter=";")
            self.gen_correspondance = {rows[1]: int(rows[0]) for rows in reader}
        with open(path_data+'/load_correspondance.csv', mode='r') as f:
            reader = csv.reader(f, delimiter=";")
            self.load_correspondance = {rows[1]: int(rows[0]) for rows in reader}
        with open(path_data+'/bid_correspondance_fr.csv', mode='r') as f:
            reader = csv.reader(f, delimiter=";")
            self.bid_correspondance = {int(rows[0]): rows[1] for rows in reader}
        self.no_redispatch_gen = pd.read_csv(path_data+"/no_redispatch_gens.csv")["no_redispatch_gen"].values.tolist()

        self.not_grouped = pd.read_csv(path_data+'/'+not_grouped_file, sep=";")
        gens = pd.DataFrame(columns=self.gen_correspondance.keys(), index = self.not_grouped["CurrentStep"].unique())
        self.no_bid_gens = pd.DataFrame(columns=self.gen_correspondance.keys())
        for index, row in gens.iterrows():
            new_row = pd.Series(gens.columns.isin(self.not_grouped[self.not_grouped["CurrentStep"]==index].GenName).astype(int),gens.columns)
            new_row.name = int(index)
            self.no_bid_gens = self.no_bid_gens.append(new_row)
        self.no_bid_gens.rename(columns=self.gen_correspondance, inplace=True)

        self.prod_ts = pd.read_csv(ts_path+"prod_df.csv", sep=";")
        self.pmax_ts = pd.read_csv(ts_path+"pmax_df.csv", sep=";")
        self.load_ts = pd.read_csv(ts_path+"load_df.csv", sep=";")
        self.max_load = self.load_ts.select_dtypes(include=[np.number]).to_numpy().max()

        # grid2op env - manages all network computations
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
        self._nb_bid_max = nb_bid_max
        self.price_normalization = price_normalization
        self.reward_norm = self._grid2op_env.backend.gen_pmax.sum()*2*180/100 + 0.5 # 180 = max redispatch price, gen_pmax = max qty sold/redispatched, 0.5 to center
        self.action_space = Box(low=-1, high=1, shape=(self._nb_bid_max, ))
        self.observation_space = Box(low= -1.2, high= 1.2,
                                     shape=(self._grid2op_env.n_line +
                                            self._nb_bid_max, ), dtype = np.float32)
        self.cumu_reward = 0
        self.cumu_step = 0

        self.all_bids = [] #bids for whole scenario
        self.no_filter_bids = [] #bids for one timestep for other areas
        self.filter_bids = [] #bids for timestep for filter area
        self.filter_area = filter_area
        self.demand = []
        self.gen_pmax = {}
        self.gen_p = None
        self.contingencies = [25, 85, 24, 54, 20, 102, 91, 35, 18, 10, 26, 93, 92, 60, 59, 58, 103] #critical lines
        self.max_vol = max_vol

        for gen in self._grid2op_env.backend._grid.get_generators():
            self.gen_pmax[gen.id] = self._grid2op_env.backend.gen_pmax[gen.id]

        self.date = START_DATE
        self.end_date = self.date
        self.current_step = (self.date - START_YEAR).days*24 + (self.date - START_YEAR).seconds//3600 +1

    def _take_action(self, action):
        for i in range(len(action)):
            delta = round(action[i]*self.price_normalization,2)
            self.filter_bids[i].price += delta #add delta to price


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
        self.filter_bids = []
        self.no_filter_bids = []
        self.demand = []
        for bid in self.all_bids:
            if bid.step == self.current_step:
                if bid.gen_name != bid.zone:
                    if bid.zone == self.filter_area:
                        self.filter_bids.append(bid)
                    else:
                        self.no_filter_bids.append(bid)
                else:
                    self.demand.append(bid)
        for zone in self.zones:
            zone.bids = []
            zone.demand = []
            for bid in self.filter_bids+self.no_filter_bids:
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

        #read data
        pmax_data = self.pmax_ts[self.pmax_ts["step"] == self.current_step]
        pmax_data.rename(columns=self.gen_correspondance, inplace=True)
        pmax_dict = pmax_data.drop(["date","step"], axis=1).to_dict('r')[0]

        prod_data = self.prod_ts[self.prod_ts["step"] == self.current_step]
        prod_data.rename(columns=self.gen_correspondance, inplace=True)
        prod_dict = prod_data.drop(["date","step"], axis=1).to_dict('r')[0]

        load_data = self.load_ts[self.load_ts["step"] == self.current_step]
        load_data.rename(columns=self.load_correspondance, inplace=True)
        load_dict_int = load_data.drop(["date","step"], axis=1).to_dict('r')[0]

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
        max_bid = 897.0 #maximum power bid by a generator in existing data

        bids.sort(key=lambda x: x.id)
        bid_price = [np.float32(bid.price/self.price_normalization) for bid in bids]
        bid_qmax = [np.float32(bid.qmax/max_bid) for bid in bids]
        bid_isSell = [np.float32(bid.isSell) for bid in bids]
        #bid_volume = [a*b for a,b in zip(bid_qmax,bid_isSell)]

        if not self.bids_grouped: #for baseline filtering
            bid_gen = [bid.gen_id for bid in bids]

        #atc = [np.float32(border.max_flow/border.max_capacity) for border in self.borders] + [np.float32(border.min_flow/border.max_capacity) for border in self.borders]

        if not self.bids_grouped: #baseline filtering
            return np.concatenate((grid2op_obs.p_or, grid2op_obs.gen_p, list(self.gen_pmax.values()),grid2op_obs.load_p,
                                   bid_price,bid_qmax, bid_isSell, bid_gen), axis=0)
        else:
            return np.concatenate((grid2op_obs.p_or/np.array(self.thermal_limits), bid_qmax), axis=0)

    def step(self, action):

        self._take_action(action)
        self.cumu_step += 1
        #run market clearing
        market_welfare, accepted_powers = Clearing(self.zones, self.borders)

        #update production in grid2op_env
        no_bid_gens_t = self.no_bid_gens.loc[[self.current_step]]
        no_order_gens = list(no_bid_gens_t.loc[:, (no_bid_gens_t == 0).any(axis=0)].columns)
        no_redispatch_gen_t = list(set(self.no_redispatch_gen + no_order_gens))
        gen_delta = [0] * len(self._grid2op_env.backend._grid.get_generators())

        for zone in self.zones:
            for bid in zone.bids:
                if bid.qmax>0:
                    accepted_power = accepted_powers[zone.id][bid.id]
                    bid.update_power(bid.isSell*accepted_power)

                    if zone.id == self.filter_area and self.bids_grouped: #de-aggregate bids in filter area for security analysis
                        remaining_power = accepted_power
                        for k,v in self.bid_correspondance.items():
                            if v == bid.gen_name and k not in no_order_gens:

                                if remaining_power >0:
                                    if bid.isSell==1:
                                        distrib_power = min(self.gen_pmax[k] - self.gen_p[k], remaining_power)
                                    else:
                                        distrib_power = min(self.gen_p[k], remaining_power)
                                    remaining_power -= distrib_power
                                    gen_delta[k] += bid.isSell*distrib_power

                    else:
                        delta_pow = min(bid.accepted_power,  self.gen_pmax[bid.gen_id] - self.gen_p[bid.gen_id]) #solve rounding errors
                        gen_delta[bid.gen_id] += delta_pow

        delta_array = np.array(gen_delta)
        gen_p_market = np.add(np.array(self.gen_p),delta_array) #sum of initial production and market dispatch

        #run security analysis
        sa_welfare = security_analysis(self._grid2op_env, [x * 1.4 for x in self.thermal_limits], self.gen_pmax, gen_p_market,
                                       no_redispatch_gen_t, self.contingencies, self.max_vol, mode="sa")

        reward_total = market_welfare - sa_welfare
        reward = reward_total/self.reward_norm + 0.6 #0.6 to center
        self.cumu_reward += reward

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

        # update grid2op_env
        grid2op_act = self._grid2op_env.action_space({"injection": {"load_p": load_p,
                                                                    "prod_p": gen_slack}})
        grid2op_obs, _, _,_ = self._grid2op_env.step(grid2op_act)
        
        obs = self._get_obs_agent(grid2op_obs, self.filter_bids)
        done = self.date == self.end_date

        if self.date == END_DATE:
            done = True
            self.date = START_DATE
            self.end_date = self.date
            self.current_step = (self.date - START_YEAR).days*24 + (self.date - START_YEAR).seconds//3600 + 1
            self.cumu_reward = 0

        return obs, reward, done, {}

    def reset(self, seed=None, options=None):
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
            if gen_name != zone and (self.bids_grouped == False or zone != self.filter_area): #filter_bids were aggregated
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

        # update grid2op_env
        grid2op_act = self._grid2op_env.action_space({"injection": {"load_p": load_p,
                                                                    "prod_p": gen_slack}})
        grid2op_obs, _, _,_ = self._grid2op_env.step(grid2op_act)
        return self._get_obs_agent(grid2op_obs, self.filter_bids)
