# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
# This file is part of RL-filtering, a project for bid filtering in European balancing platforms.

class Bid:

    def __init__(self, id,price, qmax, isSell, gen_id, gen_name, zone, start_date, step, node):
        self.id = id
        self.price = price
        self.initial_price = price
        self.qmax = qmax
        self.isSell = 1 if isSell == True else -1
        self.gen_id = gen_id #generator index in grid2op
        self.gen_name = gen_name
        self.zone = zone
        self.accepted_power = 0
        self.start_date = start_date
        self.node = node
        self.step = step

    def update_power(self,accepted_power):
        self.accepted_power = accepted_power
