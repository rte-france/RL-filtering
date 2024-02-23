# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
# This file is part of RL-filtering, a project for bid filtering in European balancing platforms.

class Zone:

    def __init__(self, id):
        self.id = id
        self.bids = []
        self.borders = []
        self.demand = []
