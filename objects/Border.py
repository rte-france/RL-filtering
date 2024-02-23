# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
# This file is part of RL-filtering, a project for bid filtering in European balancing platforms.

class Border:

    def __init__(self, id, upstream, downstream, max_capacity):
        self.id = id
        self.upstream = upstream
        self.downstream = downstream
        self.max_flow = 0
        self.min_flow = 0
        self.max_capacity = max_capacity
