# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
# This file is part of RL-filtering, a project for bid filtering in European balancing platforms.

import xpress as xp


def Clearing(zones, borders):
    #runs the market clearing

    xp.setOutputEnabled(False)

    clearing = xp.problem("Clearing")
    clearing.setControl({'presolve': True, 'threads': 1, 'outputlog': 0, 'concurrentthreads': 0, 'mipthreads': 0})

    all_bids = {}
    for zone in zones:
        all_bids[zone.id] = zone.bids + zone.demand

    ##define variables
    accepted_xp = {}
    for zone in zones:
        accepted_zone = {bid.id: xp.var(lb=0, ub=bid.qmax, name="new_power_" + bid.id, vartype=xp.continuous) for bid in
                         all_bids[zone.id]}
        accepted_xp[zone.id] = accepted_zone
    clearing.addVariable(accepted_xp)

    exchange_xp = {border.id: xp.var(lb=border.min_flow, ub=border.max_flow, name="new_exchange_" + border.id,
                                     vartype=xp.continuous) for border in borders} #exchanges are limited by ATCs
    clearing.addVariable(exchange_xp)

    ##add balance constraints
    balance_constr = {}
    for zone in zones:
        net_positions_LHS = xp.Sum(bid.isSell * accepted_xp[zone.id][bid.id] for bid in all_bids[zone.id])

        net_positions_RHS = xp.Sum(border["border_sign"] * exchange_xp[border["border_id"]] for border in zone.borders)

        balance_constr[zone.id] = xp.constraint(net_positions_LHS == net_positions_RHS, name="balance_zone_" + zone.id)
        clearing.addConstraint(balance_constr[zone.id])

    # set objective
    sw = {}
    for zone in zones:
        sw[zone.id] = xp.Sum(accepted_xp[zone.id][bid.id] * bid.isSell * bid.price for bid in all_bids[zone.id])
    clearing.setObjective(xp.Sum(sw[z.id] for z in zones), sense=xp.minimize)

    # clearing.write("clearing", "lp")
    clearing.solve()

    market_welfare = 0
    for zone in zones:
        for bid in zone.bids:
            accepted_power = clearing.getSolution(accepted_xp)[zone.id][bid.id]
            market_welfare += -bid.isSell * accepted_power * bid.initial_price

    return market_welfare, clearing.getSolution(accepted_xp)
