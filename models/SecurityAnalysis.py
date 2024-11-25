# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
# This file is part of RL-filtering, a project for bid filtering in European balancing platforms.

import xpress as xp

def security_analysis(env, thermal_limits, gen_pmax, gen_p, no_redispatch_gen, contingencies, max_vol, mode="sa"):
    # xp.controls.outputlog = 0
    xp.setOutputEnabled(False)

    grid = env.backend._grid
    backend = env.backend
    gens = grid.get_generators()
    loads = grid.get_loads()
    lines = grid.get_lines()
    trafos = grid.get_trafos()
    trafo_delta = len(lines) #index offset for trafos - they are treated separately as they have different attributes

    sa = xp.problem("Security Analysis")
    sa.setControl({'presolve': True, 'threads': 1, 'outputlog': 0, 'concurrentthreads': 0, 'mipthreads': 0})

    #defining variables
    va_bus = {i: xp.var(name="va_" + str(i), lb=-xp.infinity, vartype=xp.continuous) for i in range(grid.nb_bus())}
    sa.addVariable(va_bus) #voltage angles for all buses

    loadShedding = {load.id: xp.var(lb=0, ub=load.target_p_mw, name="loadShedding_" + str(load.id),
                                    vartype=xp.continuous) for load in grid.get_loads()}
    sa.addVariable(loadShedding)

    pfl_lines = {
        (line.id, line.bus_or_id, line.bus_ex_id): xp.var(lb=-thermal_limits[line.id], ub=thermal_limits[line.id],
                                                          name="pfl_" + str(line.id), vartype=xp.continuous) for line in
        lines}
    pfl_trafos = {
        (trafo.id + trafo_delta, trafo.bus_hv_id, trafo.bus_lv_id): xp.var(lb=-thermal_limits[trafo.id + trafo_delta],
                                                                           ub=thermal_limits[trafo.id + trafo_delta],
                                                                           name="pfl_" + str(trafo.id + trafo_delta),
                                                                           vartype=xp.continuous) for trafo in trafos}
    pfl = {**pfl_lines, **pfl_trafos} #power flow for all lines and trafos
    sa.addVariable(pfl)

    dpgp_gen = {gen.id: xp.var(lb=0, name="dpgp_" + str(gen.id),
                               vartype=xp.continuous) for gen in grid.get_generators()}  # upward redispatch
    sa.addVariable(dpgp_gen)
    dpgn_gen = {gen.id: xp.var(lb=-xp.infinity, ub=0, name="dpgn_" + str(gen.id),
                               vartype=xp.continuous) for gen in grid.get_generators()}  # downward redispatch
    sa.addVariable(dpgn_gen)
    penalize_vol = xp.var(lb=max_vol, ub=xp.infinity, name="penalize_vol", vartype=xp.continuous)
    sa.addVariable(penalize_vol) #variable to limit the volume of redispatch

    # --N mode

    # balance constraint (Kirchhoff's current law)
    balance_constr = {}
    bus_or_lines = {}
    bus_ex_lines = {}
    all_bus_gens = {}
    all_bus_loads = {}
    for i in range(grid.nb_bus()):
        # for each buses, find all the lines connected to the bus and their direction + loads and gens on the bus
        or_lines = []
        ex_lines = []

        for line in lines:
            if line.bus_or_id == i:
                or_lines.append((line.id, line.bus_or_id, line.bus_ex_id))
            if line.bus_ex_id == i:
                ex_lines.append((line.id, line.bus_or_id, line.bus_ex_id))
        for trafo in trafos:
            if trafo.bus_hv_id == i:
                or_lines.append((trafo.id + trafo_delta, trafo.bus_hv_id, trafo.bus_lv_id))
            if trafo.bus_lv_id == i:
                ex_lines.append((trafo.id + trafo_delta, trafo.bus_hv_id, trafo.bus_lv_id))

        bus_gens = []
        bus_loads = []
        for gen in gens:
            if gen.bus_id == i:
                bus_gens.append(gen.id)
        for load in loads:
            if load.bus_id == i:
                bus_loads.append(load.id)

        bus_or_lines[i] = or_lines
        bus_ex_lines[i] = ex_lines
        all_bus_gens[i] = bus_gens
        all_bus_loads[i] = bus_loads

        flows_in = xp.Sum(pfl[line] for line in ex_lines)
        flows_out = xp.Sum(pfl[line] for line in or_lines)
        np_gen = xp.Sum(gen_p[gen_id] + dpgp_gen[gen_id] + dpgn_gen[gen_id] for gen_id in bus_gens)
        np_load = xp.Sum(- loads[load_id].target_p_mw + loadShedding[load_id] for load_id in bus_loads)

        balance_constr[i] = xp.constraint(flows_out - flows_in == np_gen + np_load, name="balance_bus_" + str(i))
    sa.addConstraint(balance_constr)

    # reference bus for voltage angles
    sa.addConstraint(va_bus[12] == 0)

    # generator limits
    sa.addConstraint(gen_p[gen.id] + dpgp_gen[gen.id] + dpgn_gen[gen.id] >= 0  # pmin of generators are set to 0
                     for gen in gens)
    sa.addConstraint(gen_p[gen.id] + dpgp_gen[gen.id] + dpgn_gen[gen.id] <= gen_pmax[gen.id]
                     for gen in gens)
    sa.addConstraint(dpgp_gen[gen_id] <= 0 for gen_id in no_redispatch_gen)
    sa.addConstraint(dpgn_gen[gen_id] >= 0 for gen_id in no_redispatch_gen)
    sa.addConstraint(penalize_vol >= xp.Sum(dpgp_gen))
    sa.addConstraint(
        penalize_vol >= -xp.Sum(dpgn_gen))  # penalize_vol is larger than upward and downward redispatch volumes

    # flow_limits
    sa.addConstraint(pfl[(line.id, line.bus_or_id, line.bus_ex_id)] == (1 / (line.x_pu)) * (
                va_bus[line.bus_or_id] - va_bus[line.bus_ex_id])
                     for line in lines)
    sa.addConstraint(
        pfl[(trafo.id + trafo_delta, trafo.bus_hv_id, trafo.bus_lv_id)] == (1 / (trafo.x_pu * trafo.ratio)) * (
                    va_bus[trafo.bus_hv_id] - va_bus[trafo.bus_lv_id])
        for trafo in trafos)

    # --N-1 mode preventive

    ##defining variables
    va_ic_bus = {(i, c): xp.var(name="va_ic_" + str(i) + "_" + str(c), lb=-xp.infinity, vartype=xp.continuous)
                 for i in range(grid.nb_bus()) for c in contingencies}
    sa.addVariable(va_ic_bus) #voltage angles for each bus in all N-1 states

    pfl_ic_lines = {
        ((line.id, line.bus_or_id, line.bus_ex_id), c): xp.var(lb=-thermal_limits[line.id], ub=thermal_limits[line.id],
                                                               name="pfl_ic_" + str(line.id) + "_" + str(c),
                                                               vartype=xp.continuous) for line in lines for c in
        contingencies}
    pfl_ic_trafos = {((trafo.id + trafo_delta, trafo.bus_hv_id, trafo.bus_lv_id), c): xp.var(
        lb=-thermal_limits[trafo.id + trafo_delta], ub=thermal_limits[trafo.id + trafo_delta],
        name="pfl_ic_" + str(trafo.id + trafo_delta) + "_" + str(c), vartype=xp.continuous) for trafo in trafos for c in
                     contingencies}
    pfl_ic = {**pfl_ic_lines, **pfl_ic_trafos} #power flows for all N-1 states
    sa.addVariable(pfl_ic)

    # balance (Kirchhoff's current law)
    balance_constr_ic = {}
    for cont in contingencies:

        for i in range(grid.nb_bus()):
            or_lines = bus_or_lines[i]
            ex_lines = bus_ex_lines[i]
            bus_gens = all_bus_gens[i]
            bus_loads = all_bus_loads[i]

            flows_in_ic = xp.Sum(pfl_ic[(line, cont)] for line in ex_lines)
            flows_out_ic = xp.Sum(pfl_ic[(line, cont)] for line in or_lines)
            np_gen = xp.Sum(gen_p[gen_id] + dpgp_gen[gen_id] + dpgn_gen[gen_id] for gen_id in bus_gens)
            np_load = xp.Sum(- loads[load_id].target_p_mw + loadShedding[load_id] for load_id in bus_loads)

            balance_constr_ic[(i, cont)] = xp.constraint(flows_out_ic - flows_in_ic == np_gen + np_load,
                                                         name="balance_bus_" + str(i) + "_" + str(cont))

            # reference bus for voltage angles
            sa.addConstraint(va_ic_bus[12, cont] == 0)

        # flow_limits
        is_not_cont_line = [line.id != cont for line in lines]
        is_not_cont_trafo = [(trafo.id + trafo_delta) != cont for trafo in trafos]
        is_not_cont = is_not_cont_line + is_not_cont_trafo

        sa.addConstraint(
            pfl_ic[((line.id, line.bus_or_id, line.bus_ex_id), cont)] == is_not_cont[line.id] * (1 / line.x_pu) * (
                    va_ic_bus[(line.bus_or_id, cont)] - va_ic_bus[(line.bus_ex_id, cont)]) for line in lines)
        sa.addConstraint(pfl_ic[((trafo.id + trafo_delta, trafo.bus_hv_id, trafo.bus_lv_id), cont)] == is_not_cont[
            trafo.id + trafo_delta] * (1 / (trafo.x_pu * trafo.ratio)) * (
                                 va_ic_bus[(trafo.bus_hv_id, cont)] - va_ic_bus[(trafo.bus_lv_id, cont)]) for trafo in
                         trafos)
    sa.addConstraint(balance_constr_ic)

    # objective function
    sa.setObjective(xp.Sum(
        (backend.gen_cost_per_MW[gen.id]) * dpgp_gen[gen.id] - (180 - backend.gen_cost_per_MW[gen.id]) * dpgn_gen[
            gen.id] for gen in gens)
                    + 33000 * (xp.Sum(loadShedding[load.id] for load in loads)) + 32990 * (penalize_vol - max_vol)
                    + 0.01 * xp.Sum(  # avoid netting with no added value
        backend.gen_cost_per_MW[gen.id] * (dpgp_gen[gen.id] + dpgn_gen[gen.id]) for gen in gens)
        , sense=xp.minimize)

    sa.solve()

    if sa.getProbStatusString() == "lp_infeas":
        sa.write("sa", "lp")
        print("INFEASIBLE")
        print(infeasible)  # stops the run if a case is infeasible

    # output
    cm_costs = sum((backend.gen_cost_per_MW[gen.id] + 1) * sa.getSolution(dpgp_gen)[gen.id] + (
                backend.gen_cost_per_MW[gen.id] - 1) * sa.getSolution(dpgn_gen)[gen.id] for gen in gens) \
               + 33000 * sum(sa.getSolution(loadShedding[load.id]) for load in loads) + 32990 * (
                           sa.getSolution(penalize_vol) - max_vol)
    # the last term penalizes redispatch volumes that surpass the allowed limit

    if mode == "basecase":  # create a secure base case for baseline_filtering
        final_prod = [sum(x) for x in zip(gen_p, sa.getSolution(dpgp_gen).values(), sa.getSolution(dpgn_gen).values())]
        return final_prod
    else:  # filtering
        return cm_costs
