import xpress as xp

def security_analysis(env, thermal_limits, gen_pmax, gen_p, no_redispatch_gen, contingencies, max_vol, gen_correspondance, load_correspondance,greedyTSO = []):
    #xp.controls.outputlog = 0
    xp.setOutputEnabled(False)

    grid = env.backend._grid
    backend = env.backend
    gens = grid.get_generators()
    loads = grid.get_loads()
    lines = grid.get_lines()
    trafos = grid.get_trafos()
    trafo_delta = len(lines)

    sa = xp.problem("Security Analysis")
    sa.setControl({'presolve': True, 'threads': 1, 'outputlog': 0, 'lpthreads':0, 'mipthreads':0})

    ##defining variables
    va_bus = {i: xp.var(name = "va_"+str(i), lb = -xp.infinity,vartype = xp.continuous) for i in range(grid.nb_bus())}
    sa.addVariable(va_bus)

    loadShedding = {load.id: xp.var(lb = 0, ub = load.target_p_mw ,name = "loadShedding_"+str(load.id),
                                    vartype = xp.continuous) for load in grid.get_loads()}
    sa.addVariable(loadShedding)

    pfl_lines = {(line.id,line.bus_or_id,line.bus_ex_id): xp.var(lb = -thermal_limits[line.id],ub = thermal_limits[line.id],
                                 name = "pfl_"+str(line.id), vartype = xp.continuous) for line in lines}
    pfl_trafos = {(trafo.id+trafo_delta,trafo.bus_hv_id,trafo.bus_lv_id): xp.var(lb = -thermal_limits[trafo.id + trafo_delta],ub = thermal_limits[trafo.id + trafo_delta],
                                 name = "pfl_"+str(trafo.id+trafo_delta), vartype = xp.continuous) for trafo in trafos}
    pfl = {**pfl_lines,**pfl_trafos}
    sa.addVariable(pfl)

    dpgp_gen = {gen.id: xp.var(lb = 0,name = "dpgp_"+str(gen.id),
                               vartype = xp.continuous) for gen in grid.get_generators()} #positive redispatch
    sa.addVariable(dpgp_gen)
    dpgn_gen = {gen.id: xp.var(lb= -xp.infinity ,ub = 0 ,name = "dpgn_"+str(gen.id),
                               vartype = xp.continuous) for gen in grid.get_generators()} #negative redispatch
    sa.addVariable(dpgn_gen)
    penalize_vol = xp.var(lb= max_vol ,ub = xp.infinity ,name = "penalize_vol",vartype = xp.continuous)
    sa.addVariable(penalize_vol)

    #--N mode

    #balance constraint (loi des noeuds)
    balance_constr = {}
    bus_or_lines = {}
    bus_ex_lines = {}
    all_bus_gens = {}
    all_bus_loads = {}
    for i in range(grid.nb_bus()):
        or_lines = []
        ex_lines = []
        
        for line in lines:
            if line.bus_or_id == i:
                or_lines.append((line.id,line.bus_or_id,line.bus_ex_id))
            if line.bus_ex_id == i:
                ex_lines.append((line.id,line.bus_or_id,line.bus_ex_id))
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

        balance_constr[i] = xp.constraint(flows_out - flows_in == np_gen + np_load, name = "balance_bus_"+str(i))
    sa.addConstraint(balance_constr)

    #reference bus for voltage angles
    sa.addConstraint(va_bus[12]==0)

    #generator limits
    sa.addConstraint(gen_p[gen.id] + dpgp_gen[gen.id] + dpgn_gen[gen.id] >= 0 #no Pmin
                     for gen in gens)
    sa.addConstraint(gen_p[gen.id] + dpgp_gen[gen.id] + dpgn_gen[gen.id] <= gen_pmax[gen.id]
                     for gen in gens)
    sa.addConstraint(dpgp_gen[gen_id] <= 0 for gen_id in no_redispatch_gen)
    sa.addConstraint(dpgn_gen[gen_id] >= 0 for gen_id in no_redispatch_gen)
    sa.addConstraint(penalize_vol >= xp.Sum(dpgp_gen))
    sa.addConstraint(penalize_vol>=-xp.Sum(dpgn_gen)) #penalize volume is higher than max of redispatch values

    #flow_limits
    sa.addConstraint(pfl[(line.id, line.bus_or_id, line.bus_ex_id)] == (1 / (line.x_pu)) * (
                va_bus[line.bus_or_id] - va_bus[line.bus_ex_id]) for line in lines)
    sa.addConstraint(
        pfl[(trafo.id + trafo_delta, trafo.bus_hv_id, trafo.bus_lv_id)] == (1 / (trafo.x_pu * trafo.ratio)) * (
                    va_bus[trafo.bus_hv_id] - va_bus[trafo.bus_lv_id]) for trafo in trafos)

    #--N-1 mode preventive

    ##defining variables

    va_ic_bus = {(i,c): xp.var(name = "va_ic_"+str(i)+"_"+str(c), lb = -xp.infinity,vartype = xp.continuous)
                 for i in range(grid.nb_bus()) for c in contingencies}
    sa.addVariable(va_ic_bus)

    pfl_ic_lines = {((line.id,line.bus_or_id,line.bus_ex_id),c): xp.var(lb = -thermal_limits[line.id],ub = thermal_limits[line.id],
                                 name = "pfl_ic_"+str(line.id)+"_"+str(c), vartype = xp.continuous) for line in lines for c in contingencies}
    pfl_ic_trafos = {((trafo.id+ trafo_delta,trafo.bus_hv_id,trafo.bus_lv_id),c): xp.var(lb = -thermal_limits[trafo.id + trafo_delta],ub = thermal_limits[trafo.id + trafo_delta],
                                 name = "pfl_ic_"+str(trafo.id+trafo_delta)+"_"+str(c), vartype = xp.continuous) for trafo in trafos for c in contingencies}
    pfl_ic = {**pfl_ic_lines,**pfl_ic_trafos}
    sa.addVariable(pfl_ic)

    #balance - loi des noeuds
    balance_constr_ic = {}
    for cont in contingencies:

        for i in range(grid.nb_bus()):
            or_lines = bus_or_lines[i]
            ex_lines = bus_ex_lines[i]
            bus_gens = all_bus_gens[i]
            bus_loads = all_bus_loads[i]

            flows_in_ic = xp.Sum(pfl_ic[(line,cont)] for line in ex_lines)
            flows_out_ic = xp.Sum(pfl_ic[(line,cont)] for line in or_lines)
            np_gen = xp.Sum(gen_p[gen_id] + dpgp_gen[gen_id] + dpgn_gen[gen_id] for gen_id in bus_gens)
            np_load = xp.Sum(- loads[load_id].target_p_mw + loadShedding[load_id] for load_id in bus_loads)

            balance_constr_ic[(i,cont)] = xp.constraint(flows_out_ic - flows_in_ic == np_gen + np_load, name="balance_bus_" + str(i)+"_"+str(cont))

            # reference bus for voltage angles
            sa.addConstraint(va_ic_bus[12,cont] == 0)

        # flow_limits
        is_not_cont_line = [line.id != cont for line in lines]
        is_not_cont_trafo = [(trafo.id +trafo_delta) != cont for trafo in trafos]
        is_not_cont = is_not_cont_line + is_not_cont_trafo

        sa.addConstraint(pfl_ic[((line.id, line.bus_or_id, line.bus_ex_id),cont)] == is_not_cont[line.id]*(1 / line.x_pu)*(
                    va_ic_bus[(line.bus_or_id,cont)] - va_ic_bus[(line.bus_ex_id,cont)]) for line in lines)
        sa.addConstraint(pfl_ic[((trafo.id + trafo_delta, trafo.bus_hv_id, trafo.bus_lv_id),cont)] == is_not_cont[trafo.id + trafo_delta]*(1 / (trafo.x_pu*trafo.ratio))*(
                    va_ic_bus[(trafo.bus_hv_id,cont)] - va_ic_bus[(trafo.bus_lv_id,cont)]) for trafo in trafos)
    sa.addConstraint(balance_constr_ic)

    #objective
    sa.setObjective(xp.Sum((backend.gen_cost_per_MW[gen.id])*dpgp_gen[gen.id]-(180-backend.gen_cost_per_MW[gen.id])*dpgn_gen[gen.id] for gen in gens)
                        + 33000*(xp.Sum(loadShedding[load.id] for load in loads)) + 33200*(penalize_vol-max_vol)
                    +0.01*xp.Sum(backend.gen_cost_per_MW[gen.id]*(sa.getSolution(dpgp_gen)[gen.id]+sa.getSolution(dpgn_gen)[gen.id]) for gen in gens), sense=xp.minimize)

    #sa.write("sa_new", "lp")
    sa.solve()

    if sa.getProbStatusString() == "lp_infeas":
        sa.write("sa", "lp")
        print("INFEASIBLE !")
        raise SystemExit
        print(fhiuhziufh)

    #output
    greedy_cm_costs = {}
    Vol_up = sum(sa.getSolution(dpgp_gen[gen.id]) for gen in gens)
    Vol_down = - sum(sa.getSolution(dpgn_gen[gen.id]) for gen in gens)
    for TSO in ["fr","de","be"]:
        TSO_cost= 0
        if Vol_up > max_vol and Vol_up> Vol_down:
            TSO_vol_up = 0
        elif Vol_down > max_vol:
            TSO_vol_down = 0
        for (k,v) in gen_correspondance.items():
            if k[0] == "3" and TSO == "fr":
                TSO_cost += (backend.gen_cost_per_MW[v]+1)*sa.getSolution(dpgp_gen)[v]+(backend.gen_cost_per_MW[v]-1)*sa.getSolution(dpgn_gen)[v]
                if Vol_up > max_vol and Vol_up> Vol_down:
                    TSO_vol_up += sa.getSolution(dpgp_gen)[v]
                elif Vol_down > max_vol:
                    TSO_vol_down -= sa.getSolution(dpgn_gen)[v]
            elif k[0] == "2" and TSO == "de":
                TSO_cost += (backend.gen_cost_per_MW[v] + 1) * sa.getSolution(dpgp_gen)[v] + (
                            backend.gen_cost_per_MW[v] - 1) * sa.getSolution(dpgn_gen)[v]
                if Vol_up > max_vol and Vol_up> Vol_down:
                    TSO_vol_up += sa.getSolution(dpgp_gen)[v]
                elif Vol_down > max_vol:
                    TSO_vol_down -= sa.getSolution(dpgn_gen)[v]
            elif k[0] == "1" and TSO == "be":
                TSO_cost += (backend.gen_cost_per_MW[v] + 1) * sa.getSolution(dpgp_gen)[v] + (
                            backend.gen_cost_per_MW[v] - 1) * sa.getSolution(dpgn_gen)[v]
                if Vol_up > max_vol and Vol_up> Vol_down:
                    TSO_vol_up += sa.getSolution(dpgp_gen)[v]
                elif Vol_down > max_vol:
                    TSO_vol_down -= sa.getSolution(dpgn_gen)[v]
        if Vol_up > max_vol and Vol_up> Vol_down:
            TSO_cost += (Vol_up - max_vol) * 33200 * TSO_vol_up/Vol_up
        elif Vol_down > max_vol:
            TSO_cost += (Vol_down - max_vol) * 33200 * TSO_vol_down/Vol_down
        for (k,v) in load_correspondance.items():
            if k[4]=="3" and TSO == "fr":
                TSO_cost += 33000*sa.getSolution(loadShedding[v])
            elif k[4]=="2" and TSO == "de":
                TSO_cost += 33000*sa.getSolution(loadShedding[v])
            elif k[4]=="1" and TSO == "be":
                TSO_cost += 33000*sa.getSolution(loadShedding[v])
        greedy_cm_costs[TSO] = TSO_cost

    cm_costs = sum((backend.gen_cost_per_MW[gen.id]+1)*sa.getSolution(dpgp_gen)[gen.id]+(backend.gen_cost_per_MW[gen.id]-1)*sa.getSolution(dpgn_gen)[gen.id] for gen in gens)\
                + 33000*sum(sa.getSolution(loadShedding[load.id]) for load in loads) + 33200*(sa.getSolution(penalize_vol)-max_vol) #pénaliser les volumes qui dépassent 400

    return cm_costs, greedy_cm_costs