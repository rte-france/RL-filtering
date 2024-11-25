import pandapower as pp
import os
path = "h20200101_0000-m"
fn_ = "h20200101_0000.m"
grid = pp.converter.from_mpc(os.path.join(path, fn_))
pp.to_json(grid, "grid_v0.json")
