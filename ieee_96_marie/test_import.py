import grid2op

env = grid2op.make("ieee_96_marie")

obs = env.reset()

print(obs.load_p)
print(obs.gen_p)