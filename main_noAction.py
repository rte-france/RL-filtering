from sb3_contrib import TQC
from models.FilteringEnv_testDataParams import FilteringEnv_testDataParams

class DoNothingAgent:
    # agent that does not filter -> price delta is null

    def __init__(self, gym_env):
        self.action_space = gym_env.action_space
        self._nb_bid_max = gym_env._nb_bid_max

    def predict(self, obs, deterministic):
        delta = [0]*self._nb_bid_max
        return delta

def evaluate(env, model):
    #evaluate Do Nothing or Proposed filtering agent over 52 weeks

    sum_reward = 0
    for i in range(52):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            sum_reward += reward
    print("total reward", sum_reward)


if __name__=='__main__':
    mode = "DoNothing" #"ProposedFiltering"
    env = FilteringEnv_testDataParams(max_vol=200) #max redispatch volume

    if mode == "DoNothing":
        model = DoNothingAgent(env)
    else:
        model = TQC.load("saved_ex/TQC_LR_00003_NNsize_50_volLim200_130000_steps.zip", env=env) #load saved model
    evaluate(env, model)
