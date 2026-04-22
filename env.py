import gymnasium as gym
import numpy as np


#inherits from gym.Env
class training(gym.Env):

    def __init__(self, data_split):
        super(TradingEnv, self).__init__()

    self.data = data_split
    self.tickers = ['SPY', 'QQQ', 'TLT']

    #This rules for what the model should do: It should return array of three numbers between 0 and 1
    #Think of this as the video game controller 
    self.action_space = spaces.Box(low=0.0, high=1.0, shape=(len(self.tickers),), dtype=np.float32)

    #This is the observation space
    #Think of this like the TV for where we can view the game
    self.obs_shape = 15 * len(self.tickers) 
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32)

    #Trackers
    self.unique_dates = sorted(self.data.index.unique())
    self.curr_step = 0
    self.max_steps = 252
    self.balance = 100000

    def reset():

    def step(self, action):
        today_date = self.dates[self.curr_step]
        return








