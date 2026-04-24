import gymnasium as gym #gymnasium is a library for creating and interacting with gym environments
from gymnasium import spaces #spaces is a module for creating and interacting with gym spaces
import numpy as np #numpy is a library for numerical computing

class trading_env(gym.Env):
    def __init__(self, features, returns):
        super().__init__()
        #Get the values so it gets rid of column and row names. This is better for RL models because they suck at reading row/col names
        self.features =  features.values
        #It is the number how how the INDEX changed 
        self.returns = returns.values
        self.tickers = ['SPY', 'QQQ', 'TLT']
        self.numAssets = self.returns.shape[1] #number of assets in the portfolio
        self.observationDimension = self.features.shape[1] + self.numAssets #observation space is the range of possible observations the agent can see

        self.action_space = spaces.Box(low = -1.0, high = 1.0, shape = (self.numAssets,), dtype = np.float32) #action space is the range of possible actions the agent can take
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (self.observationDimension,), dtype = np.float32) #observation space is the range of possible observations the agent can see

        self.max_steps = len(self.returns)

        self.reset() #reset the environment to the initial state


    #normalize the weights to be between 0 and 1 and sum to 1
    def normalizingWeights(self, weights):
        w = (weights + 1) / 2 #shift the weights to be between 0 and 1
        w = np.clip(w, 0, 1) #clip the weights to be between 0 and 1

        if w.sum() == 0:
            w = np.ones(self.numAssets) / self.numAssets #if the weights sum to 0, set the weights to be equal
        else:
            w = w / w.sum() #if the weights sum to something else, normalize the weights to sum to 1

        return w

    #get the observation of the current state
    def getObservation(self):
        return np.concatenate([self.features[self.t], self.weights]).astype(np.float32) #concatenate the features and the weights to get the observation

    #reset the environment to the initial state
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        self.weights = np.ones(self.numAssets) / self.numAssets #set the weights to be equal
        self.t = 0 #set the time step to 0
        self.portfolioReturns = [] 
        return self.getObservation(), {} 


    #step the environment forward by one time step
    def step(self, action):
        newWeights = self.normalizingWeights(action)
        previousWeights = self.weights
        assetReturns = self.returns[self.t] #get the asset returns for the current time step
        window = self.returns[max(0, self.t - 60):self.t+1] #get the window of returns for the last 20 time steps
        volatility = np.std(window) + 1e-8 #add a small epsilon to avoid division by zero
        portfolioReturn = np.dot(assetReturns, newWeights) #calculate the portfolio return (profit or loss)
        turnover = np.sum(np.abs(newWeights - previousWeights)) #The turnover punishes protfolio weight changes 

        if self.t >= 60:
            vol_term = 0.2 * (portfolioReturn / volatility)
        else:
            vol_term = 0.0  # suppress noisy term early in episode

        reward = portfolioReturn - (0.0003 * turnover) + vol_term
        self.portfolioReturns.append(portfolioReturn)
        self.weights = newWeights
        self.t += 1
        done = self.t >= len(self.returns) - 1 #done is true if the time step is greater than the number of time steps
        if not done:
            observation = self.getObservation()
        else:
            observation = np.zeros(self.observationDimension, dtype = np.float32)

        return observation, reward, done, False, {} #return the observation(next state), reward, done, and info





