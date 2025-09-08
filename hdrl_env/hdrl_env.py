from astropy import units
from xuance.environment import RawEnvironment
import gym
from gym import spaces
import networkx as nx
import numpy as np
import math
from gym.spaces import Box
from xuance.common import get_configs, recursive_dict_update
from .walker_delta_net import WalkerDeltaNet
from copy import deepcopy
# 流程
# 1. 初始化环境 


class SingleEnvXC(RawEnvironment):
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4} # Example render modes
    def __init__(self, env_config):
        super(SingleEnvXC, self).__init__()
        self.env_id = env_config.env_id  # The environment id.
        self.observation_space = Box(-np.inf, np.inf, shape=[18, ])  # Define observation space.
        self.action_space = Box(-np.inf, np.inf, shape=[5, ])  # Define action space. In this example, the action space is continuous.
        self.max_episode_steps = 32  # The max episode length.
        self._current_step = 0  # The count of steps of current episode.
        
        self.init_walker_delta_net = WalkerDeltaNet()
        self.walker_delta_net = deepcopy(self.init_walker_delta_net)
        
        # create neural network for each domain
        self.ndomains = len(self.walker_delta_net.all_domains)
        # init all domians' nn model
        self.nnmodel = self.init_neural_network()
        # based on possion distribution to generate traffic
        self.walker_delta_net.randomGenFlows()
        
    def init_neural_network(self):
        temp_model = []
        for i in range(self.ndomains):
            temp_dqn = 0 # NeuralNetwork(i, self.nnodes, self.input_q_size)
            temp_model.append(temp_dqn)
        return temp_model
        

    def reset(self, **kwargs):  # Reset your environment.
        self._current_step = 0
        return self.observation_space.sample(), {}

    def step(self, action):  # Run a step with an action.
        self._current_step += 1
        observation = self.observation_space.sample()
        rewards = np.random.random()
        terminated = False
        truncated = False if self._current_step < self.max_episode_steps else True
        info = {}
        return observation, rewards, terminated, truncated, info

    def render(self, *args, **kwargs):  # Render your environment and return an image if the render_mode is "rgb_array".
        return np.ones([64, 64, 64])

    def close(self):  # Close your environment.
        return
    
    def router(self, )