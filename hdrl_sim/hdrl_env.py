
from xuance.environment import RawEnvironment
from gym import spaces
import numpy as np
from pathlib import Path
from gym.spaces import Box
from xuance.common import get_configs
from hdrl_sim.walker_delta_net import WalkerDeltaNet
from hdrl_sim.subdomain_netenv import SubdomainNetEnv
from xuance.torch.utils.operations import set_seed
from hdrl_sim.algorithm.ppoagent import GNNPPOCLIP_Agent
from xuance.environment import make_envs
from xuance.common import create_directory, get_time_string
import os
import socket
import argparse
import swanlab as wandb
# import wandb
from torch.utils.tensorboard import SummaryWriter

class HdrlEnv(RawEnvironment):
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4} # Example render modes
    def __init__(self, env_config):
        super(HdrlEnv, self).__init__()
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config/config.yaml")
        configs_dict = get_configs(file_dir=config_path)
        low_env_configs = argparse.Namespace(**configs_dict)
        
        self.env_id = env_config.env_id  # The environment id.
        self.observation_space = Box(-np.inf, np.inf, shape=[18, ])  # Define observation space.
        self.action_space = spaces.Discrete(4)  # Define action space. In this example, the action space is continuous.
        self.max_episode_steps = 32  # The max episode length.
        self._current_step = 0  # The count of steps of current episode.
        
        self.walker_delta_net = WalkerDeltaNet()

        
        self.configs = env_config
        # create neural network for each domain
        self.ndomains = len(self.walker_delta_net.all_domains)
        # init all domians' nn model
        
        from xuance.environment import REGISTRY_ENV
        REGISTRY_ENV[low_env_configs.env_name] = SubdomainNetEnv
        set_seed(low_env_configs.seed)
        envs = []
        agents = []
        # 使用 setattr 而不是字典语法
        setattr(low_env_configs, 'num_of_orbit', self.walker_delta_net.num_of_orbit)
        setattr(low_env_configs, 'sat_of_orbit', self.walker_delta_net.sat_of_orbit)
        
        for i in range(self.ndomains):
            setattr(low_env_configs, 'graph', self.walker_delta_net.get_subdomain_graph(i))
            
            setattr(low_env_configs, 'log_dir', low_env_configs.log_dir + f"domain_{i}/")
            setattr(low_env_configs, 'model_dir', low_env_configs.model_dir + f"domain_{i}/")
            env = make_envs(low_env_configs)
            envs.append(env)
            agents.append(GNNPPOCLIP_Agent(low_env_configs, envs[i]))

        self.envs = envs
        self.agents = agents
        
        # Create logger.
        time_string = get_time_string()
        if env_config.logger == "tensorboard":
            log_dir = os.path.join(os.getcwd(), env_config.log_dir, time_string)
            create_directory(log_dir)
            self.writer = SummaryWriter(log_dir)
            self.use_wandb = False
        elif env_config.logger == "wandb":
            config_dict = vars(env_config)
            log_dir = env_config.log_dir
            wandb_dir = Path(os.path.join(os.getcwd(), env_config.log_dir))
            create_directory(log_dir)
            wandb.init(config=config_dict,
                       project=env_config.project_name,
                       entity=env_config.wandb_user_name,
                       notes=socket.gethostname(),
                       dir=wandb_dir,
                       group=env_config.env_id,
                       job_type=env_config.agent,
                       name=time_string,
                       reinit=True,
                       settings=wandb.Settings(start_method="fork")
                       )
            # os.environ["WANDB_SILENT"] = "True"
            self.use_wandb = True
        else:
            raise AttributeError("No logger is implemented.")

        self.log_dir = log_dir
        
    def log_infos(self, info: dict, x_index: int):
        """
        info: (dict) information to be visualized
        n_steps: current step
        """
        if self.use_wandb:
            for k, v in info.items():
                if v is None:
                    continue
                wandb.log({k: v}, step=x_index)
        else:
            for k, v in info.items():
                if v is None:
                    continue
                try:
                    self.writer.add_scalar(k, v, x_index)
                except:
                    self.writer.add_scalars(k, v, x_index)

    def init_neural_network(self):
        temp_model = []
        for i in range(self.ndomains):
            temp_dqn = 0 # NeuralNetwork(i, self.nnodes, self.input_q_size)
            temp_model.append(temp_dqn)
        return temp_model
        

    def reset(self, **kwargs):  # Reset your environment.
        self._current_step = 0
        
        self.walker_delta_net.reset_net()
        for env in self.envs:
            env.reset()
        
        for agent in self.agents:
            agent.memory.clear()
        
        return self.observation_space.sample(), {}

    def step(self, action, flow):  # Run a step with an action.
        
        # take action. distribute flows based on action
        # parse action as the path for the flow
        flow.path = action
        # assign_flows_to_network
        self.walker_delta_net.assign_flows_to_network([flow])
        rewards = np.random.random()
        
        observation = self.walker_delta_net.get_domain_state(flow.current_domain)
        terminated = False
        truncated = False if self._current_step < self.max_episode_steps else True
        info = {}
        return observation, rewards, terminated, truncated, info

    def render(self, *args, **kwargs):  # Render your environment and return an image if the render_mode is "rgb_array".
        return np.ones([64, 64, 64])

    def close(self):  # Close your environment.
        return
    
    def router(self):
        pass
    
    # update survival time and remove expired flows
    def update_flows(self):
        self.walker_delta_net.traffic_manager.update_flow_survival_times()
        self.walker_delta_net.update_isl_state()
    
    def inject_new_flows(self):
        # self.walker_delta_net.randomGenFlows()
        # self.walker_delta_net.randomGenDomainflows()
        # self.walker_delta_net.inject_flows()
        # to_be_injected_flows = self.walker_delta_net.traffic_manager.tobe_assigned_flows

        # to_be_injected_flows = self.walker_delta_net.traffic_manager.tobe_assigned_domain_flows

        for idx in range(self.ndomains):
            self.walker_delta_net.gen_each_domain_flows(idx, arrival_rate=30)
            to_be_injected_flows = self.walker_delta_net.traffic_manager.tobe_assigned_flows
            
            self.agents[idx].train(to_be_injected_flows, idx)
            self.walker_delta_net.traffic_manager.merge_assigned_flows()
            
     
    def update_whole_network_state(self):
        self.update_flows()
        self.inject_new_flows()
        

    
    def upload_episode_rewards(self):
        
        for idx, agent in enumerate(self.agents):
            # agent.save_model("final_train_model.pth")
            
            episode_info = {}
            infos = agent.envs.buf_info.copy()
            if self.use_wandb:
                episode_info["Episode-step/agent-%d" % idx] = infos[0]["episode_step"]
                episode_info["Train-Episode-Rewards/agent-%d" % idx] = infos[0]["episode_score"]
                self.log_infos(episode_info, agent.current_step)
            else:
                
                episode_info["Episode-Steps"] = {"agent-%d" % idx: infos[0]["episode_step"]}
                episode_info["Train-Episode-Rewards"] = {"agent-%d" % idx: infos[0]["episode_score"]}
                self.log_infos(episode_info, agent.current_step)

                # episode_info["Episode-step/agent-%d" % idx] = infos[0]["episode_step"]
                # episode_info["Train-Episode-Rewards/agent-%d" % idx] = infos[0]["episode_score"]
                # agent.log_infos(episode_info, agent.current_step)

        self.get_edges_utilization()

    def save_lower_level_model(self):
        for idx, agent in enumerate(self.agents):
            agent.save_model("final_train_model.pth")
            
    # get all edges' utilization
    def get_edges_utilization(self):
        utilization = self.walker_delta_net.get_edges_utilization()
        # print(utilization)