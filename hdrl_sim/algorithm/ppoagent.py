
import torch
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from xuance.common import Union
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import Module
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.agents import OnPolicyAgent
from hdrl_sim.algorithm.ppopolicy import GNNActorCriticPolicy
from torch.distributed import destroy_process_group
from xuance.common import get_time_string, create_directory, RunningMeanStd, space2shape, EPS, Optional, Union
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import REGISTRY_Representation, REGISTRY_Learners, Module
from xuance.torch.utils import nn, NormalizeFunctions, ActivationFunctions, init_distributed_mode
from gym.spaces import Dict, Space
import numpy as np
from hdrl_sim.algorithm.ppolearner import GNNPPOCLIP_Learner
# from agent import Agent
from xuance.common import Optional, Union, DummyOnPolicyBuffer
import swanlab as wandb
class GNNPPOCLIP_Agent(OnPolicyAgent):
    """The implementation of PPO agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecEnv, SubprocVecEnv]):
        super(GNNPPOCLIP_Agent, self).__init__(config, envs)
        self.auxiliary_info_shape = {"old_logp": ()}
        self.edge_index = self.envs.get_env_attributes()
        self.memory = self._build_memory(self.auxiliary_info_shape)  # build memory
        self.policy = self._build_policy()  # build policy
        REGISTRY_Learners["gnnppocliplearner"] = GNNPPOCLIP_Learner
        self.learner = self._build_learner(self.config, self.policy)  # build learner
        
    
    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device

        # build representation.
        representation = self._build_representation_gnn(self.config.representation, self.observation_space, self.config)
        
        policy = GNNActorCriticPolicy(
            action_space=self.action_space, representation=representation,
            actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
            normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
            use_distributed_training=self.distributed_training,
            edge_index=self.edge_index)
        
        # build policy.
        # if self.config.policy == "Categorical_AC":
        #     policy = REGISTRY_Policy["Categorical_AC"](
        #         action_space=self.action_space, representation=representation,
        #         actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
        #         normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
        #         use_distributed_training=self.distributed_training)


        return policy
    
    def _build_representation_gnn(self, representation_key: str,
                              input_space: Optional[Space],
                              config: Namespace) -> Module:
        """Builds the representation module based on the configuration."""
        # if representation_name == "Basic_GAT":

        """
        Build representation for policies.

        Parameters:
            representation_key (str): The selection of representation, e.g., "Basic_MLP", "Basic_RNN", etc.
            input_space (Optional[Space]): The space of input tensors.
            config: The configurations for creating the representation module.

        Returns:
            representation (Module): The representation Module.
        """
        input_representations = dict(
            input_shape=space2shape(input_space),
            node_feature_dim=config.node_feature_dim if hasattr(config, "node_feature_dim") else 4,
            edge_feature_dim=config.edge_feature_dim if hasattr(config, "edge_feature_dim") else 6,
            node_hidden_feature_dim=config.node_hidden_feature_dim if hasattr(config, "node_hidden_feature_dim") else 4,
            hidden_sizes=config.representation_hidden_size if hasattr(config, "representation_hidden_size") else None,
            normalize=NormalizeFunctions[config.normalize] if hasattr(config, "normalize") else None,
            initialize=nn.init.orthogonal_,
            activation=ActivationFunctions[config.activation],
            fc_hidden_sizes=config.fc_hidden_sizes if hasattr(config, "fc_hidden_sizes") else None,
            device=self.device)
        
        
        representation = REGISTRY_Representation[representation_key](**input_representations)
        if representation_key not in REGISTRY_Representation:
            raise AttributeError(f"{representation_key} is not registered in REGISTRY_Representation.")
        return representation

    def get_aux_info(self, policy_output: dict = None):
        """Returns auxiliary information.

        Parameters:
            policy_output (dict): The output information of the policy.

        Returns:
            aux_info (dict): The auxiliary information.
        """
        aux_info = {"old_logp": policy_output['log_pi']}
        return aux_info


    def train(self, flows, *args):
        agent_id = args[0] if len(args) > 0 else 0
        done = args[1] if len(args) > 1 else False
        first_flow = flows[0]
        flow_attribute = {
            "flow_src": first_flow.source,
            "flow_des": first_flow.destination,
            'flow_bd': first_flow.flow_rate,
            'survival_time': first_flow.survival_time
        }
        self.envs.set_env_attributes(flow_attribute)
        self.envs.reset_obs()
        obs = self.envs.buf_obs
        num_flows = len(flows)
        
        # 处理所有flows
        for idx, flow in enumerate(flows):
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            policy_out = self.action(obs, return_dists=False, return_logpi=True)
            acts, value, logps = policy_out['actions'], policy_out['values'], policy_out['log_pi']
            next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)
            if idx == (num_flows - 1) and done:
                terminals = np.array([True]*self.n_envs)
            flow_path = infos[0].get('flow_path', [])
            flow.path = flow_path
            aux_info = self.get_aux_info(policy_out)
            self.memory.store(obs, acts, self._process_reward(rewards), value, terminals, aux_info)
            
            # 更新观察值以供下一个flow使用
            if (idx + 1) < num_flows:
                flow_attribute = {
                    "flow_src": flows[idx+1].source,
                    "flow_des": flows[idx+1].destination,
                    'flow_bd' : flows[idx+1].flow_rate,
                    'survival_time': first_flow.survival_time
                }
                self.envs.set_env_attributes(flow_attribute)
                self.envs.reset_obs()
                obs = self.envs.buf_obs


            self.returns = self.gamma * self.returns + rewards
            self.current_step += self.n_envs
            
            if idx == (num_flows - 1) and done:
                # 如果是最后一个flow且done为True，结束当前episode
                for i in range(self.n_envs):
                    
                    self.memory.finish_path(0.0, i)
        
        # # 所有flows处理完成后，检查是否需要训练
        # if self.memory.full:
        #     # 获取最终状态的价值估计用于bootstrap
        #     vals = self.get_terminated_values(obs)
        #     for i in range(self.n_envs):
        #         if terminals[i]:
        #             # 如果episode结束，最终价值为0
        #             self.memory.finish_path(0.0, i)
        #         else:
        #             # 如果episode未结束，使用价值函数估计
        #             self.memory.finish_path(vals[i], i)
            
        #     # 执行训练
        #     train_info = self.train_epochs(n_epochs=self.n_epochs)
        #     for k, v in train_info.items():
        #         if v is None:
        #             continue
        #         wandb.log({f"agent_{agent_id}/{k}": v}, step=self.current_step)
        #     self.memory.clear()

    def episode_train(self, agent_id):
        
        train_info = self.train_episode(n_epochs=self.n_epochs)
        for k, v in train_info.items():
            if v is None:
                continue
            wandb.log({f"agent_{agent_id}/{k}": v}, step=self.current_step)
        self.memory.clear()