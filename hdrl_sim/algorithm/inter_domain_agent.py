
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
from hdrl_sim.algorithm.inter_domain_learner import InterDomain_Learner
# from agent import Agent
from xuance.common import Optional, Union, DummyOnPolicyBuffer
import swanlab as wandb


class InterDomainAgent(OnPolicyAgent):
    """The implementation of PPO agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecEnv, SubprocVecEnv]):
        super(InterDomainAgent, self).__init__(config, envs)
        self.auxiliary_info_shape = {"old_logp": ()}
        self.edge_index = self.envs.get_env_attributes()
        self.memory = self._build_memory(self.auxiliary_info_shape)  # build memory
        self.policy = self._build_policy()  # build policy
        REGISTRY_Learners["interdomainlearner"] = InterDomain_Learner
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
        first_flow = flows[0]
        flow_attribute = {
            "flow_src": first_flow.source,
            "flow_des": first_flow.destination,
            'flow_bd': first_flow.flow_rate
        }
        self.envs.set_env_attributes(flow_attribute)
        self.envs.reset_obs()
        obs = self.envs.buf_obs
        num_flows = len(flows)
        
        if self.memory.full:
            vals = self.get_terminated_values(obs)
            for i in range(self.n_envs):
                    self.memory.finish_path(vals[i], i)
            train_info = self.train_epochs(n_epochs=self.n_epochs)
            # self.log_infos(train_info, self.current_step)
            self.memory.clear()
        
        
        for idx, flow in enumerate(flows):
            # step_info = {}
            
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            policy_out = self.action(obs, return_dists=False, return_logpi=True)
            acts, value, logps = policy_out['actions'], policy_out['values'], policy_out['log_pi']
            next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)
            flow_path = infos[0].get('flow_path', [])
            flow.path = flow_path
            aux_info = self.get_aux_info(policy_out)
            self.memory.store(obs, acts, self._process_reward(rewards), value, terminals, aux_info)
            if (idx + 1) < num_flows:
                flow_attribute = {
                    "flow_src": flows[idx+1].source,
                    "flow_des": flows[idx+1].destination,
                    'flow_bd' : flows[idx+1].flow_rate
                }
                self.envs.set_env_attributes(flow_attribute)
                self.envs.reset_obs()
                obs = self.envs.buf_obs
            
            
                if self.memory.full:
                    vals = self.get_terminated_values(obs)
                    for i in range(self.n_envs):
                        if terminals[i]:
                            self.memory.finish_path(0.0, i)
                        else:
                            self.memory.finish_path(vals[i], i)
                    train_info = self.train_epochs(n_epochs=self.n_epochs)
                    # self.log_infos(train_info, self.current_step)
                    for k, v in train_info.items():
                        if v is None:
                            continue
                        wandb.log({f"agent_{agent_id}/{k}": v}, step=self.current_step)
                    self.memory.clear()

            self.returns = self.gamma * self.returns + rewards

            # for i in range(self.n_envs):
            #     if terminals[i] or trunctions[i]:
            #         self.ret_rms.update(self.returns[i:i + 1])
            #         self.returns[i] = 0.0
            #         if self.atari and (~trunctions[i]):
            #             pass
            #         else:
            #             if terminals[i]:
            #                 self.memory.finish_path(0.0, i)
            #             else:
            #                 vals = self.get_terminated_values(next_obs)
            #                 self.memory.finish_path(vals[i], i)
            #             obs[i] = infos[i]["reset_obs"]
            #             self.envs.buf_obs[i] = obs[i]
            #             self.current_episode[i] += 1
            #             step_info = {}
                        # if self.use_wandb:
            #                 step_info["Episode-Steps/env-%d" % i] = infos[i]["episode_step"]
            #                 step_info["Train-Episode-Rewards/env-%d" % i] = infos[i]["episode_score"]
            #             else:
            #                 step_info["Episode-Steps"] = {"env-%d" % i: infos[i]["episode_step"]}
            #                 step_info["Train-Episode-Rewards"] = {"env-%d" % i: infos[i]["episode_score"]}
            #             self.log_infos(step_info, self.current_step)
            self.current_step += self.n_envs

