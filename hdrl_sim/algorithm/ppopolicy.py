import os
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from gym.spaces import Discrete
from xuance.common import Sequence, Optional, Callable, Union
from xuance.torch import Module, Tensor, DistributedDataParallel
from xuance.torch.utils import ModuleType
from xuance.torch.policies.core import CategoricalActorNet as ActorNet
from xuance.torch.policies.core import CategoricalActorNet_SAC as Actor_SAC
from xuance.torch.policies.core import BasicQhead, CriticNet
from xuance.torch import REGISTRY_Representation, REGISTRY_Learners, Module
from xuance.torch.utils import nn, NormalizeFunctions, ActivationFunctions, init_distributed_mode
from torch_geometric.utils import to_undirected


class GNNActorCriticPolicy(Module):
    """
    Actor-Critic for stochastic policy with categorical distributions. (Discrete action space)

    Args:
        action_space (Discrete): The discrete action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(GNNActorCriticPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = representation.output_shapes
        self.device = device
        self.res_output_dim = self.representation_info_shape['state'][0]
        self.actor = ActorNet(self.res_output_dim*16+6, self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, device)
        self.critic = CriticNet(self.res_output_dim*16+6, critic_hidden_size,
                                normalize, initialize, activation, device)

        # Prepare DDP module.
        self.distributed_training = use_distributed_training
        if self.distributed_training:
            self.rank = int(os.environ["RANK"])
            if self.representation._get_name() != "Basic_Identical":
                self.representation = DistributedDataParallel(module=self.representation, device_ids=[self.rank])
            self.actor = DistributedDataParallel(module=self.actor, device_ids=[self.rank])
            self.critic = DistributedDataParallel(module=self.critic, device_ids=[self.rank])
            
        for key, value in kwargs.items():
            if 'edge_index' in key:
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value).to(self.device)
                setattr(self, key, value)
        
        

    def forward(self, observation: Union[np.ndarray, Tensor]):
        """
        Returns the hidden states, action distribution, and values.

        Parameters:
            observation: The original observation of agent.

        Returns:
            outputs: The outputs of representation.
            a_dist: The distribution of actions output by actor.
            value: The state values output by critic.
        """
        bs = observation.shape[0]
        
        node_data, edge_index_cat, edge_attr, flow_features = self._recover_features_from_flat(observation)
        
        outputs = self.representation(node_data, edge_index_cat, edge_attr)
        
        # Reshape outputs from (bs*16, hidden_dim) to (bs, hidden_dim*16)
        state_features = outputs['state']  # Shape: (bs*16, hidden_dim)
        reshaped_features = state_features.reshape(bs, 16, self.res_output_dim)  # 16个节点，Shape: (bs, 16, hidden_dim)
        
        concatenated_features = reshaped_features.reshape(bs, -1)  # Shape: (bs, hidden_dim*16)
        
        # Concatenate flow_features if available
        if flow_features is not None:
            outputs['state'] = torch.cat([concatenated_features, flow_features], dim=1)
        else:
            outputs['state'] = concatenated_features
        
        # Continue with actor and critic
        a_dist = self.actor(outputs['state'])
        value = self.critic(outputs['state'])
        return outputs, a_dist, value[:, 0]
    
    
    def _recover_features_from_flat(self, observation, node_shape=(16, 4), edge_shape=(24, 6), flow_dim=6):
        """
        从展平的一维数组中恢复节点特征、边特征和流量信息
        
        Args:
            observation: 展平的观察数组，形状为 (batch_size, feature_dim)
                        feature_dim = 16*4 + 24*6 + 6 = 64 + 144 + 6 = 214
            node_shape: 节点特征的原始形状 (num_nodes, node_feature_dim)
            edge_shape: 边特征的原始形状 (num_edges, edge_feature_dim)
            flow_dim: 流量特征的维度
            
        Returns:
            tuple: (node_data, edge_index, edge_attr, flow_features) 
                   - node_data: 节点特征 (batch_size * num_nodes, node_feature_dim)
                   - edge_index: 边索引，使用self.edge_index
                   - edge_attr: 边特征 (batch_size * num_edges, edge_feature_dim)
                   - flow_features: 流量特征 (batch_size, flow_dim)
        """
        # 转换为tensor
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).to(self.device)
        
        batch_size = observation.shape[0]
        
        # 计算各部分的长度
        node_size = np.prod(node_shape)  # 16 * 4 = 64
        edge_size = np.prod(edge_shape)  # 24 * 6 = 144
        
        # 验证输入数组长度
        expected_size = node_size + edge_size + flow_dim  # 64 + 144 + 6 = 214
        if observation.shape[1] != expected_size:
            raise ValueError(f"Combined features length {observation.shape[1]} "
                           f"doesn't match expected size {expected_size}")
        
        # 分割数组
        node_flat = observation[:, :node_size]  # (batch_size, 64)
        edge_flat = observation[:, node_size:node_size + edge_size]  # (batch_size, 144)
        flow_features = observation[:, node_size + edge_size:]  # (batch_size, 6)
        
        # 合成一张大图，从（bs, 16, 4) 变为 （bs*16, 4)
        node_features = node_flat.reshape(batch_size, node_shape[0], node_shape[1])  # (batch_size, 16, 4)
        node_data = node_features.reshape(-1, node_shape[1])  # (batch_size * 16, 4)
        
        # 合成边特征，从（bs, 24, 6) 变为 （bs*24, 6)
        edge_features = edge_flat.reshape(batch_size, edge_shape[0], edge_shape[1])  # (batch_size, 24, 6)
        edge_attr = edge_features.reshape(-1, edge_shape[1])  # (batch_size * 24, 6)
        
        # 为每个batch创建边索引，构建大图
        edge_index_batch = []
        num_nodes = node_shape[0]  # 16
        for i in range(batch_size):
            # 为每个batch的边索引添加节点偏移
            batch_edge_index = self.edge_index + i * num_nodes  
            edge_index_batch.append(batch_edge_index)
        edge_index_cat = torch.cat(edge_index_batch, dim=1)  # (2, batch_size * num_edges)
        
        # 确保数据类型正确
        node_data = node_data.float()
        edge_attr = edge_attr.float()
        flow_features = flow_features.float()
        
        return node_data, edge_index_cat, edge_attr, flow_features
        
        
        
