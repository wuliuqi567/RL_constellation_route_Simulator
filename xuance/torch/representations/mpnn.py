from xuance.common import Sequence, Optional, Union, Callable
from xuance.torch import Module, Tensor
from xuance.torch.utils import torch, nn, cnn_block, mlp_block, ModuleType
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import MessagePassing
import numpy as np


class MPNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # "add" aggregation (Step 5).
        super(MPNNConv, self).__init__(aggr='add')
        
        self.lin_message = nn.Linear(in_channels, out_channels)
        # 更新函数
        self.lin_update = nn.Linear(in_channels + out_channels, out_channels)
        
    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        
        # 开始消息传递
        out = self.propagate(edge_index, x=x)
        
        # 更新节点特征
        out = self.lin_update(torch.cat([x, out], dim=1))
        
        return out
        
    def message(self, x_j):
        # x_j has shape [E, in_channels]
        # 第2步：计算消息
        return self.lin_message(x_j)


class MultiRound_MPNN(Module):
    """
    多轮消息传递神经网络
    实现真正的多轮消息传递机制，每轮使用相同的消息传递层
    """
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: Sequence[int],
                 heads: int = 1,
                 dropout: float = 0.0,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 n_message_passing_rounds: int = 3,  # 消息传递轮数
                 residual_connections: bool = True,  # 是否使用残差连接
                 **kwargs):
        super(MultiRound_MPNN, self).__init__()
        
        self.input_shape = input_shape
        self.hidden_dim = hidden_sizes[-1]
        self.output_dim = hidden_sizes[-1]
        self.heads = heads
        self.dropout = dropout 
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device
        self.n_message_passing_rounds = n_message_passing_rounds
        self.residual_connections = residual_connections
        self.output_shapes = {'state': (hidden_sizes[-1],)}

        # 输入投影层：将输入特征维度转换为隐藏维度
        self.input_projection = nn.Linear(13, self.hidden_dim)
        
        # 消息传递层（所有轮次共享）
        self.message_passing_layer = MPNNConv(
            in_channels=self.hidden_dim, 
            out_channels=self.hidden_dim
        )
        
        # 激活函数
        if activation is None:
            self.act = nn.ReLU()
        elif isinstance(activation, type):
            self.act = activation()
        else:
            self.act = activation
        
        # Dropout层
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 层归一化（可选）
        if normalize is not None:
            self.layer_norm = normalize(self.hidden_dim)
        else:
            self.layer_norm = nn.Identity()
        
        self.to(self.device)
        
    def forward(self, x: Optional[Tensor], edge_index: Optional[Tensor]):
        """
        前向传播
        Args:
            x: 节点特征 [N, input_dim]
            edge_index: 边索引 [2, E]
        Returns:
            dict: {'state': output_tensor}
        """
        # 输入投影
        h = self.input_projection(x)
        h = self.act(h)
        h = self.dropout_layer(h)
        
        # 多轮消息传递
        for round_idx in range(self.n_message_passing_rounds):

            # 消息传递
            h = self.message_passing_layer(h, edge_index)
            
            # 激活函数（仅在非最后一轮使用）
            if round_idx < self.n_message_passing_rounds - 1:
                # 层归一化
                h = self.layer_norm(h)
                h = self.act(h)
                # Dropout
                h = self.dropout_layer(h)
                
        # 输出投影
        output = h
        
        return {'state': output}


class Basic_MPNN(Module):
    """
    保持原有的Basic_MPNN类以兼容现有代码
    """
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: Sequence[int],
                 heads: int = 1,
                 dropout: float = 0.0,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 n_message_passing_rounds: int = 3,
                 **kwargs):
        super(Basic_MPNN, self).__init__()
        self.input_shape = input_shape
        self.hidden_dim = hidden_sizes[-2] if len(hidden_sizes) > 1 else hidden_sizes[0]
        self.output_dim = hidden_sizes[-1]
        self.heads = heads
        self.dropout = dropout 
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device
        self.n_message_passing_rounds = n_message_passing_rounds
        self.output_shapes = {'state': (hidden_sizes[-1],)}

        # 三层MPNN结构
        self.mpnn1 = MPNNConv(in_channels=13, out_channels=self.hidden_dim)
        self.mpnn2 = MPNNConv(in_channels=self.hidden_dim, out_channels=self.output_dim)
        # self.mpnn3 = MPNNConv(in_channels=self.hidden_dim, out_channels=self.output_dim)
        
        # 激活函数
        if activation is None:
            self.act = nn.ReLU()
        elif isinstance(activation, type):
            self.act = activation()
        else:
            self.act = activation
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.to(self.device)
        
    def forward(self, x: Optional[Tensor], edge_index: Optional[Tensor]):
        x = self.mpnn1(x, edge_index)
        x = self.act(x)
        x = self.dropout_layer(x)
        
        output = self.mpnn2(x, edge_index)
        # x = self.act(x) 
        # x = self.dropout_layer(x)
        
        # output = self.mpnn3(x, edge_index)
        
        return {'state': output}


class SimplifiedMultiRoundMPNN(Module):
    """
    简化版多轮消息传递神经网络
    减少不必要的激活和归一化层
    """
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: Sequence[int],
                 heads: int = 1,
                 dropout: float = 0.0,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 n_message_passing_rounds: int = 3,
                 residual_connections: bool = False,  # 默认关闭残差连接
                 use_layer_norm: bool = False,  # 可选择是否使用层归一化
                 **kwargs):
        super(SimplifiedMultiRoundMPNN, self).__init__()
        
        self.input_shape = input_shape
        self.hidden_dim = hidden_sizes[-2] if len(hidden_sizes) > 1 else hidden_sizes[0]
        self.output_dim = hidden_sizes[-1]
        self.device = device
        self.n_message_passing_rounds = n_message_passing_rounds
        self.residual_connections = residual_connections
        self.use_layer_norm = use_layer_norm
        self.output_shapes = {'state': (hidden_sizes[-1],)}

        # 输入投影层
        self.input_projection = nn.Linear(13, self.hidden_dim)
        
        # 消息传递层（所有轮次共享）
        self.message_passing_layer = MPNNConv(
            in_channels=self.hidden_dim, 
            out_channels=self.hidden_dim
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(self.hidden_dim, self.output_dim)
        
        # 激活函数（只在输入投影后使用）
        if activation is None:
            self.act = nn.ReLU()
        elif isinstance(activation, type):
            self.act = activation()
        else:
            self.act = activation
        
        # 可选的dropout（只在必要时使用）
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 可选的层归一化
        if self.use_layer_norm and normalize is not None:
            self.layer_norm = normalize(self.hidden_dim)
        else:
            self.layer_norm = nn.Identity()
        
        self.to(self.device)
        
    def forward(self, x: Optional[Tensor], edge_index: Optional[Tensor]):
        """
        简化的前向传播
        只在关键位置使用激活和归一化
        """
        # 输入投影 + 激活（保留，用于非线性变换）
        h = self.input_projection(x)
        h = self.act(h)
        
        # 多轮消息传递（简化版）
        for round_idx in range(self.n_message_passing_rounds):
            h_prev = h if self.residual_connections else None
            
            # 消息传递
            h = self.message_passing_layer(h, edge_index)
            
            # 残差连接（可选）
            if self.residual_connections and h_prev is not None:
                h = h + h_prev
            
            # 层归一化（可选，仅在最后一轮或使用残差时）
            if self.use_layer_norm and (round_idx == self.n_message_passing_rounds - 1 or self.residual_connections):
                h = self.layer_norm(h)
            
            # 激活函数（仅在非最后一轮使用）
            if round_idx < self.n_message_passing_rounds - 1:
                h = self.act(h)
        
        # 最终dropout（可选）
        h = self.dropout_layer(h)
        
        # 输出投影（不使用激活）
        output = self.output_projection(h)
        
        return {'state': output}

