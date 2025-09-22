from xuance.common import Sequence, Optional, Union, Callable
from xuance.torch import Module, Tensor
from xuance.torch.utils import torch, nn, cnn_block, mlp_block, ModuleType
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import GAT
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_undirected
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


class EdgeGNNLayer(MessagePassing):
    def __init__(self, node_in_feat, edge_in_feat, out_feat):
        # aggr='add' 表示使用相加的方式聚合邻居消息
        super().__init__(aggr='add')
        
        
        # 定义一个神经网络，用于将 (源节点特征 + 边特征) 映射为消息
        # 这里的网络结构可以任意设计，MLP 是最常见的选择
        self.message_net = nn.Sequential(
            nn.Linear(node_in_feat + edge_in_feat, out_feat),
            nn.BatchNorm1d(out_feat),
            nn.Dropout(p=0.5),
            nn.LeakyReLU()
        )
        
        # 用于将 (聚合后的消息 + 中心节点自身特征) 映射为最终输出
        self.update_net = nn.Sequential(
            nn.Linear(node_in_feat + out_feat, out_feat),
            nn.LeakyReLU()
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        # Step 3: 调用 propagate 开始消息传递
        # 任何在 propagate 中除了 edge_index 之外的参数 (如这里的 x 和 edge_attr)，
        # 都会被传递给 message() 和 update() 方法 (如果它们的签名中有对应参数)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    # --- Step 2: 定义消息函数 message() ---
    # PyG 会自动识别参数名。x_j 代表源节点 j 的特征。
    # 因为我们加入了 edge_attr 参数，PyG 会把边的属性也传进来。
    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        # x_j 的形状是 [num_edges, node_in_feat]
        # edge_attr 的形状是 [num_edges, edge_in_feat]
        
        # 1. 拼接源节点特征和边属性
        input_for_message_net = torch.cat([x_j, edge_attr], dim=-1)
        message = self.message_net(input_for_message_net)

        return message
    
    def update(self, aggr_out, x):
        # aggr_out 的形状是 [num_nodes, out_feat]
        # x 的形状是 [num_nodes, node_in_feat]
        
        # 拼接聚合消息和中心节点特征
        update_input = torch.cat([x, aggr_out], dim=-1)
        
        return self.update_net(update_input)

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

class MultiRound_MPNN_EDGE(Module):
    """
    多轮消息传递神经网络
    实现真正的多轮消息传递机制，每轮使用相同的消息传递层
    """
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: Sequence[int],
                 node_feature_dim: int = 4,
                 edge_feature_dim: int = 6,
                 node_hidden_feature_dim: int = 8,
                 heads: int = 1,
                 dropout: float = 0.0,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 n_message_passing_rounds: int = 3,  # 消息传递轮数
                 residual_connections: bool = True,  # 是否使用残差连接
                 **kwargs):
        super(MultiRound_MPNN_EDGE, self).__init__()
        
        self.input_shape = input_shape
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = node_hidden_feature_dim
        self.output_dim = node_hidden_feature_dim
        self.heads = heads
        self.dropout = dropout 
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device
        self.n_message_passing_rounds = n_message_passing_rounds
        self.residual_connections = residual_connections
        self.output_shapes = {'state': (node_hidden_feature_dim,)}
        
        for key, value in kwargs.items():
            if key == "edge_index":
                
                setattr(self, key, value)

        # 输入投影层：将输入特征维度转换为隐藏维度
        self.input_projection = nn.Linear(self.node_feature_dim, self.hidden_dim)
        
        # 消息传递层（所有轮次共享）
        self.message_passing_layer = EdgeGNNLayer(
            node_in_feat=self.hidden_dim, 
            edge_in_feat=self.edge_feature_dim,
            out_feat=self.hidden_dim
        )
        
        # 输出投影层：将隐藏维度转换为最终输出维度
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        
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
        
    def forward(self, x: Optional[Tensor], edge_index: Optional[Tensor], edge_attr: Optional[Tensor]):
        """
        前向传播
        Args:
            x: 节点特征 [N, input_dim]
            edge_index: 边索引 [2, E]
            edge_attr: 边特征 [E, edge_dim]
        """
        # 输入投影
        new_edge_index, new_edge_attr = to_undirected(edge_index, edge_attr, reduce="mean")
        h = self.input_projection(x)

        # 多轮消息传递
        for _ in range(self.n_message_passing_rounds):

            # 消息传递
            y = self.message_passing_layer(h, new_edge_index, new_edge_attr)

            # 残差连接
            if self.residual_connections:
                h = y + h  # 残差连接
                
        # 输出投影
        output = self.output_projection(h)
        
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
        
        for key, value in kwargs.items():
            if key == "edge_index":
                
                setattr(self, key, value)

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

class Basic_MPNN_EDGE(Module):
    
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
        super(Basic_MPNN_EDGE, self).__init__()
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
        
        for key, value in kwargs.items():
            if key == "edge_index":
                
                setattr(self, key, value)

        # 三层MPNN结构
        self.mpnn1 = EdgeGNNLayer(node_in_feat=13, edge_in_feat=6, out_feat=self.hidden_dim)
        self.mpnn2 = EdgeGNNLayer(node_in_feat=self.hidden_dim, edge_in_feat=6, out_feat=self.output_dim)
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
        
    def forward(self, x: Optional[Tensor], edge_index: Optional[Tensor], edge_attr: Optional[Tensor]):
        x = self.mpnn1(x, edge_index, edge_attr)
        x = self.act(x)
        x = self.dropout_layer(x)
        
        output = self.mpnn2(x, edge_index, edge_attr)
        # x = self.act(x) 
        # x = self.dropout_layer(x)
        
        # output = self.mpnn3(x, edge_index)
        
        return {'state': output}
    
