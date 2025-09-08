from xuance.common import Sequence, Optional, Union, Callable
from xuance.torch import Module, Tensor
from xuance.torch.utils import torch, nn, cnn_block, mlp_block, ModuleType
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import MessagePassing
import numpy as np

class Basic_GAT(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: Sequence[int],
                 heads: int = 1,
                 dropout: float = 0.0,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(Basic_GAT, self).__init__()
        
        self.input_shape = input_shape
        self.hidden_dim = hidden_sizes[-1]
        self.output_dim = hidden_sizes[-1]
        self.heads = heads
        self.dropout = dropout 
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device
        self.output_shapes = {'state': (hidden_sizes[-1],)}

        # 输入投影层：将输入特征维度转换为隐藏维度
        self.input_projection = nn.Linear(13, self.hidden_dim)
        
        # GAT层
        self.conv1 = GATv2Conv(
            in_channels=self.hidden_dim, 
            out_channels=self.output_dim, 
            heads=heads,
            dropout=dropout,
            concat=False  # 不拼接多头输出，而是平均
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
        
        # 层归一化
        h = self.layer_norm(h)
        
        # GAT层
        output = self.conv1(h, edge_index)
        
        return {'state': output}

