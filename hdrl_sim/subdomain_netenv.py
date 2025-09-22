import networkx as nx
from itertools import islice
from copy import deepcopy
import numpy as np
from gym.spaces import Box, Dict
from gym import spaces
from xuance.environment import RawEnvironment
import random
class SubdomainNetEnv(RawEnvironment):
    def __init__(self, env_config):
        super(SubdomainNetEnv, self).__init__()
        self.env_id = env_config.env_id  # The environment id.
        
        # graph states 208, flow states 6
        # node features: 16 nodes * 4 features = 64
        # edge features: 24 edges * 6 features = 144
        self.observation_space = Box(-np.inf, np.inf, shape=[214, ]) 
        
        self.action_space = spaces.Discrete(4)  # Define action space. In this example, the action space is continuous.
        self.max_episode_steps = 320000  # The max episode length.
        self.subdomain_graph = env_config.graph # undirected graph
        # 图的结构
        # 0-4-8-12
        # | | | |
        # 1-5-9-13
        # | | | |
        # 2-6-10-14
        # | | | |
        # 3-7-11-15

        self._current_step = 0  # The count of steps of current episode.
        
        self.num_of_orbit = env_config.num_of_orbit
        self.sat_of_orbit = env_config.sat_of_orbit
        # Mesh grid parameters - assuming 4x4 grid structure
        self.grid_rows = 4
        self.grid_cols = 4
        self.num_satellites = self.grid_rows * self.grid_cols
        
        self.flow_src = random.choice(list(self.subdomain_graph.nodes()))
        self.flow_des = random.choice(list(self.subdomain_graph.nodes()))
        self.flow_bd = 0
        self.survival_time = 7
        self.paths = None
        
        # 初始化预计算属性
        self._edge_index = None
        self.adjacency_matrix = None
        
        # 预计算固定的图结构属性
        self._precompute_graph_structure()

    def _precompute_graph_structure(self):
        """
        预计算图的固定结构属性，避免每次调用get_gnn_obs时重复计算
        
        预计算内容:
        - edge_index: 边索引矩阵
        - adjacency_matrix: 邻接矩阵  
        - node_to_idx: 节点到索引的映射
        - sorted_nodes: 排序后的节点列表
        """

        # 获取节点列表并排序
        self.sorted_nodes = list(self.subdomain_graph.nodes())
        self.sorted_nodes.sort(key=lambda x: int(x.split('_')[1]) if isinstance(x, str) and '_' in x else 0)
        
        num_nodes = len(self.sorted_nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.sorted_nodes)}
        
        # 构建边索引和邻接矩阵
        edges = list(self.subdomain_graph.edges())
        num_edges = len(edges)
        
        self._edge_index = np.zeros((2, num_edges), dtype=np.int64)
        self.adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        for i, (node1, node2) in enumerate(edges):
            # 边的节点索引
            node1_idx = self.node_to_idx[node1]
            node2_idx = self.node_to_idx[node2]
            
            self._edge_index[0, i] = node1_idx
            self._edge_index[1, i] = node2_idx
            
            # 更新邻接矩阵 (无向图，对称)
            self.adjacency_matrix[node1_idx, node2_idx] = 1
            self.adjacency_matrix[node2_idx, node1_idx] = 1
        
        print(f"预计算完成: {num_nodes}个节点, {num_edges}条边")

    @property
    def edge_index(self):
        if self._edge_index is not None:
            return deepcopy(self._edge_index)
        else:
            raise ValueError("edge_index尚未预计算，请先调用_precompute_graph_structure()")
    
    def reset(self, **kwargs):  # Reset your environment.
        self._current_step = 0
        return self.get_observation(), {}

    def reset_obs(self):
        return self.get_observation(), {}

    def get_observation(self):
        self.paths = self.get_k_shortest_paths(self.flow_src, self.flow_des)
        if self.paths is not []:
            self.add_path_encoding(self.paths)
        
        graph_obs = self.get_graph_observation()
        flow_obs = self.get_flow_observation()
        obs = np.concatenate([graph_obs, np.array(flow_obs)])
        return obs

    def get_reward(self, path:list):
        """
        优化的奖励函数设计
        - 路径时延：累积所有边的时延
        - 路径利用率：取最大利用率（瓶颈链路）
        - 使用指数函数处理时延和利用率
        - 返回加权和的负值（值越小越好的路径）
        """
        # 无效路径检查
        if path is None or len(path) < 2:
            return -10.0  # 无效路径，给予较大负奖励
        
        # 权重参数
        delay_weight = 0.6  # 时延权重
        utilization_weight = 0.4  # 利用率权重
        
        total_delay = 0.0
        max_utilization = 0.0  # 使用最大利用率而不是求和
        edge_count = 0
        
        for i in range(len(path)-1):
            edge = (path[i], path[i+1])
            edge_data = None
            
            # 查找边（处理无向图）
            if edge in self.subdomain_graph.edges():
                edge_data = self.subdomain_graph.edges[edge]
            elif (path[i+1], path[i]) in self.subdomain_graph.edges():
                edge_data = self.subdomain_graph.edges[(path[i+1], path[i])]
            else:
                print(f"Edge {edge} not found in graph")
                return -10.0  # 边不存在，给予较大负奖励
            
            # 累积时延和更新最大利用率
            total_delay += edge_data.get('delay', 0)
            current_utilization = edge_data.get('utilization', 0)
            max_utilization = max(max_utilization, current_utilization)
            edge_count += 1
        
        if edge_count == 0:
            return -10.0
        
        
        # 使用指数函数处理（值越大，惩罚越重）
        import math
        delay_penalty = math.exp(total_delay * 1000 / 200)  # delay 单位为s，假设时延在0-200ms范围
        utilization_penalty = math.exp(max_utilization * 5)-1  # 利用率在0-1范围，放大5倍
        
        # 计算加权和并取负值
        weighted_sum = delay_weight * delay_penalty + utilization_weight * utilization_penalty
        reward = -weighted_sum
        
        return reward
    
    def step(self, action:int):  # Run a step with an action.
        self._current_step += 1
        
        
        # the mean of the action values is selected path, which is an integer in [0,3]
        # the action is the selected path for the flow
        # the action is an integer in [0,3]
        # the action is the index of the path in the list of paths
        path = self.paths[action]
        self.assigned_flows_to_network(self.flow_bd, path)
        
        observation = self.get_observation()  # 使用统一的观测方法
        # take action. distribute flows based on action
        reward = self.get_reward(path)

        terminated = False
        truncated = False 
        info = {'flow_path': path}
        return observation, reward, terminated, truncated, info

    def render(self, *args, **kwargs):  # Render your environment and return an image if the render_mode is "rgb_array".
        pass

    def close(self):  # Close your environment.
        
        return
    
    def assigned_flows_to_network(self, flow_bd, path):
        
        for i in range(len(path)-1):
            edge = (path[i], path[i+1])
            if edge in self.subdomain_graph.edges():
                self.subdomain_graph.edges[edge]['occupied_bandwidth'] += flow_bd
                self.subdomain_graph.edges[edge]['utilization'] = min(
                    1.0, (self.subdomain_graph.edges[edge]['occupied_bandwidth']) / self.subdomain_graph.edges[edge]['capacity']
                )
                self.subdomain_graph.edges[edge]['available_bandwidth'] = max(
                    0, self.subdomain_graph.edges[edge]['capacity'] - self.subdomain_graph.edges[edge]['occupied_bandwidth']
                )
            elif (path[i+1], path[i]) in self.subdomain_graph.edges():
                self.subdomain_graph.edges[(path[i+1], path[i])]['occupied_bandwidth'] += flow_bd
                self.subdomain_graph.edges[(path[i+1], path[i])]['utilization'] = min(
                    1.0, (self.subdomain_graph.edges[(path[i+1], path[i])]['occupied_bandwidth']) / self.subdomain_graph.edges[(path[i+1], path[i])]['capacity']
                )
                self.subdomain_graph.edges[(path[i+1], path[i])]['available_bandwidth'] = max(
                    0, self.subdomain_graph.edges[(path[i+1], path[i])]['capacity'] - self.subdomain_graph.edges[(path[i+1], path[i])]['occupied_bandwidth']
                )

    def get_k_shortest_paths(self, source, target, k=4, weight='delay'):
        """
        获取子域网络中两节点间的前k条最短路径
        
        关键特性：
        1. 确定性：相同输入总是产生相同输出
        2. 使用Yen's算法确保最优性
        3. 支持多种权重策略
        
        Args:
            source: 源卫星节点
            target: 目标卫星节点
            k: 返回路径数量
            weight: 路径权重属性 ('delay', 'utilization', 'composite', None)
        
        Returns:
            list: 前k条路径及其相关信息，按权重递增排序
        """
        if self.subdomain_graph is None:
            return []
        

        if source not in self.subdomain_graph.nodes():
            print(f"Source node {source} not in subdomain graph")
            return []
        if target not in self.subdomain_graph.nodes():
            print(f"Target node {target} not in subdomain graph")
            return []
        
        try:
            # 根据权重类型选择合适的权重函数
            if weight == 'delay':
                weight_func = 'delay'
            elif weight == 'utilization':
                # 对于利用率，我们希望选择利用率低的路径
                weight_func = lambda u, v, d: d.get('utilization', 0)
            elif weight == 'composite':
                # 综合权重：延迟 + 利用率 * 10
                weight_func = lambda u, v, d: d.get('delay', 0) + d.get('utilization', 0) * 10
            else:
                weight_func = None
            
            # 获取k条最短路径 (使用Yen's算法，结果确定性)
            paths = list(islice(
                nx.shortest_simple_paths(self.subdomain_graph, source, target, weight=weight_func), 
                k
            ))
            
            # # 计算每条路径的详细信息
            # path_info = []
            # for i, path in enumerate(paths):
            #     path_data = self._analyze_path(path, weight)
            #     path_info.append({
            #         'rank': i + 1,  # 路径排名
            #         'path': path,
            #         'length': len(path) - 1,  # 跳数
            #         'total_delay': path_data['total_delay'],
            #         'avg_utilization': path_data['avg_utilization'],
            #         'max_utilization': path_data['max_utilization'],
            #         'path_quality': path_data['path_quality'],
            #         'weight_used': weight  # 记录使用的权重类型
            #     })
            
            return paths
            
        except nx.NetworkXNoPath:
            print(f"No path exists between {source} and {target}")
            return []
        except Exception as e:
            print(f"Error finding k shortest paths: {e}")
            return []

    def add_path_encoding(self, paths):
        """
        获取路径的链路编码表示
        
        Args:
            paths: 一组路径，每个路径是节点的列表，给路径经过的链路进行编码。假设最大就4条路径，采用one-hot编码
            如果链路没有被任意路径经过，则该链路编码为[0,0,0,0]
            如果链路被第0条路径经过，则该链路编码为[1,0,0,0]
            如果链路被第1条路径经过，则该链路编码为[0,1,0,0]
            如果链路被多条路径经过，则该链路编码为对应位置为1，其余为0，例如[1,0,1,0]
        """
        
        # 遍历所有的边，是否边都有属性path_encoding，，如果没有，则初始化为[0,0,0,0], 如果有，都置为[0,0,0,0]
        for edge in self.subdomain_graph.edges():
            if 'path_encoding' not in self.subdomain_graph.edges[edge]:
                self.subdomain_graph.edges[edge]['path_encoding'] = [0, 0, 0, 0]
            else: 
                self.subdomain_graph.edges[edge]['path_encoding'] = [0, 0, 0, 0]
        
        # 遍历每条路径，更新链路编码
        for path_idx, path in enumerate(paths):
            for i in range(len(path) - 1):
                node1 = path[i]
                node2 = path[i + 1]
                edge = (node1, node2) if (node1, node2) in self.subdomain_graph.edges() else (node2, node1)
                if edge in self.subdomain_graph.edges():
                    self.subdomain_graph.edges[edge]['path_encoding'][path_idx] = 1

    def _combine_and_flatten_features(self, node_features, edge_features):
        """
        将节点特征和边特征合并并展平为一维数组
        
        合并策略：
        1. 先将节点特征展平 [num_nodes * node_feature_dim]
        2. 再将边特征展平 [num_edges * edge_feature_dim]  
        3. 合并成一个连续的一维数组
        
        Args:
            node_features: 节点特征矩阵 [num_nodes, node_feature_dim]
            edge_features: 边特征矩阵 [num_edges, edge_feature_dim]
            
        Returns:
            np.ndarray: 展平后的一维特征数组
        """
        # 展平节点特征
        node_flat = node_features.flatten()
        
        # 展平边特征
        edge_flat = edge_features.flatten()
        
        # 合并为一个一维数组
        combined = np.concatenate([node_flat, edge_flat])
        
        return combined
    
    def _recover_features_from_flat(self, combined_features, node_shape=(16, 4), edge_shape=(24, 6)):
        """
        从展平的一维数组中恢复节点特征和边特征
        
        Args:
            combined_features: 展平的一维特征数组
            node_shape: 节点特征的原始形状 (num_nodes, node_feature_dim)
            edge_shape: 边特征的原始形状 (num_edges, edge_feature_dim)
            
        Returns:
            tuple: (node_features, edge_features) 恢复后的特征矩阵
        """
        # 计算各部分的长度
        node_size = np.prod(node_shape)
        edge_size = np.prod(edge_shape)
        
        # 验证输入数组长度
        expected_size = node_size + edge_size
        if len(combined_features) != expected_size:
            raise ValueError(f"Combined features length {len(combined_features)} "
                           f"doesn't match expected size {expected_size}")
        
        # 分割数组
        node_flat = combined_features[:node_size]
        edge_flat = combined_features[node_size:node_size + edge_size]
        
        # 恢复原始形状
        node_features = node_flat.reshape(node_shape)
        edge_features = edge_flat.reshape(edge_shape)
        
        return node_features, edge_features
    
    @staticmethod
    def recover_features_from_observation(obs):
        """
        静态方法：从观测字典中恢复特征
        
        Args:
            obs: 包含combined_features和feature_shapes的观测字典
            
        Returns:
            tuple: (node_features, edge_features)
        """
        if 'combined_features' not in obs or 'feature_shapes' not in obs:
            raise ValueError("Observation must contain 'combined_features' and 'feature_shapes'")
        
        combined = obs['combined_features']
        shapes = obs['feature_shapes']
        
        # 创建临时实例来使用恢复方法
        temp_env = SubdomainNetEnv.__new__(SubdomainNetEnv)
        return temp_env._recover_features_from_flat(
            combined, shapes['node_shape'], shapes['edge_shape']
        )

    def get_graph_observation(self):
        """
        返回适用于GNN网络的观测数据
        
        使用预计算的图结构属性(edge_index, adjacency_matrix等)提高性能
        
        GNN输入格式包括:
        - node_features: 节点特征矩阵 [num_nodes, node_feature_dim]  
        - edge_features: 边特征矩阵 [num_edges, edge_feature_dim]
        - edge_index: 邻接关系矩阵 [2, num_edges] 格式为PyTorch Geometric
        - adjacency_matrix: 传统邻接矩阵 [num_nodes, num_nodes]
        
        节点特征 (每个节点4维特征):
        - 轨道ID (归一化)
        - 轨道内位置 (归一化) 
        - 平均边利用率
        - 平均边延迟
        
        边特征 (每个边6维特征):
        - 利用率
        - 延迟  
        - 路径编码 (4维one-hot)
        
        Returns:
            dict: 包含GNN所需各个组件的字典
        """
        
        # 使用预计算的图结构属性
        num_nodes = len(self.sorted_nodes)
        
        # 1. 构建节点特征矩阵 [num_nodes, 4]
        node_features = np.zeros((num_nodes, 4))
        
        for i, node in enumerate(self.sorted_nodes):
            # 提取卫星ID
            sat_id = int(node.split('_')[1]) if isinstance(node, str) and '_' in node else 0
            
            # 轨道ID和位置 (基于Walker Delta排列)
            orbit_id = sat_id // self.sat_of_orbit  # 轨道ID
            position_in_orbit = sat_id % self.sat_of_orbit  # 轨道内位置
            
            # 归一化轨道信息
            node_features[i, 0] = orbit_id / (self.num_of_orbit - 1)  # 归一化轨道ID
            node_features[i, 1] = position_in_orbit / (self.sat_of_orbit - 1)  # 归一化位置
            
            
            # 计算该节点相邻边的平均利用率和延迟
            neighbor_edges = list(self.subdomain_graph.edges(node, data=True))
            if neighbor_edges:
                utilizations = [edge_data.get('utilization', 0) for _, _, edge_data in neighbor_edges]
                delays = [edge_data.get('delay', 0) for _, _, edge_data in neighbor_edges]
                
                node_features[i, 2] = np.mean(utilizations)
                node_features[i, 3] = np.mean(delays) / 100.0  # 假设延迟归一化除以100
            else:
                node_features[i, 2] = 1.0  
                node_features[i, 3] = 1.0  
            
        
        # 2. 构建边特征 (使用预计算的edge_index和adjacency_matrix)
        edges = list(self.subdomain_graph.edges(data=True))
        num_edges = len(edges)
        
        edge_features = np.zeros((num_edges, 6))
        
        for i, (node1, node2, edge_data) in enumerate(edges):
            # 边特征
            edge_features[i, 0] = edge_data.get('utilization', 0.0)
            edge_features[i, 1] = edge_data.get('delay', 0.0) / 100.0  # 归一化延迟
            
            # 路径编码 (如果存在)
            path_encoding = edge_data.get('path_encoding', [0, 0, 0, 0])

            edge_features[i, 2] = path_encoding[0]
            edge_features[i, 3] = path_encoding[1]
            edge_features[i, 4] = path_encoding[2]
            edge_features[i, 5] = path_encoding[3]
            
            
        # # 3. 添加反向边以确保无向图的完整表示
        # if num_edges > 0:
        #     # 为无向图添加反向边
        #     reverse_edge_index = np.zeros((2, num_edges), dtype=np.int64)
        #     reverse_edge_index[0] = edge_index[1]  
        #     reverse_edge_index[1] = edge_index[0]
            
        #     # 合并正向和反向边
        #     complete_edge_index = np.concatenate([edge_index, reverse_edge_index], axis=1)
        #     complete_edge_features = np.concatenate([edge_features, edge_features], axis=0)
        # else:
        #     # 处理无边的情况
        #     complete_edge_index = np.zeros((2, 0), dtype=np.int64)
        #     complete_edge_features = np.zeros((0, 6), dtype=np.float32)

        
        # 确保所有特征值在合理范围内
        node_features = np.clip(node_features, 0.0, 1.0)
        edge_features = np.clip(edge_features, 0.0, 1.0)
        
        # 将node_features和edge_features合并并展平为一维数组 
        # node_shape=(16, 4) + edge_shape=(24, 6) = 16*4 + 24*6 = 64 + 144 = 208
        combined_features = self._combine_and_flatten_features(node_features, edge_features)
        
        # return {
        #     'node_features': node_features.astype(np.float32),
        #     'edge_features': edge_features.astype(np.float32),
        #     'edge_index': self.edge_index,  # 使用预计算的edge_index
        #     'adjacency_matrix': self.adjacency_matrix,  # 使用预计算的adjacency_matrix
        #     'num_nodes': num_nodes,
        #     'num_edges': edge_features.shape[0],
        #     'combined_features': combined_features.astype(np.float32),
        #     'feature_shapes': {
        #         'node_shape': node_features.shape,
        #         'edge_shape': edge_features.shape
        #     }
        # }
        return combined_features.astype(np.float32)

    def get_flow_observation(self):
        """
        获取流相关的观测特征
        
        Returns:
            list: 6维流状态特征
            - 源节点轨道ID (归一化)
            - 源节点轨道内位置 (归一化)
            - 目标节点轨道ID (归一化)
            - 目标节点轨道内位置 (归一化)
            - 归一化带宽需求
            - 归一化生存时间
        """
        # 从flow_src节点名中提取ID（假设格式为 "sat_X"）
        src_id = int(self.flow_src.split('_')[1]) if isinstance(self.flow_src, str) and '_' in self.flow_src else 0
        des_id = int(self.flow_des.split('_')[1]) if isinstance(self.flow_des, str) and '_' in self.flow_des else 0
        
        # 计算轨道ID和轨道内位置
        num_of_orbit_src = (src_id // self.sat_of_orbit) / (self.num_of_orbit - 1)
        sat_of_orbit_src = (src_id % self.sat_of_orbit) / (self.sat_of_orbit - 1)
        
        num_of_orbit_des = (des_id // self.sat_of_orbit) / (self.num_of_orbit - 1)
        sat_of_orbit_des = (des_id % self.sat_of_orbit) / (self.sat_of_orbit - 1)
        
        # 归一化带宽需求 (假设范围为 10-40 Mbps)
        normal_bd = max(0.0, min(1.0, (self.flow_bd - 10) / 30)) if hasattr(self, 'flow_bd') else 0.0
        
        # 归一化生存时间 (假设范围为 7-10 秒)
        st = max(0.0, min(1.0, (self.survival_time - 7) / 3)) if hasattr(self, 'survival_time') else 0.0
        
        flow_states = [num_of_orbit_src, sat_of_orbit_src, num_of_orbit_des, sat_of_orbit_des, normal_bd, st]
        return flow_states

