import numpy as np
import networkx as nx
import h5py


def satellite_connection_graph(constellation_name, sh, t):
    file_path = "data/XML_constellation/" + constellation_name + ".h5"  # h5 file path and name
    # read the delay matrix of the shell layer of the constellation constellation at time t
    with h5py.File(file_path, 'r') as file:
        # access the existing first-level subgroup delay group
        delay_group = file['delay']
        # access the existing secondary subgroup 'shell'+str(count) subgroup
        current_shell_group = delay_group[sh.shell_name]
        # read the data set
        delay = np.array(current_shell_group['timeslot' + str(t)]).tolist()

    # create an graph object named G, which is empty at first, with no nodes and edges.
    G = nx.Graph()
    satellite_nodes = []
    for i in range(1, len(delay), 1):
        satellite_nodes.append("satellite_" + str(i))
    G.add_nodes_from(satellite_nodes)  # add nodes to graph

     # 添加边及多个属性
    for i in range(1, len(delay), 1):
        for j in range(i + 1, len(delay), 1):
            if delay[i][j] > 0:
                # 计算其他属性
                distance = delay[i][j] * 299792458  # 假设延迟转换为距离
                capacity = 2000  # 默认容量，单位为Mbps
                link_type = determine_link_type(i, j, sh)  # 自定义函数确定链路类型
                
                # 添加边及其所有属性
                G.add_edge(
                    "satellite_" + str(i), 
                    "satellite_" + str(j),
                    weight=delay[i][j],
                    delay=delay[i][j],
                    distance=distance,
                    capacity=capacity,
                    bandwidth=capacity,
                    link_type=link_type,
                    utilization=0.0,  # 初始利用率
                    available_bandwidth=capacity
                )

    return G
    
def determine_link_type(sat1, sat2, shell):
    """确定链路类型：轨道内或轨道间"""
    sats_per_orbit = shell.number_of_satellite_per_orbit
    orbit1 = (sat1 - 1) // sats_per_orbit
    orbit2 = (sat2 - 1) // sats_per_orbit
    
    if orbit1 == orbit2:
        return "intra_orbit"  # 轨道内链路
    else:
        return "inter_orbit"  # 轨道间链路
