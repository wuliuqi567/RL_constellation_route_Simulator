import numpy as np
import random
from typing import Union


def generate_service_requests(arrival_rate: float, time_interval: float = 1.0, 
                            random_seed: Union[int, None] = None) -> int:
    """
    基于泊松分布生成当前时间间隔内的服务请求数量
    
    Args:
        arrival_rate (float): 到达强度（λ），表示单位时间内平均到达的请求数
        time_interval (float): 时间间隔，默认为1.0个时间单位
        random_seed (int, optional): 随机种子，用于结果复现
        
    Returns:
        int: 当前时间间隔内的服务请求数量
        
    Raises:
        ValueError: 当到达强度小于0或时间间隔小于等于0时抛出异常
    """
    if arrival_rate < 0:
        raise ValueError("到达强度必须非负")
    if time_interval <= 0:
        raise ValueError("时间间隔必须大于0")
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 计算期望值：λ * t
    expected_requests = arrival_rate * time_interval
    
    # 使用泊松分布生成请求数量
    request_count = np.random.poisson(expected_requests)
    
    return int(request_count)


def generate_service_requests_batch(arrival_rate: float, time_interval: float = 1.0,
                                  batch_size: int = 1, random_seed: Union[int, None] = None) -> list:
    """
    批量生成多个时间间隔的服务请求数量
    
    Args:
        arrival_rate (float): 到达强度（λ）
        time_interval (float): 时间间隔，默认为1.0个时间单位
        batch_size (int): 批量大小，生成多少个时间间隔的数据
        random_seed (int, optional): 随机种子
        
    Returns:
        list: 包含各个时间间隔服务请求数量的列表
    """
    if arrival_rate < 0:
        raise ValueError("到达强度必须非负")
    if time_interval <= 0:
        raise ValueError("时间间隔必须大于0")
    if batch_size <= 0:
        raise ValueError("批量大小必须大于0")
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    expected_requests = arrival_rate * time_interval
    request_counts = np.random.poisson(expected_requests, batch_size)
    
    return request_counts.tolist()

def generate_service_tarffic_bandwidth_demand(batch_service_equests:list, traffic_demand_low = 10, traffic_demand_high = 40):
    """
    为一批服务生成流量带宽需求
    Args:
        batch_service (list): 服务列表
        traffic_demand_low (int): 流量需求下限, Mbps
        traffic_demand_high (int): 流量需求上限, Mbps
        
    Returns:
        list: 包含每个时间中每个服务的流量带宽需求的列表
    """
    if traffic_demand_low < 0 or traffic_demand_high < 0:
        raise ValueError("流量需求上下限必须非负")
    if traffic_demand_low >= traffic_demand_high:
        raise ValueError("流量需求下限必须小于上限")
    
    traffic_demand_for_each_service = [[random.randint(traffic_demand_low, traffic_demand_high) for _ in range(each_time_interval)] 
                                       for each_time_interval in batch_service_equests]
    
    return traffic_demand_for_each_service
    


def calculate_arrival_statistics(arrival_rate: float, time_interval: float = 1.0) -> dict:
    """
    计算给定到达强度下的统计信息
    
    Args:
        arrival_rate (float): 到达强度（λ）
        time_interval (float): 时间间隔
        
    Returns:
        dict: 包含期望值、方差、标准差的统计信息
    """
    if arrival_rate < 0:
        raise ValueError("到达强度必须非负")
    if time_interval <= 0:
        raise ValueError("时间间隔必须大于0")
    
    expected_value = arrival_rate * time_interval
    variance = expected_value  # 泊松分布的方差等于期望值
    std_deviation = np.sqrt(variance)
    
    return {
        "expected_value": expected_value,
        "variance": variance,
        "std_deviation": std_deviation,
        "arrival_rate": arrival_rate,
        "time_interval": time_interval
    }


# 示例使用函数
def demo_poisson_requests():
    """
    演示泊松分布服务请求生成的示例
    """
    print("=== 泊松分布服务请求生成演示 ===")
    
    # 设置参数
    arrival_rate = 5.0  # 平均每个时间单位到达5个请求
    time_interval = 1.0
    
    for i in range(5):
        requests = generate_service_requests(arrival_rate, time_interval)
        print(f"随机种子={i}, 生成的服务请求数量: {requests}")
    # 生成单个时间间隔的请求数量
    requests = generate_service_requests(arrival_rate, time_interval, random_seed=42)
    print(f"到达强度λ={arrival_rate}, 时间间隔={time_interval}")
    print(f"生成的服务请求数量: {requests}")
    
    # 批量生成
    batch_requests = generate_service_requests_batch(arrival_rate, time_interval, 
                                                   batch_size=10, random_seed=42)
    print(f"批量生成10个时间间隔的请求数量: {batch_requests}")
    
    # 生成流量带宽需求
    traffic_demands = generate_service_tarffic_bandwidth_demand(batch_requests, traffic_demand_low=10, traffic_demand_high=40)
    print(f"生成的流量带宽需求 (Mbps): {traffic_demands}")
    
    # 统计信息
    stats = calculate_arrival_statistics(arrival_rate, time_interval)
    print(f"统计信息: {stats}")
    
    return requests, batch_requests, stats


if __name__ == "__main__":
    demo_poisson_requests()