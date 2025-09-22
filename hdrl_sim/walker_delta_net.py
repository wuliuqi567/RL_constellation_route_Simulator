import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.constellation_generation.by_XML import constellation_configuration
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure the src directory is in the Python path")
    raise

import src.XML_constellation.constellation_connectivity.connectivity_mode_plugin_manager as connectivity_mode_plugin_manager
import src.XML_constellation.constellation_evaluation.exists_ISL.delay as DELAY
import src.XML_constellation.constellation_routing.routing_policy_plugin_manager as routing_policy_plugin_manager
import math
import networkx as nx
import numpy as np
import random
import json
import h5py
from copy import deepcopy
from flow import TrafficFlow, TrafficFlowsManager
from poisson import generate_service_requests, generate_service_requests_batch, generate_service_tarffic_bandwidth_demand
class WalkerDeltaNet:
    """
    Walker Delta constellation network class for satellite routing simulation
    """
    
    def __init__(self):
        # Constants
        self.RADIUS = 6371
        self.Flow_size = 0.5
        
        # Constellation parameters
        self.cons_name = ""
        self.altitude = 0
        self.num_of_orbit = 0
        self.sat_of_orbit = 0
        self.inclination = 0
        
        # Network dimensions
        self.sat_num = 0
        self.user_num = 0
        
        # Position arrays
        self.sat_pos_car = []  # satellites' positions
        self.user_pos_car = []  # user blocks' positions
        self.GS_pos_car = []  # GS positions
        
        # Traffic and connectivity
        self.pop_count = []  # traffic probability per block
        self.user_connect_sat = []  # satellite connected to each block 
        self.sat_connect_gs = []  # GS connected to each satellite 
        self.gsl_occurrence = []  # blocks served by each GSL 
        self.gsl_occurrence_num = []  # number of blocks served by each GSL 
        self.path = []  # routing paths
        
        # Capacity and bandwidth parameters
        self.link_utilization_ratio = 100
        self.isl_capacity = 20480 
        self.uplink_capacity = self.downlink_capacity = 4096
        self.bandwidth_isl = self.isl_capacity * self.link_utilization_ratio / 100
        self.bandwidth_uplink = self.uplink_capacity * self.link_utilization_ratio / 100
        self.bandwidth_downlink = self.uplink_capacity * self.link_utilization_ratio / 100
        
        # Traffic arrays
        self.link_traffic = []  # 6 links in total for a satellite
        self.isl_traffic = []  # egress traffic per satellite
        self.isl_sender_traffic = []  # ISL sending traffic
        self.isl_receiver_traffic = []  # ISL receiving traffic
        self.downlink_traffic = []  # downlink traffic per satellite
        self.uplink_traffic = []  # uplink traffic per satellite
        self.sat_cover_pop = []  # total user traffic accessed by a satellite
        self.sat_cover_user = []  
        
        # Flow management
        self.flows = []  # all candidate flows
        self.flows_selected = {}  # selected flows
        self.curr_genflow_id = 0
        self.flows_num = 0
        self.flows_cumulate_weight = []
        self.flows_sum_weight = 0
        self.constellation = self.constellation_generation()
        self.gen_nx_graph(self.constellation)
        self.traffic_manager = TrafficFlowsManager()

    def reset_net(self):
        self.clear_attr_graph()
        self.traffic_manager.reset()
    
    def clear_attr_graph(self):
        for edge in list(self.cur_timeslot_only_sat_con_graph.edges):
            self.cur_timeslot_only_sat_con_graph.edges[edge]['occupied_bandwidth'] = 0
            self.cur_timeslot_only_sat_con_graph.edges[edge]['utilization'] = 0.0
            self.cur_timeslot_only_sat_con_graph.edges[edge]['available_bandwidth'] = self.cur_timeslot_only_sat_con_graph.edges[edge]['capacity']
    
    def cir_to_car_np(self, lat, lng, h):
        """Convert circular coordinates to Cartesian coordinates"""
        x = (self.RADIUS + h) * math.cos(math.radians(lat)) * math.cos(
            math.radians(lng))
        y = (self.RADIUS + h) * math.cos(math.radians(lat)) * math.sin(
            math.radians(lng))
        z = (self.RADIUS + h) * math.sin(math.radians(lat))
        return np.array([x, y, z])

    def link_seq(self, sati, satj):
        """Determine link sequence between satellites"""
        orbiti = int(sati / self.sat_of_orbit)
        orbitj = int(satj / self.sat_of_orbit)
        if orbiti == orbitj and (satj == sati + 1 or satj == sati -
                                 (self.sat_of_orbit - 1)):
            return sati * 6
        elif satj == (sati + self.sat_of_orbit) % self.sat_num:
            return sati * 6 + 1
        elif orbiti == orbitj and (satj == sati - 1 or satj == sati +
                                   (self.sat_of_orbit - 1)):
            return sati * 6 + 2
        elif satj == (sati - self.sat_of_orbit + self.sat_num) % self.sat_num:
            return sati * 6 + 3
        else:
            return -1

    def constellation_generation(self):
        """Generate constellation and initialize network parameters"""
        constellation_name = "walker_delta_144"  # 12 orbits, 12 satellites per orbit
        
        # Generate the constellation
        constellation = constellation_configuration.constellation_configuration(
            dT=100, constellation_name=constellation_name)
        connectionModePluginManager = connectivity_mode_plugin_manager.connectivity_mode_plugin_manager()
        # execute the connectivity mode and build ISLs between satellites
        connectionModePluginManager.execute_connection_policy(constellation=constellation , dT=100)
        # initialize the routing policy plugin manager
        
        print('\t\t\tDetails of the constellations are as follows :')
        print('\t\t\tThe name of the constellation is : ', constellation.constellation_name)
        print('\t\t\tThere are ', constellation.number_of_shells, ' shell(s) in this constellation')
        print('\t\t\tThe information for each shell is as follows:')
        for sh in constellation.shells:
            print('\t\t\tshell name : ', sh.shell_name)
            print('\t\t\tshell orbit altitude(km) : ', sh.altitude)
            print('\t\t\tThe shell contains ', sh.number_of_satellites, ' satellites')
            print('\t\t\tThe shell contains ', sh.number_of_orbits, ' orbits')
            print('\t\t\tshell orbital inclination(°) : ', sh.inclination)
            print('\t\t\tshell orbital period (s) : ', sh.orbit_cycle)
            print('\t\t\t==============================================')

        self.cons_name = constellation.constellation_name
        shell = constellation.shells[0]  # the first shell
        self.altitude = shell.altitude
        self.num_of_orbit = shell.number_of_orbits
        self.sat_of_orbit = shell.number_of_satellite_per_orbit
        self.inclination = shell.inclination
        self.sat_num = self.num_of_orbit * self.sat_of_orbit
        self.user_num = math.ceil(self.inclination) * 2 * 360
        
        # Initialize arrays
        self.sat_cover_pop = [0] * self.sat_num
        self.sat_cover_user = [[] for _ in range(self.sat_num)]
        
        self.all_domains = self.partition_square_domain(row=4, col=4, num_orbs=self.num_of_orbit, 
                                                   num_sats_per_orb=self.sat_of_orbit)
        n_row_domains = self.sat_of_orbit // 4  # domains in orbit direction
        n_col_domains = self.num_of_orbit // 4  # domains in cross-orbit direction
        self.domain_graph = self.gen_domain_graph(n_row_domains, n_col_domains)
        print(f"Total {len(self.all_domains)} domains generated, each domain contains 16 satellites.")
        print(f"Domain graph has {self.domain_graph.number_of_nodes()} nodes and {self.domain_graph.number_of_edges()} edges.")

        minimum_elevation = math.radians(25)
        bound = math.sqrt((self.RADIUS + self.altitude)**2 - 
                         (self.RADIUS * math.cos(minimum_elevation))**2) - \
                self.RADIUS * math.sin(minimum_elevation)
        self.Flow_size = 0.5

        # Load satellite positions
        h5file_path = os.path.join(project_root, "data", "XML_constellation", "walker_delta_144.h5")
        with h5py.File(h5file_path, 'r') as file:
            position_group = file['position']
            shell_group = position_group['shell1']
            position_tt = np.array(shell_group['timeslot' + str(1)])
            for lla in position_tt:
                self.sat_pos_car.append(
                    self.cir_to_car_np(
                        float(lla[1]), float(lla[0]), float(lla[2])))
            self.sat_pos_car = np.array(self.sat_pos_car)
        
        self.inclination = math.ceil(self.inclination)
        # Load user positions
        for lat in range(self.inclination, -self.inclination, -1):
            for lon in range(-180, 180, 1):
                self.user_pos_car.append(self.cir_to_car_np(lat - 0.5, lon + 0.5, 0))
        self.user_pos_car = np.array(self.user_pos_car)
        
        # Load traffic distribution
        traffic_file = os.path.join(project_root, "data", "starlink_count.txt")
        with open(traffic_file, 'r') as file:
            lines = file.readlines()
            for row in range(90 - self.inclination, 90 + self.inclination):
                self.pop_count.extend([float(x) for x in lines[row].split(' ')[:-1]] + [0])

        print("generating +Grid traffic for timeslot: " + str(1) + "...")  

        # Determine satellite each block is connected to
        for user_id in range(self.inclination * 2 * 360):
            dis2 = np.sqrt(
                np.sum(np.square(self.sat_pos_car - self.user_pos_car[user_id]),
                       axis=1)) 
            if min(dis2) > bound:
                self.user_connect_sat.append(-1) 
                continue
            min_dis_sat = np.argmin(dis2) 
            self.user_connect_sat.append(min_dis_sat) 
            if self.pop_count[user_id] > 0:
                self.sat_cover_pop[min_dis_sat] += self.pop_count[user_id] 
                self.sat_cover_user[min_dis_sat].append(user_id)  

        # Load GS positions
        gs_file = os.path.join(project_root, "data", "GS.json")
        with open(gs_file, "r", encoding='utf8') as f:
            GS_info = json.load(f)
        count = 0
        for key in GS_info:
            self.GS_pos_car.append(
                self.cir_to_car_np(float(GS_info[key]['lat']),
                              float(GS_info[key]['lng']), 0))
            count = count + 1
        self.GS_pos_car = np.array(self.GS_pos_car)

        # Determine GS each satellite is connected to
        for sat_id in range(self.sat_num):
            dis2 = np.sqrt(
                np.sum(np.square(self.GS_pos_car - self.sat_pos_car[sat_id]),
                       axis=1))  
            if min(dis2) > bound:
                self.sat_connect_gs.append(-1)  # -1 for no connection
                continue
            min_dis_sat = np.argmin(dis2)  
            self.sat_connect_gs.append(min_dis_sat) 
        return constellation
    
    def gen_nx_graph(self, constellation):
        constellation_name = "walker_delta_144"
        routingPolicyPluginManager = routing_policy_plugin_manager.routing_policy_plugin_manager()
        routingPolicyPluginManager.set_routing_policy("satellite_connection_graph")
        self.cur_timeslot_only_sat_con_graph = routingPolicyPluginManager.execute_connection_policy(constellation_name , constellation.shells[0], 1)
        
    
    def partition_square_domain(self, row=4, col=4, num_orbs=12, num_sats_per_orb=12):
        """
        Partition the constellation into domains using a grid layout.
        Each domain contains satellites from adjacent orbits and positions
        """
        domains = []
        
        # Calculate number of domains in each dimension
        n_domain_rows = num_sats_per_orb // row  # domains in orbit direction
        n_domain_cols = num_orbs // col          # domains in cross-orbit direction
        
        # Column-first traversal order for domain ID assignment
        for domain_col in range(n_domain_cols):  # traverse columns (orbit direction)
            for domain_row in range(n_domain_rows):  # traverse rows (intra-orbit direction)
                domain_satellites = []
                
                # Assign satellites within current domain
                for orbit_offset in range(col):  # traverse orbits in domain
                    orbit_idx = domain_col * col + orbit_offset
                    
                    for sat_offset in range(row):  # traverse positions in domain
                        sat_idx = domain_row * row + sat_offset
                        
                        # Calculate global satellite ID
                        satellite_id = orbit_idx * num_sats_per_orb + sat_idx
                        domain_satellites.append(satellite_id+1)
                
                domains.append(sorted(domain_satellites))
        
        return domains

    def gen_domain_graph(self, n_row_domains, n_col_domains):
        """
        Generate domain graph where each domain is a node and adjacent domains are connected
        """
        num_domain = n_row_domains * n_col_domains
        domain_graph = nx.Graph()
        
        # Add nodes for each domain
        for i in range(num_domain):
            domain_graph.add_node(i, name=f"Domain{i}")
        
        # Add edges between adjacent domains
        for domain_id in range(num_domain):
            # Calculate current domain's grid coordinates
            domain_row = domain_id % n_row_domains  # row coordinate
            domain_col = domain_id // n_row_domains  # column coordinate
            
            # Connect to domain on the right
            right_col = (domain_col + 1) % n_col_domains
            right_domain = right_col * n_row_domains + domain_row
            domain_graph.add_edge(domain_id, right_domain)
            
            # Connect to domain below
            down_row = (domain_row + 1) % n_row_domains
            down_domain = domain_col * n_row_domains + down_row
            domain_graph.add_edge(domain_id, down_domain)
        
        return domain_graph

    def get_domain_links_and_pos_info(self, domain_id, all_domains: list):
        """
        Get satellite list and positions within a domain
        """
        if domain_id < 0 or domain_id >= len(all_domains):
            raise ValueError("Invalid domain ID")
        
        domain_satellites = all_domains[domain_id]
        domain_positions = self.sat_pos_car[domain_satellites]
        
        return domain_satellites, domain_positions
    
    def randomGenFlows(self, arrival_rate=10, time_interval=1.0):
        """Generate random traffic flows based on Poisson distribution"""
        
        # Generate service requests using Poisson distribution

        requests = generate_service_requests(arrival_rate, time_interval)
        traffic_demands = generate_service_tarffic_bandwidth_demand(
            [requests], traffic_demand_low=10, traffic_demand_high=40)[0]
        
        # select source and destination for each flow based on the number of population in each user block
        for i in range(traffic_demands):
            if sum(self.pop_count) == 0:
                print("No traffic demand in this time slot.")
                break
            rand_value = random.uniform(0, sum(self.pop_count))
            cumulative = 0
            for user_id, pop in enumerate(self.pop_count):
                cumulative += pop
                if cumulative >= rand_value:
                    source_block = user_id
                    break
            source_sat = self.user_connect_sat[source_block]
            if source_sat == -1 or self.sat_cover_pop[source_sat] < 1:
                print(f"Source Block {source_block} cannot be served, skipping this flow.")
                continue
            
            while True:
                dest_block = random.randint(0, self.user_num - 1)
                if dest_block != source_block and self.pop_count[dest_block] > 0:
                    break
            dest_sat = self.user_connect_sat[dest_block]
            if dest_sat == -1 or self.sat_cover_pop[dest_sat] < 1:
                print(f"Dest Block {dest_block} cannot be served, skipping this flow.")
                continue
            
            # random init survival time between 7-10 seconds
            survival_time = random.randint(7, 10)
            
            flow_bandwidth = random.randint(10, 40)
            
            flow = TrafficFlow(self.curr_genflow_id, source_block, dest_block, flow_bandwidth, survival_time)
            self.curr_genflow_id += 1
            self.traffic_manager.add_flow(flow)
            
        
        self.flows_num = self.traffic_manager.get_number_of_flows()

        print(f"Total Generated {self.flows_num} traffic flows.")
        print(f"current generated flow count: {traffic_demands}")

    def gen_each_domain_flows(self, domain_id, arrival_rate=10, time_interval=1.0):
        """Generate traffic flows for a specific domain"""
        if domain_id < 0 or domain_id >= len(self.all_domains):
            raise ValueError("Invalid domain ID")
        
        domain_satellites = self.all_domains[domain_id]
        requests = generate_service_requests(arrival_rate, time_interval, random_seed=10)
        traffic_demands = generate_service_tarffic_bandwidth_demand(
            [requests], traffic_demand_low=10, traffic_demand_high=40)[0]
        
        for each_flow_bandwidth in traffic_demands:
            
            source_sat = 'satellite_' + str(random.choice(domain_satellites))
            dest_sat = 'satellite_' + str(random.choice(domain_satellites))
            while True:
                if source_sat == dest_sat:
                    dest_sat = 'satellite_' + str(random.choice(domain_satellites))
                else:
                    break

            survival_time = random.randint(7, 10)
            flow_bandwidth = each_flow_bandwidth
            
            flow = TrafficFlow(self.curr_genflow_id, source_sat, dest_sat, flow_bandwidth, survival_time)
            self.curr_genflow_id += 1
            self.traffic_manager.add_flow(flow)
        
        self.flows_num = self.traffic_manager.get_number_of_flows()
        # print(f"Total Generated {self.flows_num} traffic flows after adding domain {domain_id}.")


    def inject_flows(self):
        """Inject newly generated flows into the network"""
        new_flows = self.traffic_manager.tobe_assigned_flows
        if not new_flows:
            print("No new flows to inject.")
            return
        
        for flow in new_flows:
            print(f"Injecting Flow ID {flow.id}: Source Block {flow.source} -> Dest Block {flow.destination}, "
                  f"Bandwidth {flow.flow_rate} Mbps, Survival Time {flow.survival_time}s")
            shorest_path = self.calculate_routing_path(flow.source, flow.destination)
            flow.path = shorest_path
            
        
        # according to the assigned path, update the ISL and satellite traffic
        self.assign_flows_to_network(self)
        
        self.traffic_manager.merge_assinged_flows()
        print(f"Total flows after injection: {self.traffic_manager.get_number_of_flows()}")
    
    def assign_flows_to_network(self, flows):
        """Assign flows to the network and update traffic states"""
        for flow in flows:
            path = flow.path
            flow_rate = flow.flow_rate
            
            if not path or flow_rate <= 0:
                print(f"Flow ID {flow.id} has invalid path or flow rate, skipping assignment.")
                continue
            
            # Update traffic on each link in the flow's path
            for i in range(len(path) - 1):
                sat_i = path[i]
                sat_j = path[i + 1]
                
                # utilization = self.cur_timeslot_only_sat_con_graph[sat_i][sat_j]['utilization']
                occupied_bandwidth = self.cur_timeslot_only_sat_con_graph[sat_i][sat_j]['occupied_bandwidth']
                capacity = self.cur_timeslot_only_sat_con_graph[sat_i][sat_j]['capacity']
                
                if occupied_bandwidth + flow_rate > capacity:
                    print(f"Flow ID {flow.id} cannot be assigned due to capacity constraints on link {sat_i}-{sat_j}.")
                    break
                
                self.cur_timeslot_only_sat_con_graph[sat_i][sat_j]['occupied_bandwidth'] += flow_rate
                self.cur_timeslot_only_sat_con_graph[sat_i][sat_j]['utilization'] = min(
                    1.0, (self.cur_timeslot_only_sat_con_graph[sat_i][sat_j]['occupied_bandwidth']) / capacity
                )
                self.cur_timeslot_only_sat_con_graph[sat_i][sat_j]['available_bandwidth'] = max(
                    0, self.cur_timeslot_only_sat_con_graph[sat_i][sat_j]['capacity'] - self.cur_timeslot_only_sat_con_graph[sat_i][sat_j]['occupied_bandwidth']
                )
                
                
    
    def update_isl_state(self):
        """
            update ISL bandwidth based on current expired flows
        """
        expired_flows = self.traffic_manager.get_expired_flows()
        for flow in expired_flows:
            flow_path = flow.path
            flow_rate = flow.flow_rate
            if not flow_path or flow_rate <= 0:
                continue
            for i in range(len(flow_path) - 1):
                sat_i = flow_path[i]
                sat_j = flow_path[i + 1]
                
                # Update occupied bandwidth first
                self.cur_timeslot_only_sat_con_graph[sat_i][sat_j]['occupied_bandwidth'] = max(
                    0, self.cur_timeslot_only_sat_con_graph[sat_i][sat_j]['occupied_bandwidth'] - flow_rate
                )
                
                # Recalculate utilization based on occupied bandwidth
                capacity = self.cur_timeslot_only_sat_con_graph[sat_i][sat_j]['capacity']
                occupied_bandwidth = self.cur_timeslot_only_sat_con_graph[sat_i][sat_j]['occupied_bandwidth']
                self.cur_timeslot_only_sat_con_graph[sat_i][sat_j]['utilization'] = occupied_bandwidth / capacity if capacity > 0 else 0.0
                
                # Update available bandwidth
                self.cur_timeslot_only_sat_con_graph[sat_i][sat_j]['available_bandwidth'] = capacity - occupied_bandwidth
        
        self.traffic_manager.clear_expired_flows()
                
    def calculate_routing_path(self, source_sat, dest_sat):
        """_summary_

        Args:
            source_sat (s_sat_id): "1"
            dest_sat (_type_): "10"
        """
        if isinstance(source_sat, int) or isinstance(dest_sat, int):
            str_source_sat = "satellite_" + str(source_sat)
            str_dest_sat = "satellite_" + str(dest_sat)
        else:
            str_source_sat = source_sat
            str_dest_sat = dest_sat
        
        
        path = nx.dijkstra_path(self.cur_timeslot_only_sat_con_graph, source=str_source_sat, target=str_dest_sat)
        return path
    
    
    def get_domain_state(self, domain_id):
        """Get the current state of a specific domain"""
        if domain_id < 0 or domain_id >= len(self.all_domains):
            raise ValueError("Invalid domain ID")
        
        domain_satellites = self.all_domains[domain_id]
        subdomain_graph = self.cur_timeslot_only_sat_con_graph.subgraph(
            [f"satellite_{sat}" for sat in domain_satellites])
        
        domain_state = {}
        
        for sat in domain_satellites:
            sat_node = f"satellite_{sat}"
            if sat_node in self.cur_timeslot_only_sat_con_graph:
                domain_state[sat] = deepcopy(self.cur_timeslot_only_sat_con_graph[sat_node])
            else:
                print(f"Satellite {sat} not found in the current network graph.")
        
        return domain_state
    
    def get_subdomain_graph(self, domain_id):
        """Get the subgraph of a specific domain"""
        if domain_id < 0 or domain_id >= len(self.all_domains):
            raise ValueError("Invalid domain ID")
        
        domain_satellites = self.all_domains[domain_id]
        # get the subgraph for the domain, including all edges between its satellites
        # 
        subdomain_graph = self.cur_timeslot_only_sat_con_graph.subgraph(
            [f"satellite_{sat}" for sat in domain_satellites])
        
        return subdomain_graph
    
    def get_edges_utilization(self):
        """Get the utilization of all edges in the current network graph"""
        edge_utilizations = {}
        for u, v, data in self.cur_timeslot_only_sat_con_graph.edges(data=True):
            # print(f"Edge {u}-{v} data: {data}")
            edge_utilizations[(u, v)] = data.get('utilization', 0.0)
            print(f"Edge {u}-{v}: Utilization {edge_utilizations[(u, v)]:.2f}")
        return edge_utilizations
        
if __name__ == "__main__":
    walker_net = WalkerDeltaNet()

