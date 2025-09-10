class TrafficFlow:
    """
    A class to simulate traffic flow in a network.
    """

    def __init__(self, id, source, destination, flow_rate, survival_time):
        """
        Initialize the TrafficFlow object.

        :param source: The source node of the traffic flow.
        :param destination: The destination node of the traffic flow.
        :param flow_rate: The rate of traffic flow (e.g., packets per second).
        """
        self._source = source
        self._destination = destination
        self._flow_rate = flow_rate
        self._id = id
        self._init_survival_time = survival_time
        self._survival_time = survival_time
        self.current_flow = 0
        self._path = []  # 初始化私有属性，而不是公共属性

    def update_flow(self, time_interval):
        """
        Update the current flow based on the flow rate and time interval.

        :param time_interval: The time interval over which to update the flow.
        """
        self.current_flow += self.flow_rate * time_interval

    def reset_flow(self):
        """
        Reset the current flow to zero.
        """
        self.current_flow = 0

    @property
    def id(self):
        """
        Get the ID of the traffic flow.
        """
        return self._id
    
    @property
    def source(self):
        """
        Get the source node of the traffic flow.
        """
        return self._source
    @property
    def destination(self):
        """ 
        Get the destination node of the traffic flow.
        """
        return self._destination
    @property
    def flow_rate(self):
        """
        Get the flow rate of the traffic flow.
        """
        return self._flow_rate

    @property
    def init_survival_time(self):
        """
        Get the initial survival time of the traffic flow.
        """
        return self._init_survival_time
    
    @property
    def survival_time(self):
        """
        Get the survival time of the traffic flow.
        """
        return self._survival_time
    
    @survival_time.setter
    def survival_time(self, new_survival_time):
        """
        Set a new survival time for the traffic flow.

        :param new_survival_time: The new survival time to set.
        """
        if new_survival_time < 0:
            raise ValueError("Survival time must be non-negative")
        self._survival_time = new_survival_time

    @property
    def path(self):
        """
        Get the current path of the traffic flow.
        """
        return self._path
    
    @path.setter
    def path(self, new_path):
        """
        Set a new path for the traffic flow.

        :param new_path: A list representing the new path of nodes.
        """
        if new_path is None:
            self._path = []
        elif isinstance(new_path, list):
            self._path = new_path.copy()  # 创建副本以避免外部修改
        else:
            raise TypeError("Path must be a list or None")
    
    @path.deleter
    def path(self):
        """
        Delete the path (reset to empty list).
        """
        self._path = []
    
    def __str__(self):
        """
        Return a string representation of the traffic flow.
        """
        return f"TrafficFlow(Source: {self._source}, Destination: {self._destination}, Flow Rate: {self._flow_rate}, Current Flow: {self.current_flow})"
    

# Class which stores all the traffic flows in the network
class TrafficFlowsManager:
    """
    A class to manage multiple traffic flows in the network.
    """

    def __init__(self):
        """
        Initialize the TrafficFlowManager object.
        """
        self.flows = []
        self.expired_flows = []
        self.tobe_assigned_flows = []

    def add_flow(self, flow):
        """
        Add a new traffic flow to the manager.

        :param flow: An instance of TrafficFlow to be added.
        """
        if isinstance(flow, TrafficFlow):
            self.tobe_assigned_flows.append(flow)
        else:
            raise TypeError("Only TrafficFlow instances can be added")
        
    def merge_assinged_flows(self):
        """
        Merge the to-be-assigned flows into the main flows list.
        """
        self.flows.extend(self.tobe_assigned_flows)
        self.tobe_assigned_flows = []

    def remove_flow(self, flow):
        """
        Remove a traffic flow from the manager.

        :param flow: An instance of TrafficFlow to be removed.
        """
        if flow in self.flows:
            self.flows.remove(flow)
            self.expired_flows.append(flow)
        else:
            raise ValueError("The specified flow is not managed by this manager")

    def get_expired_flows(self):
        """
        Get a list of expired traffic flows.

        :return: A list of expired TrafficFlow instances.
        """
        return self.expired_flows.copy()

    def get_all_flows(self):
        """
        Get a list of all managed traffic flows.

        :return: A list of TrafficFlow instances.
        """
        return self.flows.copy()  # Return a copy to prevent external modification
    
    def get_number_of_flows(self):
        """
        Get the number of managed traffic flows.

        :return: The number of TrafficFlow instances.
        """
        return len(self.flows)
    
    def get_one_flow(self, index):
        """
        Get a specific traffic flow by index.

        :param index: The index of the traffic flow to retrieve.
        :return: The TrafficFlow instance at the specified index.
        """
        if 0 <= index < len(self.flows):
            return self.flows[index]
        else:
            raise IndexError("Index out of range")
        
    def update_flow_survival_times(self, time_interval=1):
        """
        Update the survival times of all managed traffic flows.

        :param time_interval: The time interval over which to update survival times.
        """
        expired_flows = []
        for flow in self.flows:
            new_survival_time = flow.survival_time - time_interval
            if new_survival_time <= 0:
                expired_flows.append(flow)
                
            else:
                flow.survival_time = new_survival_time
        
        for flow in expired_flows:
            self.remove_flow(flow)