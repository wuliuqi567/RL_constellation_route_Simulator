class TrafficFlow:
    """
    A class to simulate traffic flow in a network.
    """

    def __init__(self, source, destination, flow_rate, id):
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

    def add_flow(self, flow):
        """
        Add a new traffic flow to the manager.

        :param flow: An instance of TrafficFlow to be added.
        """
        if isinstance(flow, TrafficFlow):
            self.flows.append(flow)
        else:
            raise TypeError("Only TrafficFlow instances can be added")

    def remove_flow(self, flow):
        """
        Remove a traffic flow from the manager.

        :param flow: An instance of TrafficFlow to be removed.
        """
        if flow in self.flows:
            self.flows.remove(flow)
        else:
            raise ValueError("The specified flow is not managed by this manager")

    def get_all_flows(self):
        """
        Get a list of all managed traffic flows.

        :return: A list of TrafficFlow instances.
        """
        return self.flows.copy()  # Return a copy to prevent external modification
    
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