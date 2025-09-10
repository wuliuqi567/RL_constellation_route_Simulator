"""

Author : zhifenghan

Date : 2025/05/10

Function : This script is used to test the functionality of the traffic generation plugin.
           The generated traffic will be under 'data/'

"""

import src.XML_constellation.constellation_traffic.traffic_plugin_manager as traffic_plugin_manager
import src.XML_constellation.constellation_routing.routing_policy_plugin_manager as routing_plugin_manager
from src.constellation_generation.by_duration.constellation_configuration import constellation_configuration

def traffic_generation():
    duration = 10
    dT = 1
    cons_name = 'Starlink'
    constellation = constellation_configuration(duration, dT, cons_name, shell_index=1)

    Traffic = traffic_plugin_manager.traffic_plugin_manager()
    for t in range(1, duration + 1):
        Traffic.execute_traffic_policy(constellation, t)

    print("Traffic generation is completed for " + str(duration) + " s.")
    print()


import kits.get_h5file_tree_structure as GET_H5FILE_TREE_STRUCTURE
import h5py
import numpy as np
def view_h5file_tree_structure():
    with h5py.File('data/XML_constellation/Starlink_shell1.h5', 'r') as file:
        GET_H5FILE_TREE_STRUCTURE.print_hdf5_structure(file)
        

def get_h5file_satellite_delay_data():
    # read the delay matrix data in the h5 file
    with h5py.File('data/XML_constellation/Starlink_shell1.h5', 'r') as file:
        # access the existing first-level subgroup delay group
        delay = file['delay']
        # access the existing secondary subgroup 'shell'+str(count) subgroup
        current_shell_group = delay['shell1']
        # read dataset
        delay_matrix = np.array(current_shell_group['timeslot1']).tolist()

    print(delay_matrix)

if __name__ == '__main__':
    traffic_generation()
    # get_h5file_satellite_delay_data()
