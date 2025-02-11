import pandas as pd
import pandapower as pp
from create import create_DC_network
from plot_utilities import plot_network_with_plotly
from loadflow_utilities import perform_dc_load_flow,perform_load_flow_with_sizing,perform_dc_load_flow_with_droop,perform_timestep_dc_load_flow
from worst_case_utilities import perform_comprehensive_sizing,validate_network_performance
from tqdm import tqdm



path = 'grid_data_input_file_building_demo.xlsx'
#path = 'grid_data_input_file_WIP_v1.xlsx'
path_cable_catalogue = "cable_catalogue.xlsx"
path_converter_catalogue = "Converters_Library.xlsx"

net, cable_catalogue, use_case = create_DC_network(path, path_cable_catalogue, path_converter_catalogue)

net=perform_comprehensive_sizing(net,cable_catalogue,use_case)

#net = perform_dc_load_flow(net, use_case)

#net = perform_dc_load_flow_with_droop(net, use_case)

perform_timestep_dc_load_flow(net,use_case)

#net=perform_dc_load_flow(net,use_case,PDU_droop_control=True)
#plot_network_with_plotly(net)

#plot_network_with_plotly(net)

#validate_network_performance(big_net,use_case)
