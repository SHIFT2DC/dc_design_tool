import pandas as pd
import pandapower as pp
from create import create_DC_network
from plot_utilities import plot_network_with_plotly
from loadflow_utilities import perform_dc_load_flow,perform_load_flow_with_sizing
from tqdm import tqdm

path = 'grid_data_input_file_WIP_v1.xlsx'
path_cable_catalogue = "cable_catalogue.xlsx"
path_converter_catalogue = "Converters_Library.xlsx"

net, cable_catalogue, uc = create_DC_network(path, path_cable_catalogue, path_converter_catalogue)

net=perform_dc_load_flow(net)

net = perform_load_flow_with_sizing(net, cable_catalogue, uc)

plot_network_with_plotly(net)



# TODO sizing coef sécu -> ok
#stand by loss conv -> ok
# read droop curve -> OK
# not cable sizing et conv -> ok

# worst case
# warning tension haute