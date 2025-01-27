import pandas as pd
import pandapower as pp
from create import create_DC_network
from plot_utilities import plot_network_with_plotly
from loadflow_utilities import LF_DC, LF_sizing

path = 'grid_data_input_file_WIP_v1.xlsx'
path_cable_catalogue = "cable_catalogue.xlsx"
path_converter_catalogue = "Converters_Library.xlsx"

net, cable_catalogue, uc = create_DC_network(path, path_cable_catalogue, path_converter_catalogue)

net=LF_DC(net)

net = LF_sizing(net, cable_catalogue, uc)

plot_network_with_plotly(net)



# TODO sizing coef sécu
#stand by loss conv
# read droop curve ->OK
# worst case
# warning tension haute
# not cable sizing et conv
