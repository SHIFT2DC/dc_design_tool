
import pandas as pd
import pandapower as pp
from create import create_DC_network
from plot_utilities import plot_network_with_plotly   
from loadflow_utilities import New_LF,new_optimisation_cable_size

path='grid_data_input_file_WIP_v1.xlsx'
path_cable_catalogue="cable_catalogue.xlsx"

net,cable_catalogue=create_DC_network(path,path_cable_catalogue)

net=New_LF(net)

plot_network_with_plotly(net)

net=new_optimisation_cable_size(net, cable_catalogue)



plot_network_with_plotly(net)