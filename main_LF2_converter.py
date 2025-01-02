
import pandas as pd
import pandapower as pp
from create import read_DC_csv,create_DC_network,create_DC_network_with_converter
from plot_utilities import plot_voltage   
from loadflow_utilities import ldf_DC_converter,optimisation_cable_size

xl=read_DC_csv('grid_data_input_file.xlsx')
path_cable_catalogue="cable_catalogue.xlsx"

net,cable_catalogue=create_DC_network_with_converter(xl,path_cable_catalogue)

new_net=ldf_DC_converter(net)

plot_voltage(new_net)



