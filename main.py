import pandas as pd
import pandapower as pp
from create import create_DC_network
from plot_utilities import plot_network_with_plotly
from loadflow_utilities import perform_dc_load_flow,perform_load_flow_with_sizing,perform_dc_load_flow_with_droop,perform_timestep_dc_load_flow
from worst_case_utilities import perform_comprehensive_sizing,validate_network_performance

# Insert the path of the input file : 
path = 'grid_data_input_file_building_demo.xlsx'

# Insert the path of the catalogues : 
path_cable_catalogue = "cable_catalogue.xlsx"
path_converter_catalogue = "Converters_Library.xlsx"

#Read files and create an un-sized network :
net, cable_catalogue, use_case = create_DC_network(path, path_cable_catalogue, path_converter_catalogue)

#Sized the network according to the wortcases of the input file and catalogues
net = perform_comprehensive_sizing(net, cable_catalogue, use_case)

#Plot and save the result of the sized network on the 3 worstcases
scenario1, scenario2, scenario3 = validate_network_performance(net, use_case)

#Run a DC loadflow with droop control on the given or generated profiles
net, results = perform_timestep_dc_load_flow(net, use_case)

#save result of the loadflow
results.to_excel("timesteps_LF_results.xlsx")