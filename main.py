import pandas as pd
import pandapower as pp
from create import create_DC_network
from plot_utilities import plot_network_with_plotly
from loadflow_utilities import perform_dc_load_flow, perform_load_flow_with_sizing, perform_dc_load_flow_with_droop, perform_timestep_dc_load_flow
from worst_case_utilities import perform_comprehensive_sizing, validate_network_performance

# Insert the path of the input file
path = 'grid_data_input_file_building_demo.xlsx'
# Insert the path of the catalogues
path_cable_catalogue = "cable_catalogue.xlsx"
path_converter_catalogue = "Converters_Library.xlsx"

# Read files and create an initial un-sized DC network
net, cable_catalogue, use_case = create_DC_network(path, path_cable_catalogue, path_converter_catalogue)

# Size the network according to the worst case scenarios defined in the input file and to the catalogues
net = perform_comprehensive_sizing(net, cable_catalogue, use_case)

# Evaluate the sized network's performance under the 3 worst-case scenarios (plot and save results)
scenario1, scenario2, scenario3 = validate_network_performance(net, use_case)

# Execute a DC load flow with droop control considering time-series data (user-defined or generated profiles)
net, results = perform_timestep_dc_load_flow(net, use_case)
# Save load flow results to an Excel file
results.to_excel("timesteps_LF_results.xlsx")
