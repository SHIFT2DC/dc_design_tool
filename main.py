import pandas as pd
import pandapower as pp
from utilities_create import create_DC_network
from utilities_plot import plot_network_evaluation_results_with_plotly, save_sizing_results_to_excel, plot_network_sizing_results_with_plotly
from utilities_load_flow import perform_dc_load_flow, perform_load_flow_with_sizing, perform_dc_load_flow_with_droop, perform_timestep_dc_load_flow
from utilities_worst_case_sizing import perform_comprehensive_sizing, validate_network_performance

# Insert the path of the input file
path = 'input_file_grid_data.xlsx'
# Insert the path of the catalogues
path_cable_catalogue = "catalogue_cable.xlsx"
path_converter_catalogue = "catalogue_converter.xlsx"

# Read files and create an initial un-sized DC network
net, cable_catalogue, use_case, node_data = create_DC_network(path, path_cable_catalogue, path_converter_catalogue)

# Size the network according to the worst case scenarios defined in the input file and to the catalogues
net = perform_comprehensive_sizing(net, cable_catalogue, use_case, node_data)
# Save sizing results - sized network
save_sizing_results_to_excel(net, node_data, 'output_sizing_results_file.xlsx')
plot_network_sizing_results_with_plotly(net, node_data, rf'output_sizing_results_plot_network.html')

# Evaluate the sized network's performance under the 3 worst-case scenarios, and save results
scenario1, scenario2, scenario3 = validate_network_performance(net, use_case, node_data)

'''
# Execute a load flow with droop control considering time-series data (user-defined or generated profiles)
net, results = perform_timestep_dc_load_flow(net, use_case, node_data)
# Save load flow results 
results.to_excel("timesteps_LF_results.xlsx")
'''
