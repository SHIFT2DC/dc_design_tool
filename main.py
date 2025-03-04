import pandas as pd
import pandapower as pp
from utilities_create import create_dc_network
from utilities_plot import (plot_network_evaluation_results_with_plotly, save_sizing_results_to_excel,
                            plot_network_sizing_results_with_plotly, save_kpis_results_to_excel)
from utilities_load_flow import (perform_dc_load_flow, perform_load_flow_with_sizing, perform_dc_load_flow_with_droop,
                                 perform_timestep_dc_load_flow)
from utilities_worst_case_sizing import perform_comprehensive_sizing, validate_network_performance
from utilities_kpis import (calculate_efficiency_kpi, calculate_total_investment_cost_cables_converters_kpi,
                            calculate_total_weight_cables_converters_kpi, calculate_lifecycle_emissions_kpi)

# Insert the path of the input file
path = 'input_file_grid_data.xlsx'
# Insert the path of the catalogues
path_cable_catalogue = "catalogue_cable.xlsx"
path_converter_catalogue = "catalogue_converter.xlsx"

# Read files and create an initial un-sized DC network
net, cable_catalogue, use_case, node_data = create_dc_network(path, path_cable_catalogue, path_converter_catalogue)

# Size the network according to the worst case scenarios defined in the input file and to the catalogues
net = perform_comprehensive_sizing(net, cable_catalogue, use_case, node_data)
# Save sizing results - sized network
save_sizing_results_to_excel(net, node_data, 'output_sizing_results_file.xlsx')
plot_network_sizing_results_with_plotly(net, node_data, rf'output_sizing_results_plot_network.html')

# Evaluate the sized network's performance under the 3 worst-case scenarios, and save results
scenario1, scenario2, scenario3 = validate_network_performance(net, use_case, node_data)

# Execute a load flow with droop control considering time-series data (user-defined or generated profiles)
net_snapshots, results = perform_timestep_dc_load_flow(net, use_case, node_data)
# Save load flow results 
results.to_excel("timesteps_LF_results.xlsx")

# Calculate KPIs
# Efficiency
timestep_hours = use_case['Parameters for annual simulations']['Simulation time step (mins)'] / 60
(efficiency_ratio, total_consumed_energy_mwh, total_generated_energy_mwh,
 total_losses_cables_mwh, total_losses_converters_mwh) = calculate_efficiency_kpi(net_snapshots, timestep_hours)
# Economic
total_capex_keur, capex_details = calculate_total_investment_cost_cables_converters_kpi(net, use_case,
                                                                                        path_converter_catalogue)
# Environmental
total_weight_kg, weight_details = calculate_total_weight_cables_converters_kpi(net, use_case, path_converter_catalogue)
total_lifecycle_emissions_kg_co2 = calculate_lifecycle_emissions_kpi(net, use_case, path_converter_catalogue)
# Save KPIs results
save_kpis_results_to_excel(
    'output_kpis_results_file.xlsx',
    (efficiency_ratio, total_consumed_energy_mwh, total_generated_energy_mwh, total_losses_cables_mwh,
     total_losses_converters_mwh),
    (total_capex_keur, capex_details),
    (total_weight_kg, weight_details, total_lifecycle_emissions_kg_co2)
)

