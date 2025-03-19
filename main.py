import pandas as pd
import pandapower as pp
from utilities_create import create_dc_network
from utilities_plot_save import (plot_network_evaluation_results_with_plotly, save_sizing_results_to_excel,
                                 plot_network_sizing_results_with_plotly, save_kpis_results_to_excel)
from utilities_load_flow import (perform_dc_load_flow, perform_load_flow_with_sizing, perform_dc_load_flow_with_droop,
                                 perform_timestep_dc_load_flow)
from utilities_worst_case_sizing import perform_comprehensive_sizing, validate_network_performance
from utilities_kpis import (calculate_efficiency_kpi, calculate_total_investment_cost_cables_converters_kpi,
                            calculate_total_maintenance_cost_cables_converters_kpi,
                            calculate_total_weight_cables_converters_kpi, calculate_lifecycle_emissions_kpi,
                            calculate_energy_savings, calculate_total_capex_difference)

import os

##########################################################
# Insert the path of the input file
# path = 'input_file_grid_data.xlsx'
# path = 'input_file_grid_data_building.xlsx'
# path = 'input_file_grid_data_port.xlsx'
# path = 'input_file_grid_data_datacenter.xlsx'
# Insert the path of the catalogues
path_cable_catalogue = "catalog_cable.xlsx"
path_converter_catalogue = "catalog_converter.xlsx"

# Define the output directory
output_dir = "output"
# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

##########################################################
# Read files and create an initial un-sized DC network
net, cable_catalogue, use_case, node_data = create_dc_network(path, path_cable_catalogue, path_converter_catalogue)

##########################################################
# Size the network according to the worst case scenarios defined in the input file and to the catalogues
net = perform_comprehensive_sizing(net, cable_catalogue, use_case, node_data)
# Save sizing results - sized network
save_sizing_results_to_excel(net, node_data, os.path.join(output_dir, 'output_sizing_results_file.xlsx'))
plot_network_sizing_results_with_plotly(net, node_data, os.path.join(output_dir, rf'output_sizing_results_plot_network.html'))

# Evaluate the sized network's performance under the 3 worst-case scenarios, and save results
scenario1, scenario2, scenario3 = validate_network_performance(net, use_case, node_data)

# Execute a load flow with droop control considering time-series data (user-defined or generated profiles)
net_snapshots, results = perform_timestep_dc_load_flow(net, use_case, node_data)
# Save load flow results
results.to_excel(os.path.join(output_dir, "output_timesteps_LF_results.xlsx"))

##########################################################
# Calculate KPIs for DC grid
# Efficiency
timestep_hours = use_case['Parameters for annual simulations']['Simulation time step (mins)'] / 60
(efficiency_ratio, total_consumed_energy_mwh, total_generated_energy_mwh,
 total_losses_cables_mwh, total_losses_converters_mwh) = calculate_efficiency_kpi(net_snapshots, timestep_hours)
# Economic
total_capex_keur, capex_details = calculate_total_investment_cost_cables_converters_kpi(net, use_case,
                                                                                        path_converter_catalogue)
total_opex_keur, opex_details = calculate_total_maintenance_cost_cables_converters_kpi(net, use_case)
# Environmental
total_weight_kg, weight_details = calculate_total_weight_cables_converters_kpi(net, use_case, path_converter_catalogue)
total_lifecycle_emissions_kg_co2 = calculate_lifecycle_emissions_kpi(net, use_case, path_converter_catalogue)

# Import KPIs for AC grid # Assumption: inputs from user
ac_total_generated_energy_mwh = (
    use_case)['Parameters of equivalent AC grid for KPIs comparison']['Total energy generated in AC grid (MWh)']
ac_efficiency_ratio = use_case['Parameters of equivalent AC grid for KPIs comparison']['Efficiency of AC grid (%)']
ac_total_capex_keur = use_case['Parameters of equivalent AC grid for KPIs comparison']['Total CAPEX of AC grid (kEUR)']

# Compare KPIs for AC grid and DC grid
energy_savings_mwh, energy_savings_percent = calculate_energy_savings(total_generated_energy_mwh,
                                                                      ac_total_generated_energy_mwh)
capex_difference_keur, capex_difference_percent = calculate_total_capex_difference(total_capex_keur,
                                                                                   ac_total_capex_keur)

# Save KPIs results
save_kpis_results_to_excel(
    os.path.join(output_dir, 'output_kpis_results_file.xlsx'),
    (efficiency_ratio, total_consumed_energy_mwh, total_generated_energy_mwh, total_losses_cables_mwh,
     total_losses_converters_mwh, energy_savings_mwh, energy_savings_percent),
    (total_capex_keur, capex_details, capex_difference_keur, capex_difference_percent, total_opex_keur, opex_details),
    (total_weight_kg, weight_details, total_lifecycle_emissions_kg_co2)
)
