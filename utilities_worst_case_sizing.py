import pandapower as pp
from utilities_load_flow import perform_dc_load_flow, perform_load_flow_with_sizing
from utilities_plot_save import plot_network_evaluation_results_with_plotly
import copy
import math
import os

# Define the output directory
output_dir = "output"
# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


def calculate_storage_sizing_scenario(network, cable_catalogue, use_case, node_data):
    """
    Calculates battery storage requirements for worst-case scenario 1.
    
    Args:
        network (pp.Network): Pandapower network model
        cable_catalogue: Cable specifications database
        use_case (dict): Scenario configuration parameters
        node_data
        
    Returns:
        tuple: (Modified network, Battery specifications dictionary)
    """
    # Create a deep copy to avoid modifying original network
    scenario_network = copy.deepcopy(network)
    
    # Extract scenario parameters
    params = use_case['Worst case scenario 1 for sizing of Storage DC/DC converter ']
    load_percent = params['Loads power factor (%)']
    load_expansion = use_case['Sizing factor']['Loads expansion factor (%)']
    pv_percent = params['PV power factor (%)']
    ev_percent = params['EV Charging station consumption (%)']
    storage_duration = params['Storage duration at nominal power (hours)']

    # Adjust network components based on scenario parameters
    scenario_network.load['p_mw'] *= (load_percent/100) * (load_expansion/100)
    scenario_network.sgen['p_mw'] *= pv_percent/100
    ev_mask = scenario_network.storage['name'].str.contains('EV')
    scenario_network.storage.loc[ev_mask, 'p_mw'] *= ev_percent/100

    # Deactivate external grid and batteries
    scenario_network.ext_grid['in_service'] = False
    battery_mask = scenario_network.storage['name'].str.contains('Battery')
    scenario_network.storage.loc[battery_mask, 'in_service'] = False

    battery_specs = {}
    for idx, battery in scenario_network.storage[battery_mask].iterrows():
        # Create temporary external grid for sizing calculation
        pp.create_ext_grid(scenario_network, bus=battery['bus'], vm_pu=1.0)
        
        # Perform load flow with component sizing
        sized_network = perform_load_flow_with_sizing(scenario_network, cable_catalogue, use_case, node_data)
        
        # Calculate battery specifications
        grid_power = sized_network.res_ext_grid.loc[sized_network.ext_grid['in_service'], 'p_mw'].values[0]
        nominal_power = 5 * math.ceil(grid_power * 1000 / 5)  # Round up to nearest 5kW
        
        battery_specs[battery['name']] = {
            "power_kw": nominal_power,
            "energy_kwh": nominal_power * storage_duration
        }
    
    return sized_network, battery_specs


def apply_battery_specifications(network, battery_specs):
    """
    Applies calculated battery specifications to the network model.
    
    Args:
        network (pp.Network): Pandapower network to modify
        battery_specs (dict): Battery specifications from sizing calculation
    """
    for name, specs in battery_specs.items():
        battery_mask = network.storage['name'] == name
        network.storage.loc[battery_mask, 'p_mw'] = abs(specs['power_kw'] / 1000)
        network.storage.loc[battery_mask, 'max_e_mwh'] = abs(specs['energy_kwh'] / 1000)
        network.storage.loc[battery_mask, 'p_nom_mw'] = abs(specs['power_kw'] / 1000)


def create_scenario_network(network, cable_catalogue, use_case, scenario_name, node_data):
    """
    Creates a network configuration for a given scenario by adjusting components (loads, generation, storage, etc.)
    based on the scenario's parameters and sizing factors.
    
    Args:
        network (pp.Network): Base network model
        cable_catalogue: Cable specifications database
        use_case (dict): Scenario parameters
        scenario_name (str): Name of scenario configuration
        node_data
        
    Returns:
        pp.Network: Configured network model
    """
    scenario_network = copy.deepcopy(network)
    params = use_case[scenario_name]
    sizing_params = use_case['Sizing factor']

    # Apply parameter adjustments
    load_percent = params['Loads power factor (%)']
    load_expansion = sizing_params['Loads expansion factor (%)']
    pv_percent = params['PV power factor (%)']
    ev_percent = params['EV Charging station consumption (%)']
    storage_percent = params.get('Storage power contribution (%)', 100)

    scenario_network.load['p_mw'] *= (load_percent/100) * (load_expansion/100)
    scenario_network.sgen['p_mw'] *= pv_percent/100
    
    # Adjust EV storage components
    ev_mask = scenario_network.storage['name'].str.contains('EV')
    scenario_network.storage.loc[ev_mask, 'p_mw'] *= ev_percent/100

    # Adjust battery storage if specified
    if 'Storage power contribution (%)' in params:
        battery_mask = scenario_network.storage['name'].str.contains('Battery')
        current_power = abs(scenario_network.storage.loc[battery_mask, 'p_mw'])
        scenario_network.storage.loc[battery_mask, 'p_mw'] = -current_power * storage_percent/100

    # Perform load flow analysis with component sizing
    return perform_load_flow_with_sizing(scenario_network, cable_catalogue, use_case, node_data)


def evaluate_scenario_performance(network, use_case, scenario_name, node_data):
    """
    Evaluates network performance for a given scenario with visualization.
    
    Args:
        network (pp.Network): Network model to test
        use_case (dict): Scenario parameters
        scenario_name (str): Name of scenario configuration
        node_data
        
    Returns:
        pp.Network: Analyzed network model
    """
    scenario_network = copy.deepcopy(network)
    params = use_case[scenario_name]
    sizing_params = use_case['Sizing factor']

    # Apply parameter adjustments
    load_percent = params['Loads power factor (%)']
    load_expansion = sizing_params['Loads expansion factor (%)']
    pv_percent = params['PV power factor (%)']
    ev_percent = params['EV Charging station consumption (%)']
    storage_percent = params.get('Storage power contribution (%)', 100)

    scenario_network.load['p_mw'] *= (load_percent/100) * (load_expansion/100)
    scenario_network.sgen['p_mw'] *= pv_percent/100
    
    # Adjust storage components
    ev_mask = scenario_network.storage['name'].str.contains('EV')
    scenario_network.storage.loc[ev_mask, 'p_mw'] *= ev_percent/100

    battery_mask = scenario_network.storage['name'].str.contains('Battery')
    current_power = abs(scenario_network.storage.loc[battery_mask, 'p_mw'])
    scenario_network.storage.loc[battery_mask, 'p_mw'] = -current_power * storage_percent/100

    # Perform and visualize load flow
    scenario_network = perform_dc_load_flow(scenario_network, use_case, node_data)
    pp.to_excel(scenario_network, os.path.join(output_dir, rf'output_evaluation_results_scenario_{scenario_name.split(" ")[3]}_file.xlsx'))
    plot_network_evaluation_results_with_plotly(
        scenario_network, node_data,
        os.path.join(output_dir,
                     rf'output_evaluation_results_scenario_{scenario_name.split(" ")[3]}_plot_network.html')
    )
    return scenario_network


def merge_network_components(base_network, comparison_network):
    """
    Merges two networks while keeping the larger-rated components.
    
    Args:
        base_network (pp.Network): Primary network model
        comparison_network (pp.Network): Secondary network model
        
    Returns:
        pp.Network: Merged network with maximum component ratings
    """
    merged_network = copy.deepcopy(base_network)
    
    # Merge converters (keep higher power ratings)
    converter_mask = comparison_network.converter['P'] > merged_network.converter['P']
    merged_network.converter.loc[converter_mask] = comparison_network.converter.loc[converter_mask]
    
    # Merge lines (keep higher current capacity)
    line_mask = comparison_network.line['max_i_ka'] > merged_network.line['max_i_ka']
    merged_network.line.loc[line_mask] = comparison_network.line.loc[line_mask]
    
    return merged_network


def perform_comprehensive_sizing(network, cable_catalogue, use_case, node_data):
    """
    Executes complete network sizing process across all scenarios.
    
    Args:
        network (pp.Network): Base network model
        cable_catalogue: Cable specifications database
        use_case (dict): Scenario configuration parameters
        node_data
        
    Returns:
        pp.Network: Fully sized network model
    """
    # Scenario 1: Storage sizing
    storage_network = None
    # Check if storage converter sizing is required
    storage_converter_mask = network.converter.loc[network.converter['type'] == 'Storage DC/DC Converter']
    storage_sizing_not_needed = network.converter.loc[storage_converter_mask.index, "conv_rank"].isna().all()
    if not storage_sizing_not_needed:  # Only perform storage sizing if required
        storage_network, battery_specs = calculate_storage_sizing_scenario(network, cable_catalogue, use_case,
                                                                           node_data)
        apply_battery_specifications(network, battery_specs)
    
    # Scenario 2: Cable and converter sizing
    scenario2_network = create_scenario_network(
        network, cable_catalogue, use_case,
        'Worst case scenario 2 for sizing of cables and  PDU DC/DC,  DC/AC, PV DC/DC and EV DC/DC converters ',
        node_data)
    
    # Scenario 3: AC/DC converter sizing
    scenario3_network = create_scenario_network(
        network, cable_catalogue, use_case,
        'Worst case scenario 3 for sizing cables and AC/DC converter', node_data)
    
    # Combine results from all scenarios
    if storage_network is not None:
        combined_network = merge_network_components(scenario2_network, storage_network)
    else:
        combined_network = scenario2_network  # Skip storage merging if storage sizing was not performed

    final_network = merge_network_components(combined_network, scenario3_network)
    
    return final_network


def validate_network_performance(network, use_case, node_data):
    """
    Validates network performance across all defined scenarios.
    
    Args:
        network (pp.Network): Network model to validate
        use_case (dict): Scenario configuration parameters
        node_data
        
    Returns:
        tuple: Networks from all test scenarios
    """
    # Check if storage converter sizing was required, to see if scenario 1 was launched or not
    storage_converter_mask = network.converter.loc[network.converter['type'] == 'Storage DC/DC Converter']
    storage_sizing_not_needed = network.converter.loc[storage_converter_mask.index, "conv_rank"].isna().all()
    if not storage_sizing_not_needed:  # Only evaluate
        scenario1 = evaluate_scenario_performance(
            network, use_case,
            'Worst case scenario 1 for sizing of Storage DC/DC converter ', node_data)
    else:
        scenario1 = None
    
    scenario2 = evaluate_scenario_performance(
        network, use_case,
        'Worst case scenario 2 for sizing of cables and  PDU DC/DC,  DC/AC, PV DC/DC and EV DC/DC converters ',
        node_data)
    
    scenario3 = evaluate_scenario_performance(
        network, use_case,
        'Worst case scenario 3 for sizing cables and AC/DC converter', node_data)
    
    return scenario1, scenario2, scenario3
