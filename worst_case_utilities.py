import pandapower as pp
from loadflow_utilities import perform_dc_load_flow, perform_load_flow_with_sizing
from plot_utilities import plot_network_with_plotly
import copy
import math


def calculate_storage_sizing_scenario(network, cable_catalogue, use_case):
    """
    Calculates battery storage requirements for worst-case scenario 1.
    
    Args:
        network (pp.Network): Pandapower network model
        cable_catalogue: Cable specifications database
        use_case (dict): Scenario configuration parameters
        
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
        sized_network = perform_load_flow_with_sizing(scenario_network, cable_catalogue, use_case)
        
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
        network.storage.loc[battery_mask, 'p_mw'] = 0
        network.storage.loc[battery_mask, 'max_e_mwh'] = abs(specs['energy_kwh'] / 1000)
        network.storage.loc[battery_mask, 'p_nom_mw'] = abs(specs['power_kw'] / 1000)


def create_scenario_network(network, cable_catalogue, use_case, scenario_name):
    """
    Creates a network configuration for a given scenario.
    
    Args:
        network (pp.Network): Base network model
        cable_catalogue: Cable specifications database
        use_case (dict): Scenario parameters
        scenario_name (str): Name of scenario configuration
        
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
    return perform_load_flow_with_sizing(scenario_network, cable_catalogue, use_case)


def evaluate_scenario_performance(network, use_case, scenario_name):
    """
    Evaluates network performance for a given scenario with visualization.
    
    Args:
        network (pp.Network): Network model to test
        use_case (dict): Scenario parameters
        scenario_name (str): Name of scenario configuration
        
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
    scenario_network = perform_dc_load_flow(scenario_network, use_case)
    plot_network_with_plotly(scenario_network)
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


def perform_comprehensive_sizing(network, cable_catalogue, use_case):
    """
    Executes complete network sizing process across all scenarios.
    
    Args:
        network (pp.Network): Base network model
        cable_catalogue: Cable specifications database
        use_case (dict): Scenario configuration parameters
        
    Returns:
        pp.Network: Fully sized network model
    """
    # Scenario 1: Storage sizing
    storage_network, battery_specs = calculate_storage_sizing_scenario(network, cable_catalogue, use_case)
    apply_battery_specifications(network, battery_specs)
    
    # Scenario 2: Cable and converter sizing
    scenario2_network = create_scenario_network(
        network, cable_catalogue, use_case,
        'Worst case scenario 2 for sizing of cables and  PDU DC/DC,  DC/AC, PV DC/DC and EV DC/DC converters '
    )
    
    # Scenario 3: AC/DC converter sizing
    scenario3_network = create_scenario_network(
        network, cable_catalogue, use_case,
        'Worst case scenario 3 for sizing cables and AC/DC converter'
    )
    
    # Combine results from all scenarios
    combined_network = merge_network_components(scenario2_network,storage_network)
    final_network = merge_network_components(combined_network, scenario3_network)
    
    return final_network


def validate_network_performance(network, use_case):
    """
    Validates network performance across all defined scenarios.
    
    Args:
        network (pp.Network): Network model to validate
        use_case (dict): Scenario configuration parameters
        
    Returns:
        tuple: Networks from all test scenarios
    """
    scenario1 = evaluate_scenario_performance(
        network, use_case,
        'Worst case scenario 1 for sizing of Storage DC/DC converter '
    )
    
    scenario2 = evaluate_scenario_performance(
        network, use_case,
        'Worst case scenario 2 for sizing of cables and  PDU DC/DC,  DC/AC, PV DC/DC and EV DC/DC converters '
    )
    
    scenario3 = evaluate_scenario_performance(
        network, use_case,
        'Worst case scenario 3 for sizing cables and AC/DC converter'
    )
    
    return scenario1, scenario2, scenario3