"""
DC Load Flow Module

This module provides functions for performing DC load flow calculations on electrical networks.
It handles the cleanup of network data and performs sequential load flow calculations on subnetworks.
"""

import pandapower as pp
import numpy as np
import pandas as pd
from ast import literal_eval
from utilities_net_topology import (separate_subnetworks, sorting_network, merge_networks,
                                    find_lines_between_given_line_and_ext_grid, get_bus_distances)
from typing import Dict, List
from tqdm import tqdm
import warnings


def clean_network(net: pp.pandapowerNet, original_net: pp.pandapowerNet) -> pp.pandapowerNet:
    """
    Cleans the network by removing temporary components and restoring converter data.

    Args:
        net (pp.pandapowerNet): The network to clean.
        original_net (pp.pandapowerNet): The original network with converter data.

    Returns:
        pp.pandapowerNet: The cleaned network.
    """
    # Restore converter data from the original network
    net.converter = original_net.converter
    net.res_converter = original_net.res_converter

    # Remove temporary loads used for converter emulation
    remove_temporary_loads(net)

    # Remove temporary external grids used for converter emulation
    remove_temporary_ext_grids(net)

    return net


def remove_temporary_loads(net: pp.pandapowerNet) -> None:
    """
    Removes temporary loads from the network.

    Args:
        net (pp.pandapowerNet): The network to clean.
    """
    del_load = ['Load of net' in str(x) for x in net.load.name.values]
    net.load.drop(net.load.loc[del_load].index, inplace=True)
    net.res_load.drop(net.res_load.loc[del_load].index, inplace=True)


def remove_temporary_ext_grids(net: pp.pandapowerNet) -> None:
    """
    Removes temporary external grids from the network.

    Args:
        net (pp.pandapowerNet): The network to clean.
    """
    del_ext_grid = ['Converter emulation' in str(x) for x in net.ext_grid.name.values]
    net.ext_grid.drop(net.ext_grid.loc[del_ext_grid].index, inplace=True)
    net.res_ext_grid.drop(net.res_ext_grid.loc[del_ext_grid].index, inplace=True)


def calculate_converter_power(power: float, converter: pd.Series) -> tuple:
    """
    Calculates the converter power flow and losses.

    Args:
        power (float): Power flow through the converter in MW.
        converter (pd.Series): Converter data series.

    Returns:
        tuple: A tuple containing the adjusted power, efficiency, and power loss.
    """
    # Get efficiency from interpolation of the efficiency curve
    efficiency = interpolate_efficiency(power, converter)

    # Calculate adjusted power based on flow direction
    adjusted_power = power * efficiency if power < 0 else power / efficiency
    adjusted_power = adjusted_power + converter['stand_by_loss'].iloc[0]
    # Calculate power loss
    power_loss = abs(power - adjusted_power)
    return adjusted_power, efficiency, power_loss


def interpolate_efficiency(power: float, converter: pd.Series) -> float:
    """
    Interpolates the efficiency of the converter based on the power flow.

    Args:
        power (float): Power flow through the converter in MW.
        converter (pd.Series): Converter data series.

    Returns:
        float: The interpolated efficiency.
    """
    return np.interp(
        abs(power),
        converter.efficiency.values[0][:, 0] / 1000,  # Convert kW to MW
        converter.efficiency.values[0][:, 1]
    )


def add_upstream_ext_grids(network_dict: Dict, network_id: int, tmp_net: pp.pandapowerNet, net) -> None:
    """
    Adds external grids for upstream connections in the subnetwork.

    Args:
        network_dict (Dict): Dictionary containing all subnetwork data.
        network_id (int): ID of the subnetwork to process.
        tmp_net (pp.pandapowerNet): The temporary network to modify.
    """
    for upstream in network_dict[network_id]['direct_upstream_network']:
        bus = [x[1] for x in network_dict[upstream[0]]['direct_downstream_network']
               if x[0] == network_id][0]
        v = 1.0
        if ((not network_dict[upstream[0]]['network'].res_bus.empty) and
                (net.converter.loc[net.converter['name'] == upstream[2], 'type'].str.contains('PDU').values[0])):
            v = max(0.98, network_dict[upstream[0]]['network'].res_bus.loc[upstream[1], 'vm_pu'])
            v = min(1.02, v)
            
        if len(tmp_net.ext_grid.loc[tmp_net.ext_grid['name'] == 'Converter emulation']) == 0:
            pp.create_ext_grid(tmp_net, bus=bus, vm_pu=v, name='Converter emulation')
        else:
            tmp_net.ext_grid.loc[tmp_net.ext_grid['name'] == 'Converter emulation', 'vm_pu'] = v


def update_upstream_network(network_dict: Dict, network_id: int, tmp_net: pp.pandapowerNet, net: pp.pandapowerNet) -> None:
    """
    Updates the upstream network with the results of the current subnetwork.

    Args:
        network_dict (Dict): Dictionary containing all subnetwork data.
        network_id (int): ID of the subnetwork to process.
        tmp_net (pp.pandapowerNet): The temporary network with results.
        net (pp.pandapowerNet): The main network containing converter data.
    """
    for upstream in network_dict[network_id]['direct_upstream_network']:
        up_net = network_dict[upstream[0]]['network']
        power = tmp_net.res_ext_grid.p_mw.values[0]

        # Get converter data and calculate power flow
        converter = net.converter.loc[net.converter.name == upstream[2]]
        adjusted_power, _, power_loss = calculate_converter_power(power, converter)
        # Add load to upstream network
        add_load_to_upstream(up_net, upstream[1], adjusted_power, network_id)

        # Update network and results
        network_dict[upstream[0]]['network'] = up_net
        update_converter_results(net, upstream[2], adjusted_power, power, power_loss)


def add_load_to_upstream(up_net: pp.pandapowerNet, bus: int, adjusted_power: float, network_id: int) -> None:
    """
    Adds a load to the upstream network.

    Args:
        up_net (pp.pandapowerNet): The upstream network to modify.
        bus (int): The bus where the load will be added.
        adjusted_power (float): The power value for the load.
        network_id (int): The ID of the subnetwork.
    """
    if len(up_net.load.loc[up_net.load['name'] == f'Load of net {network_id}']) != 0:
        up_net.load.loc[up_net.load['name'] == f'Load of net {network_id}', 'p_mw'] = adjusted_power
    else:
        pp.create_load(
            up_net,
            bus=bus,
            p_mw=adjusted_power,
            q_mvar=0,
            name=f'Load of net {network_id}'
        )


def update_converter_results(net: pp.pandapowerNet, converter_name: str, adjusted_power: float,
                             power: float, power_loss: float) -> None:
    """
    Updates the converter results in the main network.

    Args:
        net (pp.pandapowerNet): The main network.
        converter_name (str): The name of the converter.
        adjusted_power (float): The adjusted power value.
        power (float): The original power value.
        power_loss (float): The power loss.
    """
    net.res_converter.loc[net.converter.name == converter_name, 'p_mw'] = adjusted_power
    net.res_converter.loc[net.converter.name == converter_name, 'loading (%)'] = (
            power / net.converter.loc[net.converter.name == converter_name, 'P'] * 100
    )
    net.res_converter.loc[net.converter.name == converter_name, 'pl_mw'] = power_loss


def process_subnetwork(network_id: int, network_dict: Dict, loadflowed_subs: List[int],
                       net: pp.pandapowerNet, node_data) -> None:
    """
    Processes a single subnetwork in the DC load flow calculation.

    Args:
        network_id (int): ID of the subnetwork to process.
        network_dict (Dict): Dictionary containing all subnetwork data.
        loadflowed_subs (List[int]): List of already processed subnetworks.
        net (pp.pandapowerNet): Main network containing converter data.
        node_data: node data
    """
    tmp_net = network_dict[network_id]['network']

    # Add external grids for upstream connections
    add_upstream_ext_grids(network_dict, network_id, tmp_net, net)

    # Run power flow
    pp.runpp(tmp_net)

    # Adjust results if network contains 1 bus
    if len(tmp_net.bus) == 1:
        bus_idx = tmp_net.bus.index.values[0]
        matching_row = node_data.loc[node_data['Node number'] == bus_idx]
        matching_row['Component type'] = matching_row['Component type'].str.replace(' ', '').str.lower()
        component_type = matching_row['Component type'].values[0]
        if component_type == 'acgrid':
            node_idx = tmp_net.ext_grid.index.values[0]
            tmp_net.res_bus.loc[bus_idx, 'p_mw'] = tmp_net.res_ext_grid.loc[node_idx, 'p_mw']
        elif component_type in ['dcload', 'acload']:
            node_idx = tmp_net.load.index.values[0]
            tmp_net.res_bus.loc[bus_idx, 'p_mw'] = tmp_net.res_load.loc[node_idx, 'p_mw']
        elif component_type == 'ev':
            node_idx = tmp_net.storage.index.values[0]
            tmp_net.res_bus.loc[bus_idx, 'p_mw'] = tmp_net.res_storage.loc[node_idx, 'p_mw']
        elif component_type == 'storage':
            node_idx = tmp_net.storage.index.values[0]
            tmp_net.res_bus.loc[bus_idx, 'p_mw'] = tmp_net.res_storage.loc[node_idx, 'p_mw']
        elif component_type == 'pv':
            node_idx = tmp_net.sgen.index.values[0]
            tmp_net.res_bus.loc[bus_idx, 'p_mw'] = tmp_net.res_sgen.loc[node_idx, 'p_mw']
    # print(tmp_net.ext_grid)

    network_dict[network_id]['network'] = tmp_net
    loadflowed_subs.append(network_id)

    # Process upstream networks
    update_upstream_network(network_dict, network_id, tmp_net, net)


def perform_dc_load_flow(net: pp.pandapowerNet, use_case: dict, node_data, PDU_droop_control=False) -> pp.pandapowerNet:
    """
    Performs DC load flow calculation on the network.

    Args:
        net (pp.pandapowerNet): The network to analyze.
        use_case
        PDU_droop_control
        node_data

    Returns:
        pp.pandapowerNet: The network with load flow results.
    """
    # Separate and sort subnetworks
    subnetwork_list = separate_subnetworks(net)
    network_dict = sorting_network(net, subnetwork_list)
    err = 1
    prev_diff_volt = [0]*len(net.converter.loc[net.converter['type'].str.contains('PDU')])
    diff_volt = [10]*len(net.converter.loc[net.converter['type'].str.contains('PDU')])
    i = 0
    if PDU_droop_control:
        while ((err > 1e-8) and (sum(abs(np.array(prev_diff_volt)-np.array(diff_volt))) > 1e-8)) and (i < 100):
            i += 1
            # Initialize results
            loadflowed_subs = []
            initialize_converter_results(net)
            # Process subnetworks sequentially
            process_all_subnetworks(network_dict, loadflowed_subs, net, node_data)
            err = 0
            prev_diff_volt = diff_volt
            diff_volt = []
            for network_id in network_dict.keys():
                for upstream in network_dict[network_id]['direct_upstream_network']:
                    bus = [x[1] for x in network_dict[upstream[0]]['direct_downstream_network']
                           if x[0] == network_id][0]
                    if net.converter.loc[net.converter['name'] == upstream[2], 'type'].str.contains('PDU').values[0]:
                        v_upstream = network_dict[upstream[0]]['network'].res_bus.loc[upstream[1], 'vm_pu']
                        v_downstream = network_dict[network_id]['network'].res_bus.loc[bus, 'vm_pu']
                        err += abs(v_upstream-v_downstream)
                        diff_volt.append(v_downstream)
    else:
        loadflowed_subs = []
        initialize_converter_results(net)
        # Process subnetworks sequentially
        process_all_subnetworks(network_dict, loadflowed_subs, net, node_data)

    # Merge results and clean network
    net_res = merge_networks([network_dict[n]['network'] for n in network_dict.keys()])
    net = clean_network(net_res, net)

    _, max_v = define_voltage_limits(use_case)
    check_high_voltage_nodes(net, voltage_threshold=max_v)

    return net


def initialize_converter_results(net: pp.pandapowerNet) -> None:
    """
    Initializes the converter results DataFrame.

    Args:
        net (pp.pandapowerNet): The network to initialize.
    """
    net.res_converter = pd.DataFrame(
        data=np.empty((len(net.converter), 3)),
        columns=["p_mw", "loading (%)", 'pl_mw']
    )


def process_all_subnetworks(network_dict: Dict, loadflowed_subs: List[int], net: pp.pandapowerNet, node_data) -> None:
    """
    Processes all subnetworks in the network.

    Args:
        network_dict (Dict): Dictionary containing all subnetwork data.
        loadflowed_subs (List[int]): List of already processed subnetworks.
        net (pp.pandapowerNet): The main network.
        node_data: node data
    """
    while not all(elem in loadflowed_subs for elem in network_dict.keys()):
        unprocessed = set(network_dict.keys()) - set(loadflowed_subs)

        for network_id in unprocessed:
            if all_downstream_processed(network_dict, network_id, loadflowed_subs):
                process_subnetwork(network_id, network_dict, loadflowed_subs, net, node_data)


def all_downstream_processed(network_dict: Dict, network_id: int, loadflowed_subs: List[int]) -> bool:
    """
    Checks if all downstream networks of a given network are processed.

    Args:
        network_dict (Dict): Dictionary containing all subnetwork data.
        network_id (int): ID of the subnetwork to check.
        loadflowed_subs (List[int]): List of already processed subnetworks.

    Returns:
        bool: True if all downstream networks are processed, False otherwise.
    """
    return all(
        elem in loadflowed_subs
        for elem in [x[0] for x in network_dict[network_id]['direct_downstream_network']]
    )


def perform_load_flow_with_sizing(net: pp.pandapowerNet, cable_catalogue: pd.DataFrame, use_case: Dict,
                                  node_data) -> pp.pandapowerNet:
    """
    Performs DC load flow calculation with sizing adjustments for converters and cables.

    Args:
        net (pp.pandapowerNet): The pandapower network.
        cable_catalogue (pd.DataFrame): DataFrame containing cable specifications.
        use_case (Dict): Dictionary specifying project details and constraints.
        node_data: node data

    Returns:
        pp.pandapowerNet: The updated network after DC load flow and sizing adjustments.
    """
    # Step 1: Define voltage limits based on the use case
    min_v, max_v = define_voltage_limits(use_case)
    cable_factor, AC_DC_factor, converter_factor = define_sizing_security_factor(use_case)

    # Step 2: Perform initial DC load flow analysis
    net = perform_dc_load_flow(net, use_case, node_data, PDU_droop_control=False)

    # Step 3: Adjust converter sizing based on load flow results
    adjust_converter_sizing(net, AC_DC_factor, converter_factor)

    # Step 4: Process subnetworks and perform cable adjustments
    net = process_subnetworks_with_cable_sizing(net, cable_catalogue, min_v, max_v, cable_factor, node_data)

    _, max_v = define_voltage_limits(use_case)
    check_high_voltage_nodes(net, voltage_threshold=max_v)

    return net


def define_voltage_limits(use_case: Dict) -> tuple:
    """
    Defines the voltage limits based on the use case.

    Args:
        use_case (Dict): Dictionary specifying project details and constraints.

    Returns:
        tuple: A tuple containing the minimum and maximum voltage limits.
    """
    if use_case['Project details']['Ecosystem'] in ['CurrentOS']:
        return 0.98, 1.02
    elif use_case['Project details']['Ecosystem'] in ['ODCA',]:
        return 0.95, 1.05
    

def define_sizing_security_factor(use_case: Dict) -> tuple:
    """
    Defines the sizing security factot based on the use case.

    Args:
        use_case (Dict): Dictionary specifying project details and constraints.

    Returns:
        tuple: A tuple containing the minimum and maximum voltage limits.
    """
    cable_factor = use_case['Sizing factor']['Cable sizing security factor (%)']
    AC_DC_factor = use_case['Sizing factor']['AC/DC converter sizing security factor (%)']
    converter_factor = use_case['Sizing factor']['Other converters sizing security factor (%)']
    return cable_factor, AC_DC_factor, converter_factor


def process_subnetworks_with_cable_sizing(net: pp.pandapowerNet, cable_catalogue: pd.DataFrame, min_v: float,
                                          max_v: float, cable_factor: int, node_data) -> pp.pandapowerNet:
    """
    Processes all subnetworks and adjusts cable sizing based on load flow results.

    Args:
        net (pp.pandapowerNet): The pandapower network.
        cable_catalogue (pd.DataFrame): DataFrame containing cable specifications.
        min_v (float): Minimum voltage limit.
        max_v (float): Maximum voltage limit.
        node_data: node data
    """
    # Separate the network into subnetworks
    subnetwork_list = separate_subnetworks(net)
    dic_of_subs = sorting_network(net, subnetwork_list)

    # Initialize list of processed subnetworks
    loadflowed_subs = []

    # Process subnetworks sequentially
    while not all(sub in loadflowed_subs for sub in dic_of_subs.keys()):
        for n in set(dic_of_subs.keys()) - set(loadflowed_subs):
            if all_downstream_processed(dic_of_subs, n, loadflowed_subs):
                process_single_subnetwork_with_cable_sizing(dic_of_subs, n, cable_catalogue, min_v, max_v,
                                                            loadflowed_subs, net, cable_factor, node_data)
    
    net_res = merge_networks([dic_of_subs[n]['network'] for n in dic_of_subs.keys()])
    return clean_network(net_res, net)
    

def process_single_subnetwork_with_cable_sizing(dic_of_subs: Dict, n: int, cable_catalogue: pd.DataFrame, min_v: float,
                                                max_v: float, loadflowed_subs: List[int], net: pp.pandapowerNet,
                                                cable_factor: int, node_data) -> None:
    """
    Processes a single subnetwork and adjusts cable sizing.

    Args:
        dic_of_subs (Dict): Dictionary containing all subnetwork data.
        n (int): ID of the subnetwork to process.
        cable_catalogue (pd.DataFrame): DataFrame containing cable specifications.
        min_v (float): Minimum voltage limit.
        max_v (float): Maximum voltage limit.
        loadflowed_subs (List[int]): List of already processed subnetworks.
        net (pp.pandapowerNet): The main network.
        node_data: node data
    """
    tmp_net = dic_of_subs[n]['network']

    # Add external grids for upstream connections
    add_upstream_ext_grids_for_sizing(dic_of_subs, n, tmp_net)

    # Run a power flow
    pp.runpp(tmp_net)

    # Adjust results if network contains 1 bus
    if len(tmp_net.bus) == 1:
        bus_idx = tmp_net.bus.index.values[0]
        matching_row = node_data.loc[node_data['Node number'] == bus_idx]
        matching_row['Component type'] = matching_row['Component type'].str.replace(' ', '').str.lower()
        component_type = matching_row['Component type'].values[0]
        if component_type == 'acgrid':
            node_idx = tmp_net.ext_grid.index.values[0]
            tmp_net.res_bus.loc[bus_idx, 'p_mw'] = tmp_net.res_ext_grid.loc[node_idx, 'p_mw']
        elif component_type in ['dcload', 'acload']:
            node_idx = tmp_net.load.index.values[0]
            tmp_net.res_bus.loc[bus_idx, 'p_mw'] = tmp_net.res_load.loc[node_idx, 'p_mw']
        elif component_type == 'ev':
            node_idx = tmp_net.storage.index.values[0]
            tmp_net.res_bus.loc[bus_idx, 'p_mw'] = tmp_net.res_storage.loc[node_idx, 'p_mw']
        elif component_type == 'storage':
            node_idx = tmp_net.storage.index.values[0]
            tmp_net.res_bus.loc[bus_idx, 'p_mw'] = tmp_net.res_storage.loc[node_idx, 'p_mw']

    # adjust cable sizing
    adjust_cable_sizing(tmp_net, cable_catalogue, min_v, max_v, cable_factor)
    # Update the subnetwork in the dictionary
    dic_of_subs[n]['network'] = tmp_net
    loadflowed_subs.append(n)

    # Update upstream networks with results
    update_upstream_networks_with_results(dic_of_subs, n, tmp_net, net)


def add_upstream_ext_grids_for_sizing(dic_of_subs: Dict, n: int, tmp_net: pp.pandapowerNet) -> None:
    """
    Adds external grids for upstream connections in the subnetwork.

    Args:
        dic_of_subs (Dict): Dictionary containing all subnetwork data.
        n (int): ID of the subnetwork to process.
        tmp_net (pp.pandapowerNet): The temporary network to modify.
    """
    for upstream in dic_of_subs[n]['direct_upstream_network']:
        bus = [x[1] for x in dic_of_subs[upstream[0]]['direct_downstream_network'] if x[0] == n][0]
        pp.create_ext_grid(tmp_net, bus=bus, vm_pu=1.0, name='Converter emulation')


def update_upstream_networks_with_results(dic_of_subs: Dict, n: int, tmp_net: pp.pandapowerNet, net: pp.pandapowerNet) -> None:
    """
    Updates the upstream networks with the results of the current subnetwork.

    Args:
        dic_of_subs (Dict): Dictionary containing all subnetwork data.
        n (int): ID of the subnetwork to process.
        tmp_net (pp.pandapowerNet): The temporary network with results.
        net (pp.pandapowerNet): The main network.
    """
    for upstream in dic_of_subs[n]['direct_upstream_network']:
        up_net = dic_of_subs[upstream[0]]['network']
        power = tmp_net.res_ext_grid.p_mw.values[0]

        # Get converter data and calculate power flow
        converter = net.converter.loc[net.converter.name == upstream[2]]
        efficiency = interpolate_efficiency(power, converter)
        adjusted_power = power * efficiency if power < 0 else power / efficiency

        # Add load to upstream network
        add_load_to_upstream(up_net, upstream[1], adjusted_power, n)

        # Update network and results
        dic_of_subs[upstream[0]]['network'] = up_net
        update_converter_results(net, upstream[2], adjusted_power, power, power - adjusted_power)


def adjust_converter_sizing(net: pp.pandapowerNet, AC_DC_factor: int, converter_factor: int) -> None:
    """
    Adjusts the sizing of converters based on the load flow results.

    Args:
        net (pp.pandapowerNet): The network to adjust.
    """
    for i in net.converter.index:
        if net.converter.loc[i, 'conv_rank'] is not None:
            tmp_cc = net.converter.loc[i, 'converter_catalogue']
            new_c, new_conv_rank = select_converter_based_on_power(tmp_cc, net.res_converter.loc[i, 'p_mw'],
                                                                   AC_DC_factor, converter_factor)
            update_converter_efficiency_curve(net, i, new_c)
            update_converter_attributes(net, i, new_c, int(new_conv_rank))


def select_converter_based_on_power(tmp_cc: pd.DataFrame, power_mw: float, AC_DC_factor: int, converter_factor: int) -> tuple:
    """
    Selects the appropriate converter based on the power requirements.

    Args:
        tmp_cc (pd.DataFrame): The converter catalogue.
        power_mw (float): The power flow through the converter in MW.

    Returns:
        pd.Series: The selected converter.
    """
    if 'AC/DC' in tmp_cc.loc[0, 'Converter type']:
        factor = AC_DC_factor
    else:
        factor = converter_factor
    if (tmp_cc['Nominal power (kW)']*(1-factor/100) > abs(power_mw * 1000)).values.any():
        # Find new converter with minimum capacity above required power
        filtered_tmp_cc = tmp_cc[tmp_cc['Nominal power (kW)']*(1-factor/100) > abs(power_mw * 1000)]
        return (filtered_tmp_cc.loc[filtered_tmp_cc['Nominal power (kW)'].idxmin()],
                filtered_tmp_cc['Nominal power (kW)'].idxmin())
    else:
        # Otherwise, select the largest converter
        return tmp_cc.loc[tmp_cc['Nominal power (kW)'].idxmax()], tmp_cc['Nominal power (kW)'].idxmax()


def update_converter_efficiency_curve(net: pp.pandapowerNet, i: int, new_c: pd.Series) -> None:
    """
    Updates the efficiency curve of the converter.

    Args:
        net (pp.pandapowerNet): The network.
        i (int): The index of the converter.
        new_c (pd.Series): The selected converter.
    """
    if net.converter.loc[i, 'efficiency'] == 'user-defined':
        eff = net.converter.loc[i, 'efficiency']
        p_previous = eff[:, 0] * 100 / net.converter.loc[i, 'P']
        p = p_previous / 100 * new_c['Nominal power (kW)']
        efficiency = np.vstack((p, eff[:, 1])).T
    else:
        eff_str = new_c['Efficiency curve [Xi;Yi], i={1,2,3,4}, \nwith X= Factor of Nominal Power (%), Y=Efficiency (%)']
        eff = np.array(literal_eval('[' + eff_str.replace(';', ',') + ']'))
        e = eff[:, 1].astype('float') / 100
        p = eff[:, 0] / 100 * new_c['Nominal power (kW)']
        efficiency = np.vstack((p, e)).T

    net.converter.at[i, 'efficiency'] = efficiency


def update_converter_attributes(net: pp.pandapowerNet, i: int, new_c: pd.Series, new_conv_rank: int) -> None:
    """
    Updates the attributes of the converter.

    Args:
        net (pp.pandapowerNet): The network.
        i (int): The index of the converter.
        new_c (pd.Series): The selected converter.
    """
    net.converter.loc[i, 'conv_rank'] = new_conv_rank
    net.converter.loc[i, 'P'] = new_c['Nominal power (kW)'] / 1000


def adjust_cable_sizing(subnet: pp.pandapowerNet, cable_catalogue: pd.DataFrame, min_v: float, max_v: float,
                        cable_factor: int) -> None:
    """
    Adjusts the sizing of cables based on the load flow results.
    Adjusts from upstream to downstream

    Args:
        subnet (pp.pandapowerNet): The subnetwork to adjust.
        cable_catalogue (pd.DataFrame): DataFrame containing cable specifications.
        min_v (float): Minimum voltage limit.
        max_v (float): Maximum voltage limit.
    """

    # Compute distances of buses from the slack bus
    bus_distances = get_bus_distances(subnet)

    # Assign a distance to each line (minimum of its two connected buses)
    subnet.line["distance"] = subnet.line.apply(lambda row: min(bus_distances.get(row.from_bus, np.inf),
                                                                bus_distances.get(row.to_bus, np.inf)), axis=1)

    # Sort lines by distance (ascending) to process from upstream (closer to slack) to downstream
    for line_id in subnet.line.sort_values(by="distance", ascending=True).index:
        if not np.isnan(subnet.line.loc[line_id, 'cable_rank']):
            optimal = False
            while not optimal:
                optimal = adjust_single_cable(subnet, line_id, cable_catalogue, min_v, cable_factor)

    '''for line_id in subnet.res_line.i_ka.sort_values(ascending=False).index:
        if not np.isnan(subnet.line.loc[line_id, 'cable_rank']):
            optimal = False
            while not optimal:
                optimal = adjust_single_cable(subnet, line_id, cable_catalogue, min_v, cable_factor)'''


def adjust_single_cable(subnet: pp.pandapowerNet, line_id: int, cable_catalogue: pd.DataFrame, min_v: float,
                        cable_factor: int) -> bool:
    """
    Adjusts the sizing of a single cable.

    Args:
        subnet (pp.pandapowerNet): The subnetwork.
        line_id (int): The ID of the cable to adjust.
        cable_catalogue (pd.DataFrame): DataFrame containing cable specifications.
        min_v (float): Minimum voltage limit.

    Returns:
        bool: True if the cable is optimally sized, False otherwise.
    """
    tension_of_interest = 'vm_to_pu' if subnet.res_line.loc[line_id, 'p_from_mw'] > 0 else 'vm_from_pu'
    idx_new_cable = subnet.line.loc[line_id, "cable_rank"]
    idx_cable = idx_new_cable
    load_flow_converge = True
    # Perform iterative cable resizing
    while (subnet.res_line.loc[line_id, "loading_percent"] < (100 * (1-cable_factor/100)) and
           subnet.res_line.loc[line_id, tension_of_interest] > min_v and
           idx_new_cable >= 1 and load_flow_converge):
        idx_cable = idx_new_cable
        idx_new_cable -= 1
        new_cable = cable_catalogue.loc[idx_new_cable]
        subnet.line.r_ohm_per_km.loc[line_id] = new_cable['R'] * 1000
        subnet.line.max_i_ka.loc[line_id] = new_cable['Imax'] / 1000
        subnet.line.cable_rank.loc[line_id] = idx_new_cable
        subnet.line.section.loc[line_id] = new_cable['section']
        try:
            pp.runpp(subnet)
        except:
            print("Load flow convergence issue.")
            load_flow_converge = False
    # Restore cable to previous size if necessary
    if not (subnet.res_line.loc[line_id, "loading_percent"] < (100 * (1-cable_factor/100)) and
            subnet.res_line.loc[line_id, tension_of_interest] > min_v and
            load_flow_converge):
        new_cable = cable_catalogue.loc[idx_cable]
        subnet.line.r_ohm_per_km.loc[line_id] = new_cable['R'] * 1000
        subnet.line.max_i_ka.loc[line_id] = new_cable['Imax'] / 1000
        subnet.line.cable_rank.loc[line_id] = idx_cable
        subnet.line.section.loc[line_id] = new_cable['section']
    pp.runpp(subnet)

    # Check for downstream constraints
    return check_downstream_line_size(subnet, line_id, cable_catalogue)


def check_downstream_line_size(subnet: pp.pandapowerNet, line_id: int, cable_catalogue: pd.DataFrame) -> bool:
    """
    Checks for downstream constraints and adjusts cable sizes if necessary.

    Args:
        subnet (pp.pandapowerNet): The subnetwork.
        line_id (int): The ID of the cable to check.
        cable_catalogue (pd.DataFrame): DataFrame containing cable specifications.

    Returns:
        bool: True if no further adjustments are needed, False otherwise.
    """
    optimal = True
    for downstream_line in find_lines_between_given_line_and_ext_grid(subnet, line_id):
        if subnet.line.loc[downstream_line, 'cable_rank'] < subnet.line.loc[line_id, 'cable_rank']:
            new_cable = cable_catalogue.loc[subnet.line.loc[downstream_line, 'cable_rank'] + 1]
            subnet.line.r_ohm_per_km.loc[downstream_line] = new_cable['R'] * 1000
            subnet.line.max_i_ka.loc[downstream_line] = new_cable['Imax'] / 1000
            subnet.line.cable_rank.loc[downstream_line] += 1
            subnet.line.section.loc[downstream_line] = new_cable['section']
            optimal = False
    pp.runpp(subnet)
    return optimal


def check_high_voltage_nodes(net, voltage_threshold=1.1):
    """
    Checks for high voltage buses not connected to static generators via transformers.
    
    Args:
        net (pp.Network): Pandapower network to analyze
        voltage_threshold (float): Voltage threshold in per unit (default: 1.1)
    
    Raises:
        UserWarning: When high voltage buses are found without generator connections
    """

    # Identify over voltage buses
    high_voltage_buses = net.res_bus[net.res_bus.vm_pu > voltage_threshold].index
    
    for bus in high_voltage_buses:
        # Find connected transformers
        connected_converter = net.converter[(net.converter.from_bus == bus) | (net.converter.to_bus == bus)]
        
        sgen_connected = False
        # Check for generators on transformer secondary side
        for _, conv in connected_converter.iterrows():
            other_side_bus = conv.from_bus if conv.to_bus == bus else conv.to_bus
            
            if other_side_bus in net.sgen.bus.values:
                sgen_connected = True
                break
                
        # Raise warning if no generator connection
        if not sgen_connected:
            bus_voltage = net.res_bus.at[bus, 'vm_pu']
            warnings.warn(
                "******WARNING******\n"
                f"High voltage alert! Bus {bus} (voltage = {bus_voltage:.3f} pu)\n "
                f"has no static generator connection through transformers.\n"
                "******WARNING******\n",
                category=UserWarning,
                stacklevel=2
            )


def update_network(net, t):
    """
    Updates the power values of loads, generators (sgen), and storage units in the pandapower network
    based on their predefined power profiles at a given time step t.

    Parameters:
    net (pandapowerNet): The pandapower network object containing load, sgen, and storage elements.
    t (int): The current time step index.

    The function adjusts the power values (p_mw) based on the predefined power profile (`power_profile`).
    If t exceeds the profile length, a warning is displayed, and the last available value is used.
    """

    for i, _ in net.load.iterrows():
        if not np.isnan(net.load.loc[i, 'power_profile']).any():
            if t < len(net.load.loc[i, 'power_profile']):
                net.load.loc[i, 'p_mw'] = net.load.loc[i, 'power_profile'][t]*net.load.loc[i, 'p_nom_mw']
            else:
                print('WARNING INCORRECT Number of timestep in User-define profile')
                net.load.loc[i, 'p_mw'] = net.load.loc[i, 'power_profile'][-1]*net.load.loc[i, 'p_nom_mw']
    
    for i, _ in net.sgen.iterrows():
        if not np.isnan(net.sgen.loc[i, 'power_profile']).any():
            if t < len(net.sgen.loc[i, 'power_profile']):
                net.sgen.loc[i, 'p_mw'] = net.sgen.loc[i, 'power_profile'][t]*net.sgen.loc[i, 'p_nom_mw']
            else:
                print('WARNING INCORRECT Number of timestep in User-define profile')
                net.sgen.loc[i, 'p_mw'] = net.sgen.loc[i, 'power_profile'][-1]*net.sgen.loc[i, 'p_nom_mw']

    for i, _ in net.storage.iterrows():
        if not np.isnan(net.storage.loc[i, 'power_profile']).any():
            if t < len(net.storage.loc[i, 'power_profile']):
                net.storage.loc[i, 'p_mw'] = net.storage.loc[i, 'power_profile'][t]*net.storage.loc[i, 'p_nom_mw']
            else:
                print('WARNING INCORRECT Number of timestep in User-define profile')
                net.storage.loc[i, 'p_mw'] = net.storage.loc[i, 'power_profile'][-1]*net.storage.loc[i, 'p_nom_mw']


def format_result_dataframe(df, net):
    for i, row in net.bus.iterrows():
        df[f'node {i} : v_pu'] = np.nan
    for i, row in net.line.iterrows():
        df[f'line {row.from_bus} - {row.to_bus} : i_ka'] = np.nan
        df[f'line {row.from_bus} - {row.to_bus} : loading'] = np.nan
        df[f'line {row.from_bus} - {row.to_bus} : pl_mw'] = np.nan
    for i, row in net.load.iterrows():
        df[f'load {row["name"]} : p_mw'] = np.nan
    for i, row in net.sgen.iterrows():
        df[f'sgen {row["name"]} : p_mw'] = np.nan
    for i, row in net.storage.iterrows():
        df[f'storage {row["name"]} : p_mw'] = np.nan
        if "Battery" in row["name"]:
            df[f'storage {row["name"]} : SOC'] = np.nan
    for i, row in net.converter.iterrows():
        df[f'{row["name"]} : p_mw'] = np.nan
        df[f'{row["name"]} : loading'] = np.nan
        df[f'{row["name"]} : pl_mw'] = np.nan
    return df


def fill_result_dataframe(df, t, net):
    for i, row in net.bus.iterrows():
        df.loc[t, f'node {i}: v_pu'] = net.res_bus.loc[i, 'vm_pu']
    for i, row in net.line.iterrows():
        df.loc[t, f'line {row.from_bus} - {row.to_bus}: i_ka'] = net.res_line.loc[i, 'i_ka']
        df.loc[t, f'line {row.from_bus} - {row.to_bus}: loading'] = net.res_line.loc[i, 'loading_percent']
        df.loc[t, f'line {row.from_bus} - {row.to_bus}: pl_mw'] = net.res_line.loc[i, 'pl_mw']
    for i, row in net.load.iterrows():
        df.loc[t, f'load {row["name"]}: p_mw'] = net.res_load.loc[i, 'p_mw']
    for i, row in net.sgen.iterrows():
        df.loc[t, f'sgen {row["name"]}: p_mw'] = net.res_sgen.loc[i, 'p_mw']
    for i, row in net.storage.iterrows():
        df.loc[t, f'storage {row["name"]}: p_mw'] = net.res_storage.loc[i, 'p_mw']
        if "Battery" in row["name"]:
            df.loc[t, f'storage {row["name"]}: SOC'] = net.storage.loc[i, 'soc_percent']
    for i, row in net.converter.iterrows():
        df.loc[t, f'{row["name"]}: p_mw'] = net.res_converter.loc[i, 'p_mw']
        df.loc[t, f'{row["name"]}: loading'] = net.res_converter.loc[i, 'loading (%)']
        df.loc[t, f'{row["name"]}: pl_mw'] = net.res_converter.loc[i, 'pl_mw']
    return df


def perform_timestep_dc_load_flow(net, use_case, node_data):
    """
    Performs a time-step DC load flow simulation for the given network over a defined period.

    Parameters:
    net (pandapowerNet): The pandapower network object.
    use_case (dict): A dictionary containing simulation parameters such as timestep and duration.
    node_data (DataFrame): Data related to network nodes, possibly containing component details.

    Returns:
    net_snapshots (list): A list of network snapshots at each time step.
    results (DataFrame): A DataFrame containing simulation results for each time step.
    """
    # Retrieve simulation parameters from the use_case dictionary
    timestep = use_case['Parameters for annual simulations']['Simulation time step (mins)']
    timelaps = use_case['Parameters for annual simulations']['Simulation period (days)']

    # Compute the total number of simulation steps based on time step and duration
    number_of_timestep = int(timelaps*24*60/timestep)

    # Initialize an empty DataFrame to store results with time step index
    results = pd.DataFrame(index=range(number_of_timestep))
    # Initialize a list to store network snapshots at each time step
    net_snapshots = []

    # Loop through each time step in the simulation
    for t in tqdm(range(number_of_timestep)):
        # Update network with time step t data
        update_network(net, t)

        # Perform Load flow analysis with PDU droop control
        net = perform_dc_load_flow(net, use_case, node_data, PDU_droop_control=True)
        # Perform load flow considering droop control behavior at time t
        net = perform_dc_load_flow_with_droop(net, use_case, t, timestep/60, node_data)

        # Store network snapshot
        net_snapshots.append(net.deepcopy())
        # Store the simulation results for the current time step
        results = fill_result_dataframe(results, t, net)
    
    return net_snapshots, results

 
def perform_dc_load_flow_with_droop(net: pp.pandapowerNet, use_case: dict, t, time_step_duration, node_data) -> pp.pandapowerNet:
    """
    Performs DC load flow calculation on the network with iterative droop control adjustments.

    Args:
        net (pp.pandapowerNet): The DC network model in pandapower.
        use_case (dict): Dictionary containing simulation parameters.
        t (int): Current time step.
        time_step_duration (float): Duration of the time step in hours.
        node_data: Additional data related to network nodes.

    Returns:
        pp.pandapowerNet: The updated network with load flow results after droop correction.
    """
    # Initialize iterative process variables
    error = 1  # Initial error value
    nb_it = 0  # Iteration counter
    tol = 1e-2  # Tolerance for convergence
    max_it = 3  # Maximum number of iterations

    # Perform initial DC load flow with droop control enabled
    net = perform_dc_load_flow(net, use_case, node_data, PDU_droop_control=True)

    # Initialize bus voltage arrays for tracking convergence
    bus_voltages_previous = np.zeros(net.res_bus.vm_pu.values.shape)
    bus_voltages = np.zeros(net.res_bus.vm_pu.values.shape)

    # Iteratively adjust droop control settings until convergence or max iterations reached
    while (abs(error) > tol) and (nb_it < max_it):

        # Store previous iteration's bus voltages
        bus_voltages_previous = bus_voltages

        # Apply droop correction based on the current time step
        net, soc_list = droop_correction(net, t, time_step_duration)

        # Perform DC load flow again with updated parameters
        net = perform_dc_load_flow(net, use_case, node_data, PDU_droop_control=True)

        # Retrieve updated bus voltages
        bus_voltages = net.res_bus.vm_pu.values

        # Compute error to check convergence
        error = compute_error(bus_voltages, bus_voltages_previous, net)

        # Display iteration status
        print('nb_it : ', nb_it, ' error value : ', error)
        # Increment iteration counter
        nb_it = nb_it + 1

        # If maximum iterations reached, print warning
        if nb_it == max_it:
            print('\nout by iteration\n error : ', error)

    # Update state of charge (SOC) for batteries in storage elements
    c = 0
    for i, _ in net.storage.iterrows():
        if 'Battery' in net.storage.loc[i, 'name']:  # Check if the storage unit is a battery
            soc = soc_list[c]  # Retrieve the corresponding SOC value
            print(soc_list)  # Debugging: Print SOC list
            print(soc)  # Debugging: Print individual SOC value
            net.storage.loc[i, 'soc_percent'] = soc  # Update SOC in the network model
            c += 1  # Move to the next SOC value in the list
        
    return net


def compute_error(bus_voltages, bus_voltages_previous, net):
    """
    Computes the voltage deviation error between two consecutive iterations in the droop control adjustment process.

    Args:
        bus_voltages (numpy.ndarray): Current iteration's bus voltages in pu.
        bus_voltages_previous (numpy.ndarray): Previous iteration's bus voltages in pu.
        net (pp.pandapowerNet): The network model used for calculations.

    Returns:
        float: Maximum voltage deviation as a percentage.
    """
    # Compute the absolute relative deviation of voltages
    voltage_deviation = abs(bus_voltages - bus_voltages_previous) / bus_voltages
    # Find the maximum deviation across all buses
    max_deviation = max(voltage_deviation)

    # Debugging output (commented out but useful for analysis)
    # print(pd.DataFrame(data=np.vstack((bus_voltages,bus_voltages_previous,voltage_deviation*100)).T,index=net.res_bus.index,columns=['v_bus','prev_v_bus','dev']))
    # print('load  :' ,net.load['p_mw'])
    # print('sgen  :' ,net.sgen['p_mw'])
    print('storage  :\n', net.storage['p_mw'])

    return max_deviation * 100  # Error in percentage


def droop_correction(net, t, time_step_duration):
    """
    Adjusts the power setpoints of network elements (loads, generators, and storage)
    based on their droop control curves.

    Args:
        net (pp.pandapowerNet): The network model being analyzed.
        t (int): Current time step index.
        time_step_duration (float): Duration of each time step in hours.

    Returns:
        tuple:
            - net (pp.pandapowerNet): Updated network with new power setpoints.
            - soc_list (list): List of updated state-of-charge (SOC) values for batteries.
    """
    soc_list = []  # Store SOC values for batteries
    attributes = ['load', 'sgen', 'storage']   # Network elements subject to droop correction
    for attr in attributes:
        # Iterate over each element (row) in the corresponding dataframe (load, sgen, storage)
        for i, _ in getattr(net, attr).iterrows():
            # Get voltage of the bus where the element is connected
            voltage_bus, v_asset = get_voltage_bus(i, attr, net)

            # Retrieve the droop curve for the element
            droop_curve = get_droop_curve(i, attr, net)

            # Special case for batteries in storage elements
            if (attr == 'storage') and ('Battery' in getattr(net, attr).loc[i, 'name']):
                # Compute power adjustment and SOC based on droop characteristics
                p_droop, soc = interpolate_bess_p_droop(i, attr, net, droop_curve, t, time_step_duration, v_asset)
                soc_list.append(soc)
            else:
                # Compute power adjustment based on droop characteristics
                p_droop = interpolate_p_droop(i, attr, net, droop_curve, t, v_asset)

            # Update the power setpoint of the element
            getattr(net, attr).loc[i, 'p_mw'] = p_droop

    return net, soc_list


def get_voltage_bus(i, attr, net):
    """
    Retrieves the voltage of the bus to which a given network element (load, generator, or storage) is connected.

    If the element is directly connected to a bus, the voltage of that bus is returned.
    If the element is connected through a converter, the voltage of the opposite-side bus of the converter is returned.

    Args:
        i (int): Index of the network element.
        attr (str): Type of network element ('load', 'sgen', or 'storage').
        net (pp.pandapowerNet): The pandapower network model.

    Returns:
        tuple:
            - voltage_bus (int): The index of the bus where voltage should be retrieved.
            - v_asset (float): The per-unit voltage value of the bus.
    """
    # Retrieve the bus where the asset (load, generator, or storage) is connected
    asset_bus = getattr(net, attr).loc[i, 'bus']
    # Check if a converter is connected to this bus
    converter_connected = net.converter[(net.converter.from_bus == asset_bus) | (net.converter.to_bus == asset_bus)]

    if converter_connected.empty:
        # If no converter is connected, use the asset's direct bus
        voltage_bus = asset_bus
    else:
        # If a converter is connected, retrieve its index
        converter_id = converter_connected.index[0]
        # Identify the bus on the opposite side of the converter
        if net.converter.loc[converter_id, 'from_bus'] == asset_bus:
            voltage_bus = net.converter.loc[converter_id, 'to_bus']
        else:
            voltage_bus = net.converter.loc[converter_id, 'from_bus']

    # Retrieve the voltage (per unit) at the determined bus
    v_asset = net.res_bus.loc[voltage_bus, 'vm_pu'].item()

    return voltage_bus, v_asset


def get_droop_curve(i, attr, net):
    """
    Retrieves the droop curve for a given network element (load, generator, or storage).

    If the element is directly connected to a bus, the function checks whether the droop curve is defined for it.
    If the droop curve is missing, a default curve is assigned.
    If the element is connected through a converter, the function retrieves the droop curve from the converter.

    Args:
        i (int): Index of the network element.
        attr (str): Type of network element ('load', 'sgen', or 'storage').
        net (pp.pandapowerNet): The pandapower network model.

    Returns:
        np.ndarray: The droop curve of the element, either from the element itself or the associated converter.
    """
    # Retrieve the bus where the asset (load, generator, or storage) is connected
    asset_bus = getattr(net, attr).loc[i, 'bus']
    # Check if a converter is connected to this bus
    converter_connected = net.converter[(net.converter.from_bus == asset_bus) | (net.converter.to_bus == asset_bus)]

    if converter_connected.empty:
        # If no converter is connected, check if the element has a defined droop curve
        if np.isnan(net.load.loc[i, 'droop_curve']).any():
            # Assign a default droop curve if none is provided
            droop_curve = np.array([[1.5,1], [1.1,1], [1,1], [1,1], [0.99,1], [0.95,1]])
        else:
            # Use the droop curve defined in the load element
            droop_curve = net.load.loc[i, 'droop_curve']
    else:
        # If a converter is connected, retrieve its index
        converter_id = converter_connected.index[0]
        # Check if the converter has a defined droop curve
        if net.converter.loc[converter_id, 'droop_curve'] is None:
            # Assign a default droop curve if the converter has none
            droop_curve = np.array([[1.5,1], [1.1,1], [1,1], [1,1], [0.99,1], [0.95,1]])
        else:
            # Use the droop curve defined in the converter
            droop_curve = net.converter.loc[converter_id, 'droop_curve']

    return droop_curve


def interpolate_p_droop(i, attr, net, droop_curve, t, v_asset):
    """
    Interpolates the droop power setpoint based on the voltage at the asset's bus.

    This function uses the droop curve to determine the power adjustment for a given voltage level.
    The droop curve consists of voltage (x-axis) and corresponding power setpoints (y-axis).

    Args:
        i (int): Index of the network element.
        attr (str): Type of network element ('load', 'sgen', or 'storage').
        net (pp.pandapowerNet): The pandapower network model.
        droop_curve (np.ndarray): 2D array where the first column is voltage and the second is power factor.
        t (int): Current time step.
        v_asset (float): The voltage at the asset's bus.

    Returns:
        float: The interpolated power setpoint based on the droop curve.
    """
    # Extract power and voltage points from the droop curve
    p_droop_curve = droop_curve[:, 1]  # Power scaling factors
    v_droop_curve = droop_curve[:, 0]  # Corresponding voltage levels

    # Ensure the droop curve is sorted in ascending order of voltage
    v_droop_curve, p_droop_curve = zip(*sorted(zip(v_droop_curve, p_droop_curve)))
    # Convert tuples back to NumPy arrays
    v_droop_curve = np.array(v_droop_curve)
    p_droop_curve = np.array(p_droop_curve)

    # Compute the base power setpoint at this time step
    p_setpoint = getattr(net, attr).loc[i, 'power_profile'][t]*getattr(net, attr).loc[i, 'p_nom_mw']

    # Scale the droop power curve using the base setpoint
    p_droop_curve = p_droop_curve*p_setpoint

    # Interpolate the power setpoint for the given asset voltage
    p_droop = np.interp(v_asset, v_droop_curve, p_droop_curve)

    return p_droop


def interpolate_bess_p_droop(i, attr, net, droop_curve, t, duration, v_asset):
    p_droop_curve = droop_curve[:, 1]
    v_droop_curve = droop_curve[:, 0]
    v_droop_curve, p_droop_curve = zip(*sorted(zip(v_droop_curve, p_droop_curve)))
    v_droop_curve = np.array(v_droop_curve)
    p_droop_curve = np.array(p_droop_curve)

    p_setpoint = getattr(net, attr).loc[i, 'p_nom_mw']
    p_droop_curve = p_droop_curve*p_setpoint
    p_droop = np.interp(v_asset, v_droop_curve, p_droop_curve)

    soc_init = getattr(net, attr).loc[i, 'soc_percent'].item()
    max_energy = getattr(net, attr).loc[i, 'max_e_mwh'].item()

    # SOC computation
    soc_change = ((p_droop * duration) / max_energy) * 100  # SOC change (in %) Misses change 1 by period duration of each time step
    soc_final = soc_change + soc_init  # if is_positive else ini_SOC - SOC_change

    if soc_final <= 0:
        soc_max_var = soc_init
        p_droop = (soc_max_var / (100 * duration)) * max_energy
        soc_final = soc_init + soc_max_var
    elif soc_final > 100:
        soc_max_var = 100 - soc_init
        p_droop = (soc_max_var / (100 * duration)) * max_energy
        soc_final = soc_init + soc_max_var

    return p_droop, soc_final
