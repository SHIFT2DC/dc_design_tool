"""
DC Load Flow Module

This module provides functions for performing DC load flow calculations on electrical networks.
It handles the cleanup of network data and performs sequential load flow calculations on subnetworks.
"""


import pandapower as pp
import numpy as np
import pandas as pd
import copy
import math as m
from ast import literal_eval
from topology_utilities import separate_subnetworks, sorting_network, merge_networks,find_lines_between_given_line_and_ext_grid
from typing import Dict, List


def clean_network(net: pp.pandapowerNet, original_net: pp.pandapowerNet) -> pp.pandapowerNet:
    """
    Clean the network by removing temporary components and restoring converter data.
    
    Args:
        net: Network to clean
        original_net: Original network with converter data
        
    Returns:
        Cleaned network
    """
    # Restore converter data from original network
    net.converter = original_net.converter
    net.res_converter = original_net.res_converter
    
    # Remove temporary loads used for converter emulation
    del_load = ['Load of net' in str(x) for x in net.load.name.values]
    net.load.drop(net.load.loc[del_load].index, inplace=True)
    net.res_load.drop(net.res_load.loc[del_load].index, inplace=True)
    
    # Remove temporary external grids used for converter emulation
    del_ext_grid = ['Converter emulation' in str(x) for x in net.ext_grid.name.values]
    net.ext_grid.drop(net.ext_grid.loc[del_ext_grid].index, inplace=True)
    net.res_ext_grid.drop(net.res_ext_grid.loc[del_ext_grid].index, inplace=True)
    
    return net


def calculate_converter_power(power: float, converter: pd.Series) -> tuple:
    """
    Calculate converter power flow and losses.
    
    Args:
        power: Power flow through converter in MW
        converter: Converter data series
        
    Returns:
        Tuple of (adjusted power, efficiency, power loss)
    """
    # Get efficiency from interpolation of efficiency curve
    efficiency = np.interp(
        abs(power), 
        converter.efficiency.values[0][:, 0] / 1000,  # Convert kW to MW
        converter.efficiency.values[0][:, 1]
    )
    
    # Calculate adjusted power based on flow direction
    adjusted_power = power * efficiency if power < 0 else power / efficiency
    
    # Calculate power loss
    power_loss = power - adjusted_power
    
    return adjusted_power, efficiency, power_loss


def process_subnetwork(
    network_id: int,
    network_dict: Dict,
    loadflowed_subs: List[int],
    net: pp.pandapowerNet
) -> None:
    """
    Process a single subnetwork in the DC load flow calculation.
    
    Args:
        network_id: ID of the subnetwork to process
        network_dict: Dictionary containing all subnetwork data
        loadflowed_subs: List of already processed subnetworks
        net: Main network containing converter data
    """
    tmp_net = network_dict[network_id]['network']
    
    # Add external grids for upstream connections
    for upstream in network_dict[network_id]['direct_upstream_network']:
        bus = [x[1] for x in network_dict[upstream[0]]['direct_downstream_network'] 
               if x[0] == network_id][0]
        pp.create_ext_grid(tmp_net, bus=bus, vm_pu=1.0, name='Converter emulation')
    
    # Run power flow
    pp.runpp(tmp_net)
    network_dict[network_id]['network'] = tmp_net
    loadflowed_subs.append(network_id)
    
    # Process upstream networks
    for upstream in network_dict[network_id]['direct_upstream_network']:
        up_net = network_dict[upstream[0]]['network']
        power = tmp_net.res_ext_grid.p_mw.values[0]
        
        # Get converter data and calculate power flow
        converter = net.converter.loc[net.converter.name == upstream[2]]
        adjusted_power, _, power_loss = calculate_converter_power(power, converter)
        
        # Add load to upstream network
        pp.create_load(
            up_net,
            bus=upstream[1],
            p_mw=adjusted_power,
            q_mvar=0,
            name=f'Load of net {network_id}'
        )
        
        # Update network and results
        network_dict[upstream[0]]['network'] = up_net
        net.res_converter.loc[net.converter.name == upstream[2], 'p_mw'] = adjusted_power
        net.res_converter.loc[net.converter.name == upstream[2], 'loading (%)'] = \
            power / net.converter.loc[net.converter.name == upstream[2], 'P'] * 100
        net.res_converter.loc[net.converter.name == upstream[2], 'pl_mw'] = power_loss


def LF_DC(net: pp.pandapowerNet) -> pp.pandapowerNet:
    """
    Perform DC load flow calculation on the network.
    
    This function:
    1. Separates the network into subnetworks
    2. Processes each subnetwork sequentially based on topology
    3. Merges results back into a single network
    
    Args:
        net: Network to analyze
        
    Returns:
        Network with load flow results
    """
    # Separate and sort subnetworks
    subnetwork_list = separate_subnetworks(net)
    network_dict = sorting_network(net, subnetwork_list)
    
    # Initialize results
    loadflowed_subs = []
    net.res_converter = pd.DataFrame(
        data=np.empty((len(net.converter), 3)),
        columns=["p_mw", "loading (%)", 'pl_mw']
    )
    
    # Process subnetworks sequentially
    while not all(elem in loadflowed_subs for elem in network_dict.keys()):
        unprocessed = set(network_dict.keys()) - set(loadflowed_subs)
        
        for network_id in unprocessed:
            # Check if all downstream networks are processed
            downstream_processed = all(
                elem in loadflowed_subs 
                for elem in [x[0] for x in network_dict[network_id]['direct_downstream_network']]
            )
            
            if downstream_processed:
                process_subnetwork(network_id, network_dict, loadflowed_subs, net)
    
    # Merge results and clean network
    net_res = merge_networks([network_dict[n]['network'] for n in network_dict.keys()])
    net = clean_network(net_res, net)
    
    return net


def LF_sizing(net, cable_catalogue, use_case):
    """
    Perform DC load flow calculation with sizing adjustments for converters and cables.

    Args:
        net: The pandapower network.
        cable_catalogue: DataFrame containing cable specifications.
        use_case: Dictionary specifying project details and constraints.

    Returns:
        The updated network after DC load flow and sizing adjustments.
    """
    # Step 1: Define voltage limits based on the use case
    def define_voltage_limits(use_case):
        if use_case['Project details']['Ecosystem'] in ['ODCA', 'CurrentOS']:
            return 0.98, 1.02
        else:
            return 0.95, 1.05

    # Step 2: Adjust converter sizing
    def adjust_converter_sizing(net):
        for i in net.converter.index:
            if net.converter.loc[i, 'conv_rank'] is not None:
                tmp_cc = net.converter.loc[i, 'converter_catalogue']
                if (tmp_cc['Nominal power (kW)'] > (net.res_converter.loc[i, 'p_mw'] * 1000)).values.any():
                    # Find new converter with minimum capacity above required power
                    filtered_tmp_cc = tmp_cc[tmp_cc['Nominal power (kW)'] > abs(net.res_converter.loc[i, 'p_mw'] * 1000)]
                    new_c = filtered_tmp_cc.loc[filtered_tmp_cc['Nominal power (kW)'].idxmin()]
                else:
                    # Otherwise, select the largest converter
                    new_c = tmp_cc.loc[tmp_cc['Nominal power (kW)'].idxmax()]

                # Update converter efficiency curve
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

                # Update converter attributes
                net.converter.loc[i, 'conv_rank'] = filtered_tmp_cc['Nominal power (kW)'].idxmin()
                net.converter.at[i, 'efficiency'] = efficiency
                net.converter.loc[i, 'P'] = new_c['Nominal power (kW)'] / 1000

    # Step 3: Adjust cable sizing
    def adjust_cable_sizing(subnet, cable_catalogue, min_v, max_v):
        for line_id in subnet.res_line.i_ka.sort_values(ascending=False).index:
            optimal = False
            while not optimal:
                # Determine the correct voltage level to monitor
                tension_of_interest = 'vm_to_pu' if subnet.res_line.loc[line_id, 'p_from_mw'] > 0 else 'vm_from_pu'
                idx_new_cable = subnet.line.loc[line_id, "cable_rank"]
                idx_cable = idx_new_cable
                load_flow_converge = True

                # Perform iterative cable resizing
                while (subnet.res_line.loc[line_id, "loading_percent"] < 100 and
                       subnet.res_line.loc[line_id, tension_of_interest] > min_v and
                       idx_new_cable >= 1 and load_flow_converge):
                    idx_cable = idx_new_cable
                    idx_new_cable -= 1
                    new_cable = cable_catalogue.loc[idx_new_cable]
                    subnet.line.r_ohm_per_km.loc[line_id] = new_cable['R'] * 1000
                    subnet.line.max_i_ka.loc[line_id] = new_cable['Imax'] / 1000
                    subnet.line.cable_rank.loc[line_id] = idx_new_cable
                    try:
                        pp.runpp(subnet)
                    except:
                        print("Load flow convergence issue.")
                        load_flow_converge = False

                # Restore cable to previous size if necessary
                if not (subnet.res_line.loc[line_id, "loading_percent"] < 100 and
                        subnet.res_line.loc[line_id, tension_of_interest] > min_v and
                        load_flow_converge):
                    new_cable = cable_catalogue.loc[idx_cable]
                    subnet.line.r_ohm_per_km.loc[line_id] = new_cable['R'] * 1000
                    subnet.line.max_i_ka.loc[line_id] = new_cable['Imax'] / 1000
                    subnet.line.cable_rank.loc[line_id] = idx_cable
                pp.runpp(subnet)

                # Check for downstream constraints
                optimal = True
                for downstream_line in find_lines_between_given_line_and_ext_grid(subnet, line_id):
                    if subnet.line.loc[downstream_line, 'cable_rank'] < subnet.line.loc[line_id, 'cable_rank']:
                        new_cable = cable_catalogue.loc[subnet.line.loc[downstream_line, 'cable_rank'] + 1]
                        subnet.line.r_ohm_per_km.loc[downstream_line] = new_cable['R'] * 1000
                        subnet.line.max_i_ka.loc[downstream_line] = new_cable['Imax'] / 1000
                        subnet.line.cable_rank.loc[downstream_line] += 1
                        optimal = False
                pp.runpp(subnet)

    # Main logic of LF_sizing
    min_v, max_v = define_voltage_limits(use_case)
    net = LF_DC(net)  # Perform initial load flow analysis
    adjust_converter_sizing(net)  # Adjust converters

    # Process subnetworks and perform cable adjustments
    subnetwork_list = separate_subnetworks(net)
    dic_of_subs = sorting_network(net, subnetwork_list)
    loadflowed_subs = []

    while not all(sub in loadflowed_subs for sub in dic_of_subs.keys()):
        for n in set(dic_of_subs.keys()) - set(loadflowed_subs):
            if all(downstream_sub in loadflowed_subs for downstream_sub in
                   [x[0] for x in dic_of_subs[n]['direct_downstream_network']]):
                tmp_net = dic_of_subs[n]['network']
                for upstream in dic_of_subs[n]['direct_upstream_network']:
                    bus = [x[1] for x in dic_of_subs[upstream[0]]['direct_downstream_network'] if x[0] == n][0]
                    pp.create_ext_grid(tmp_net, bus=bus, vm_pu=1.0, name='Converter emulation')
                pp.runpp(tmp_net)
                adjust_cable_sizing(tmp_net, cable_catalogue, min_v, max_v)
                dic_of_subs[n]['network'] = tmp_net
                loadflowed_subs.append(n)

                # Update upstream networks
                for upstream in dic_of_subs[n]['direct_upstream_network']:
                    up_net = dic_of_subs[upstream[0]]['network']
                    power = tmp_net.res_ext_grid.p_mw.values[0]
                    converter = net.converter.loc[net.converter.name == upstream[2]]
                    efficiency = np.interp(abs(power), converter.efficiency.values[0][:, 0] / 1000,
                                           converter.efficiency.values[0][:, 1])
                    adjusted_power = power * efficiency if power < 0 else power / efficiency
                    pp.create_load(up_net, bus=upstream[1], p_mw=adjusted_power, q_mvar=0, name=f'Load of net {n}')
                    dic_of_subs[upstream[0]]['network'] = up_net
                    net.res_converter.loc[net.converter.name == upstream[2], 'p_mw'] = adjusted_power
                    net.res_converter.loc[net.converter.name == upstream[2], 'loading (%)'] = power / converter['P'] * 100
                    net.res_converter.loc[net.converter.name == upstream[2], 'pl_mw'] = power - adjusted_power

    net_res = merge_networks([dic_of_subs[n]['network'] for n in dic_of_subs.keys()])
    net = clean_network(net_res, net)
    return net
