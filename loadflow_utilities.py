import pandapower as pp
import numpy as np
import pandas as pd
import copy

def ldf_DC(net):
    """
    Runs a load flow (power flow) analysis on the given DC network.

    Parameters:
    net (pandapowerNet): The pandapower network object.

    Returns:
    pandapowerNet: The network object after running the load flow analysis.
    """
    try:
        pp.runpp(net)
    except pp.LoadflowNotConverged:
        print("Power Flow did not converge")
    
    return net

def preprocess_ldf_DC(net):
    """
    Preprocesses the DC network for load flow analysis by handling converters and their efficiencies.

    Parameters:
    net (pandapowerNet): The pandapower network object.

    Returns:
    tuple: The original network and a new network with updated elements.
    """
    for idx_converter, converter in net.converter.iterrows():
        # Identify the element connected to the 'from_bus' of the converter
        if (net.load.bus.values == converter.from_bus).any():
            elm = net.load[net.load.bus.values == converter.from_bus]
        elif (net.storage.bus.values == converter.from_bus).any():
            elm = net.storage[net.storage.bus.values == converter.from_bus]
        elif (net.sgen.bus.values == converter.from_bus).any():
            elm = net.sgen[net.sgen.bus.values == converter.from_bus]
        else:
            raise ValueError("Converter unconnected")

        # Calculate efficiency of the converter
        eff = np.interp(elm.p_mw, converter.efficiency[:, 0] / 1000, converter.efficiency[:, 1])[0]
        loss = 1 - eff

        # Update the power values of loads,storage and generator based on the efficiency of the converter
        if (net.load.bus.values == converter.to_bus).any():
            net.load.loc[net.load.bus.values == converter.to_bus, 'p_mw'] = elm.p_mw.values[0] * (1 + loss)
        elif (net.storage.bus.values == converter.to_bus).any():
            if elm.p_mw.values[0] > 0:
                net.storage.loc[net.storage.bus.values == converter.to_bus, 'p_mw'] = elm.p_mw.values[0] * (1 - loss)
            else:
                net.storage.loc[net.storage.bus.values == converter.to_bus, 'p_mw'] = elm.p_mw.values[0] * (1 + loss)
        elif (net.sgen.bus.values == converter.to_bus).any():
            net.sgen.loc[net.sgen.bus.values == converter.to_bus, 'p_mw'] = elm.p_mw.values[0] * (1 - loss)

    # Create a deep copy of the network for further processing
    new_net = copy.deepcopy(net)

    # Lists to keep track of elements to be deleted
    del_load = []
    del_node = []
    del_storage = []
    del_sgen = []

    # Identify loads, storage, and generators connected to converters for deletion
    for idx_load, load in net.load.iterrows():
        if load.bus in net.converter.from_bus.values:
            del_load.append(idx_load)
            del_node.append(load.bus)

    for idx_storage, storage in net.storage.iterrows():
        if storage.bus in net.converter.from_bus.values:
            del_storage.append(idx_storage)
            del_node.append(storage.bus)

    for idx_sgen, sgen in net.sgen.iterrows():
        if sgen.bus in net.converter.from_bus.values:
            del_sgen.append(idx_sgen)
            del_node.append(sgen.bus)

    # Delete identified elements from the new network
    new_net.load.drop(del_load, inplace=True)
    new_net.storage.drop(del_storage, inplace=True)
    new_net.sgen.drop(del_sgen, inplace=True)
    new_net.bus.drop(del_node, inplace=True)

    return net, new_net

def ldf_DC_converter(net):
    """
    Runs a load flow analysis on a DC network with converters.

    Parameters:
    net (pandapowerNet): The pandapower network object.

    Returns:
    pandapowerNet: The network object after running the load flow analysis.
    """
    net, new_net = preprocess_ldf_DC(net)
    new_net = ldf_DC(new_net)
    return new_net

import math as m

def optimisation_cable_size(net, cable_catalogue):
    """
    Optimizes the cable sizes in the DC network based on load flow results.

    Parameters:
    net (pandapowerNet): The pandapower network object.
    cable_catalogue (pd.DataFrame): The cable catalogue data.

    Returns:
    pandapowerNet: The network object with optimized cable sizes.
    """
    net = ldf_DC(net)
    former_cable_ranks = [0] * net.line.cable_rank.values
    load_flow_converge = True

    # Iteratively optimize cable sizes until convergence
    while not (former_cable_ranks == net.line.cable_rank.values).all():
        former_cable_ranks = net.line.cable_rank.values.copy()
        for i in range(len(net.line)):
            idx_new_cable = (cable_catalogue['Imax'] / 1000 > net.res_line.loc[i, 'i_ka']).idxmax()
            
            if idx_new_cable > net.line.cable_rank.loc[i]:
                idx_new_cable = m.floor((idx_new_cable + net.line.cable_rank.loc[i]) / 2)
            else:
                idx_new_cable = m.ceil((idx_new_cable + net.line.cable_rank.loc[i]) / 2)
            print(idx_new_cable)
            new_cable = cable_catalogue.loc[idx_new_cable]
            net.line.r_ohm_per_km.loc[i] = new_cable['R'] * 1000
            net.line.max_i_ka.loc[i] = new_cable['Imax'] / 1000
            net.line.cable_rank.loc[i] = idx_new_cable
        net = ldf_DC(net)
        if np.isnan(net.res_line.i_ka.loc[0]):
            load_flow_converge = False
            break

    # Revert to former cable sizes if load flow does not converge
    if not load_flow_converge:
        for i in range(len(net.line)):
            new_cable = cable_catalogue.loc[former_cable_ranks[i]]
            net.line.r_ohm_per_km.loc[i] = new_cable['R'] * 1000
            net.line.max_i_ka.loc[i] = new_cable['Imax'] / 1000
            net.line.cable_rank.loc[i] = former_cable_ranks[i]
        net = ldf_DC(net)
    
    return net
