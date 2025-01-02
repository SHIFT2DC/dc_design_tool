import pandas as pd
import pandapower as pp
from ast import literal_eval
import numpy as np
from read_cable_catalogue import read_cable_catalogue, process_cable_catalogue

def read_DC_csv(path):
    """
    Reads an Excel file from the specified path.

    Parameters:
    path (str): The path to the Excel file.

    Returns:
    pd.ExcelFile: The loaded Excel file.
    """
    print(">Read the Excel file")
    xl_file = pd.ExcelFile(path)
    return xl_file

def create_DC_network(xl_file, path_cable_catalogue, return_buses=False):
    """
    Creates a DC network from the provided Excel file and cable catalogue.

    Parameters:
    xl_file (pd.ExcelFile): The Excel file containing network data.
    path_cable_catalogue (str): The path to the cable catalogue file.
    return_buses (bool): Whether to return the buses dictionary.

    Returns:
    net (pandapowerNet): The created pandapower network.
    cable_catalogue (pd.DataFrame): The processed cable catalogue.
    buses (dict, optional): Dictionary of buses if return_buses is True.
    """
    # Create an empty pandapower network
    net = pp.create_empty_network()
    
    # Parse the necessary sheets from the Excel file
    line_data = xl_file.parse('lines')
    node_data = xl_file.parse('nodes')
    base_values = xl_file.parse('baseValues')

    # Read and process the cable catalogue
    cable_catalogue = read_cable_catalogue(path_cable_catalogue)
    cable_info = xl_file.parse('cable_info')
    cable_catalogue = process_cable_catalogue(cable_catalogue, cable_info)

    # Strip any leading/trailing whitespace from column names
    base_values.columns = base_values.columns.str.strip()

    print(">Define the base voltage and power from the baseValue sheet")
    # Extract base voltage and power values
    base_voltage = base_values[base_values['Variable'].str.strip() == 'Voltage']['Base_value']
    base_power = base_values[base_values['Variable'].str.strip() == 'Power']['Base_value']

    # Check if base values are present
    if base_voltage.empty or base_power.empty:
        print("Error: 'Base Value' for Voltage or Power not found in the 'baseValue' sheet.")
        raise ValueError("Base values for Voltage or Power are missing.")
    
    # Convert base voltage to kV and base power to kW
    base_voltage_kV = base_voltage.values[0] / 1000  # Convert to kV
    base_power_kW = base_power.values[0]
    
    # Create a set of all unique nodes
    nodes = set(line_data['Node_i']).union(set(line_data['Node_j'])).union(set(node_data['Node_j']))
    nodes = list(nodes)
    nodes.sort()
    
    # Create buses for each node
    buses = {}
    for node in nodes:
        buses[node] = pp.create_bus(net, vn_kv=base_voltage_kV, name=f"Bus {node}")

    # Create lines based on the line data
    for _, row in line_data.iterrows():
        if 'Rij_ohm_km' in row.keys():
            pp.create_line_from_parameters(net,
                                           from_bus=buses[row['Node_i']],
                                           to_bus=buses[row['Node_j']],
                                           length_km=row['Length_ij_km'],
                                           r_ohm_per_km=row['Rij_ohm_km'],
                                           x_ohm_per_km=row['Xij_ohm_km'],
                                           c_nf_per_km=row['Cij_ohm_km'],
                                           max_i_ka=row['Imax_ij_kA'])
        else:
            cable = cable_catalogue.iloc[-1]
            pp.create_line_from_parameters(net,
                                           from_bus=buses[row['Node_i']],
                                           to_bus=buses[row['Node_j']],
                                           length_km=row['Length_ij_km'],
                                           r_ohm_per_km=cable['R'] * 1000,
                                           x_ohm_per_km=1e-20,
                                           c_nf_per_km=0,
                                           max_i_ka=cable['Imax'] / 1000,
                                           cable_rank=len(cable_catalogue) - 1)
    
    # Create loads, storage, and generators based on node data
    for _, row in node_data.iterrows():
        if (row['Type_j'].strip() == 'P'):  # Only create loads for 'P' type nodes
            pp.create_load(net,
                           bus=buses[row['Node_j']],
                           p_mw=row['Vj_V/Pj_kW'] / 1000,  # Convert kW to MW
                           q_mvar=0)  # assuming zero reactive power
        elif (row['Type_j'].strip().lower() == 'storage'):  # storage
            pp.create_storage(net,
                              bus=buses[row['Node_j']],
                              p_mw=row['Vj_V/Pj_kW'] / 1000,  # Max active power, Convert kW to MW
                              max_e_mwh=row['E_kWh'] / 1000,  # Max energy capacity in MWh
                              soc_percent=50)  # Assuming initial state of charge at 50%
        elif (row['Type_j'].strip().lower() == 'gen'):
            pp.create_sgen(net,
                           bus=buses[row['Node_j']],
                           p_mw=row['Vj_V/Pj_kW'] / 1000)  # Active power in MW
    
    # Identify and create the slack bus (external grid connection)
    slack_nodes = node_data[node_data['Type_j'].str.strip() == 'V']['Node_j']

    if slack_nodes.empty:
        print("Error: No slack node ('V' type) found in the 'nodes' sheet.")
        raise ValueError("Slack node is missing.")
    slack_node = slack_nodes.values[0]
    pp.create_ext_grid(net, bus=buses[slack_node], vm_pu=1.0)
    
    if return_buses:
        return net, cable_catalogue, buses
    else:
        return net, cable_catalogue

def create_DC_network_with_converter(xl_file, path_cable_catalogue):
    """
    Creates a DC network with converters from the provided Excel file and cable catalogue.

    Parameters:
    xl_file (pd.ExcelFile): The Excel file containing network data.
    path_cable_catalogue (str): The path to the cable catalogue file.

    Returns:
    net (pandapowerNet): The created pandapower network with converters.
    cable_catalogue (pd.DataFrame): The processed cable catalogue.
    """
    # Create the base DC network and get the buses dictionary
    net, cable_catalogue, buses = create_DC_network(xl_file, path_cable_catalogue, return_buses=True)
    
    # Parse the converter data from the Excel file
    converter_data = xl_file.parse('converter')

    # Initialize the converter DataFrame in the network
    net.converter = pd.DataFrame(columns=converter_data.columns)
    
    # Add each converter to the network
    for _, row in converter_data.iterrows():
        new_row = {
            "name": row['name'],
            "from_bus": buses[row['from_bus']],
            "to_bus": buses[row['to_bus']],
            "type": row['type'],
            "Pmax": row['Pmax'] / 1000,  # Convert kW to MW
            "efficiency": np.array(literal_eval(row['efficiency']))
        }
        net.converter.loc[len(net.converter)] = new_row
    
    return net, cable_catalogue



#En devellopement 
def create_DC_network_with_transformer(xl_file, path_cable_catalogue):
    """
    Creates a DC network with converters from the provided Excel file and cable catalogue.

    Parameters:
    xl_file (pd.ExcelFile): The Excel file containing network data.
    path_cable_catalogue (str): The path to the cable catalogue file.

    Returns:
    net (pandapowerNet): The created pandapower network with converters.
    cable_catalogue (pd.DataFrame): The processed cable catalogue.
    """
    # Create the base DC network and get the buses dictionary
    net, cable_catalogue, buses = create_DC_network(xl_file, path_cable_catalogue, return_buses=True)
    
    # Parse the converter data from the Excel file
    converter_data = xl_file.parse('converter')

    # Initialize the converter DataFrame in the network    
    # Add each converter to the network
    for _, row in converter_data.iterrows():
        pp.create_transformer_from_parameters(net, hv_bus=buses[row['to_bus']], lv_bus=buses[row['from_bus']], sn_mva=1e8,
                                      vn_hv_kv=row['hv']/1000, vn_lv_kv=row['lv']/1000, vk_percent=1e-4,
                                      vkr_percent=0.0, pfe_kw=0, i0_percent=0.0,
                                      shift_degree=0, name="Transformer")
    
    return net, cable_catalogue
