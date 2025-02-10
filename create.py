import pandas as pd
import pandapower as pp
from ast import literal_eval
import numpy as np
from read_cable_catalogue import read_cable_catalogue, process_cable_catalogue
from reading_utilities import read_UC_Definition
from topology_utilities import separate_subnetworks, sorting_network, merge_networks


def create_DC_network(path: str, path_cable_catalogue: str, path_converter_catalogue: str) -> tuple:
    """
    Creates a DC network from the provided Excel file and cable catalogue.

    Args:
        path (str): Path to the Excel file containing network data.
        path_cable_catalogue (str): Path to the cable catalogue file.
        path_converter_catalogue (str): Path to the converter catalogue file.

    Returns:
        tuple: A tuple containing:
            - net (pandapower.Network): The created pandapower network.
            - cable_catalogue (pd.DataFrame): The processed cable catalogue.
            - Uc (dict): Dictionary containing UC definition data.
    """
    # Create an empty pandapower network
    net = pp.create_empty_network()

    # Read Excel file and UC definition
    xl_file = pd.ExcelFile(path)
    Uc = read_UC_Definition(xl_file)

    # Parse necessary sheets from the Excel file
    line_data = xl_file.parse('Lines')
    node_data = xl_file.parse('Assets Nodes')
    converter_data = xl_file.parse('Converters')
    converter_default = xl_file.parse('Default droop curves')


    # Read and process the cable catalogue
    cable_catalogue = read_cable_catalogue(path_cable_catalogue)
    cable_info = Uc['Conductor parameters']
    cable_catalogue = process_cable_catalogue(cable_catalogue, cable_info)

    # Read and filter the converter catalogue
    converter_catalogue = pd.ExcelFile(path_converter_catalogue).parse('Converters')
    converter_catalogue = converter_catalogue.loc[
        converter_catalogue['Ecosystem'] == Uc['Project details']['Ecosystem']
    ]

    # Create buses and components
    _create_buses_and_components(net, node_data, converter_default)
    
    # Create converters
    _create_converters(net, converter_data, converter_default, converter_catalogue)

    _create_lines(net, line_data, cable_catalogue)

    # Handle subnetworks and merge them
    subnetworks = separate_subnetworks(net)
    for subnetwork in subnetworks:
        _fix_zero_voltages(subnetwork)
    net = merge_networks(subnetworks)

    # Create converters
    _create_converters(net, converter_data, converter_default, converter_catalogue)

    return net, cable_catalogue, Uc


def _create_buses_and_components(net: pp.pandapowerNet, node_data: pd.DataFrame, converter_default) -> None:
    """
    Creates buses and components (loads, storages, etc.) in the network.

    Args:
        net (pp.pandapowerNet): The pandapower network.
        node_data (pd.DataFrame): DataFrame containing node data.
    """
    for _, row in node_data.iterrows():
        bus = pp.create_bus(
            net,
            index=row['Node number'],
            vn_kv=row['Operating nominal voltage (V)'] / 1000,
            name=f"Bus {row['Node number']}"
        )

        component_type = row['Component type'].replace(' ', '').lower()
        if component_type == 'acgrid':
            pp.create_ext_grid(net, bus=bus, vm_pu=1.0)
        elif component_type in ['dcload', 'acload']:
            l=pp.create_load(
                net,
                name='load ' + str(bus),
                bus=bus,
                p_mw=row['Maximum power (kW)'] / 1000,  # Convert kW to MW
                q_mvar=0
            )
            if 'default' in row['Droop curve of asset']:
                if 'droop_curve' not in net.load.columns:
                    net.load['droop_curve']=np.nan
                    net.load['droop_curve'] = net.load['droop_curve'].astype('object')
                str_dc=converter_default.loc[converter_default['Converter type']==row['Component type'],'Default Droop curve'].values[0]
                net.load.at[l,'droop_curve']=np.array(literal_eval('[' + str_dc.replace(';', ',') + ']'))
        elif component_type=='ev':
            pp.create_storage(
                net,
                name='EV ' + str(bus),
                bus=bus,
                p_mw=row['Maximum power (kW)'] / 1000,  # Convert kW to MW
                max_e_mwh=row['Maximum power (kW)'] / 1000 * 4,  # Convert kWh to MWh
                soc_percent=100  # Initial state of charge at 50%
            )
        elif component_type == 'storage':
            if not np.isnan(row['Maximum power (kW)']):
                pp.create_storage(
                    net,
                    name='Battery ' + str(bus),
                    bus=bus,
                    p_mw=row['Maximum power (kW)'] / 1000,  # Convert kW to MW
                    max_e_mwh=row['Capacity (kWh)'] / 1000,  # Convert kWh to MWh
                    soc_percent=50  # Initial state of charge at 50%
                )
            else :
                pp.create_storage(
                    net,
                    name='Battery ' + str(bus),
                    bus=bus,
                    p_mw=0,  # Convert kW to MW
                    max_e_mwh=0,  # Convert kWh to MWh
                    soc_percent=50  # Initial state of charge at 50%
                )
        elif component_type == 'pv':
            pp.create_sgen(
                net,
                bus=bus,
                p_mw=row['Maximum power (kW)'] / 1000  # Active power in MW
            )

        # Create linked converter bus if needed
        if not np.isnan(row['Node number for directly linked converter']):
            linked_bus_index = int(row['Node number for directly linked converter'])
            if linked_bus_index not in net.bus.index:
                pp.create_bus(
                    net,
                    index=linked_bus_index,
                    vn_kv=0,
                    name=f"Bus {linked_bus_index}"
                )


def _create_lines(net: pp.pandapowerNet, line_data: pd.DataFrame, cable_catalogue: pd.DataFrame) -> None:
    """
    Creates lines in the network based on the line data.

    Args:
        net (pp.pandapowerNet): The pandapower network.
        line_data (pd.DataFrame): DataFrame containing line data.
        cable_catalogue (pd.DataFrame): Processed cable catalogue.
    """
    for _, row in line_data.iterrows():
        # Create buses if they don't exist
        for node in [row['Node_i'], row['Node_j']]:
            if node not in net.bus.index:
                pp.create_bus(
                    net,
                    index=node,
                    vn_kv=0,
                    name=f"Bus {node}"
                )

        # Create the line
        if not np.isnan(row['Resistance (ohm/m)']):
            pp.create_line_from_parameters(
                net,
                from_bus=int(row['Node_i']),
                to_bus=int(row['Node_j']),
                length_km=row['Line length (m)'] / 1000,
                r_ohm_per_km=row['Resistance (ohm/m)']*1000,
                x_ohm_per_km=1e-20,
                c_nf_per_km=0,
                max_i_ka=1e3,
                cable_rank=None
            )
        else:
            cable = cable_catalogue.iloc[-1]
            pp.create_line_from_parameters(
                net,
                from_bus=int(row['Node_i']),
                to_bus=int(row['Node_j']),
                length_km=row['Line length (m)'] / 1000,
                r_ohm_per_km=cable['R'] * 1000,
                x_ohm_per_km=1e-20,
                c_nf_per_km=0,
                max_i_ka=cable['Imax'] / 1000,
                cable_rank=len(cable_catalogue) - 1
            )


def _fix_zero_voltages(subnetwork: pp.pandapowerNet) -> None:
    """
    Fixes zero voltages in the subnetwork by setting them to the first non-zero voltage.

    Args:
        subnetwork (pp.pandapowerNet): The subnetwork to fix.
    """
    if sum(np.isclose(subnetwork.bus.vn_kv.values, 0)):
        non_zero_voltage = subnetwork.bus.loc[~np.isclose(subnetwork.bus.vn_kv.values, 0), 'vn_kv'].iloc[0]
        subnetwork.bus.loc[np.isclose(subnetwork.bus.vn_kv.values, 0), 'vn_kv'] = non_zero_voltage


def _create_converters(net: pp.pandapowerNet, converter_data: pd.DataFrame, converter_default: pd.DataFrame, converter_catalogue: pd.DataFrame) -> None:
    """
    Creates converters in the network based on the converter data.

    Args:
        net (pp.pandapowerNet): The pandapower network.
        converter_data (pd.DataFrame): DataFrame containing converter data.
        converter_default (pd.DataFrame): DataFrame containing default converter data.
        converter_catalogue (pd.DataFrame): DataFrame containing converter catalogue data.
    """
    net.converter = pd.DataFrame(
        columns=[
            'name', 'from_bus', 'to_bus', 'type', 'P', 'efficiency', 'stand_by_loss',
            'efficiency curve', 'droop_curve', 'conv_rank', 'converter_catalogue'
        ]
    )

    converter_data = converter_data.rename(columns={
        "Converter name": "name",
        "Node_i number": 'from_bus',
        "Node_j number": 'to_bus',
        "Converter type": "type",
        "Nominal power (kW)": 'Nominal power (kW)',
        "Efficiency curve if user-defined": 'efficiency curve',
        "Voltage level V_i (V)": "V_i",
        "Voltage level V_j (V)": "V_j",
        "Droop curve if user-defined": 'droop_curve'
    })

    for _, row in converter_data.iterrows():
        if not np.isnan(row['Nominal power (kW)']):
            _add_converter(net, row, converter_default)
        else:
            _add_converter_from_catalogue(net, row, converter_catalogue, converter_default)


def _add_converter(net: pp.pandapowerNet, row: pd.Series, converter_default: pd.DataFrame) -> None:
    """
    Adds a converter to the network based on the provided row data.

    Args:
        net (pp.pandapowerNet): The pandapower network.
        row (pd.Series): Row containing converter data.
        converter_default (pd.DataFrame): DataFrame containing default converter data.
    """
    # Create buses if they don't exist
    for bus_key in ['from_bus', 'to_bus']:
        if row[bus_key] not in net.bus.index:
            voltage_key = 'V_i' if bus_key == 'from_bus' else 'V_j'
            voltage = row[voltage_key] / 1000 if not np.isnan(row[voltage_key]) else 100 / 1000
            pp.create_bus(
                net,
                index=row[bus_key],
                vn_kv=voltage,
                name=f"Bus {row[bus_key]}"
            )
        else:
            voltage_key = 'V_i' if bus_key == 'from_bus' else 'V_j'
            if not np.isnan(row[voltage_key]):
                net.bus.loc[row[bus_key], 'vn_kv'] = row[voltage_key] / 1000


    # Calculate efficiency and droop curves
    efficiency = _calculate_efficiency(row)
    droop_curve = _calculate_droop_curve(row, converter_default)

    # Add converter to the network
    new_row = {
        "name": row['name'],
        "from_bus": row['from_bus'],
        "to_bus": row['to_bus'],
        "type": row['type'],
        "P": row['Nominal power (kW)'] / 1000,  # Convert kW to MW
        "efficiency": efficiency,
        "efficiency curve": row['efficiency curve'],
        "droop_curve": droop_curve,
        'converter_catalogue': None,
        'conv_rank': None,
        'stand_by_loss' : 0
    }
    net.converter.loc[len(net.converter)] = new_row


def _add_converter_from_catalogue(net: pp.pandapowerNet, row: pd.Series, converter_catalogue: pd.DataFrame, converter_default: pd.DataFrame) -> None:
    """
    Adds a converter to the network based on the converter catalogue.

    Args:
        net (pp.pandapowerNet): The pandapower network.
        row (pd.Series): Row containing converter data.
        converter_catalogue (pd.DataFrame): DataFrame containing converter catalogue data.
        converter_default (pd.DataFrame): DataFrame containing default converter data.
    """
    type_c = row['type']
    tmp_cc = converter_catalogue.loc[converter_catalogue['Converter type'] == type_c].copy()

    if np.isnan(row["V_i"]):
        tmp_cc = tmp_cc.loc[
            (tmp_cc['Voltage level V1 (V)'] == row["V_j"]) |
            (tmp_cc['Voltage level V2 (V)'] == row["V_j"])
        ]
    else:
        tmp_cc = tmp_cc.loc[
            ((tmp_cc['Voltage level V1 (V)'] == row["V_j"]) & (tmp_cc['Voltage level V2 (V)'] == row["V_i"])) |
            ((tmp_cc['Voltage level V1 (V)'] == row["V_i"]) & (tmp_cc['Voltage level V2 (V)'] == row["V_j"]))
        ]

    tmp_cc.reset_index(inplace=True, drop=True)
    conv = tmp_cc.loc[tmp_cc['Nominal power (kW)'].idxmin()]

    # Create buses if they don't exist
    for bus_key in ['from_bus', 'to_bus']:
        if row[bus_key] not in net.bus.index:
            voltage_key = 'V_i' if bus_key == 'from_bus' else 'V_j'
            voltage = row[voltage_key] / 1000 if not np.isnan(row[voltage_key]) else 100 / 1000
            pp.create_bus(
                net,
                index=row[bus_key],
                vn_kv=voltage,
                name=f"Bus {row[bus_key]}"
            )
        else:
            voltage_key = 'V_i' if bus_key == 'from_bus' else 'V_j'
            if not np.isnan(row[voltage_key]):
                net.bus.loc[row[bus_key], 'vn_kv'] = row[voltage_key] / 1000

    # Calculate efficiency and droop curves
    efficiency = _calculate_efficiency(row, conv)
    droop_curve = _calculate_droop_curve(row, converter_default)

    # Add converter to the network
    new_row = {
        "name": row['name'],
        "from_bus": row['from_bus'],
        "to_bus": row['to_bus'],
        "type": row['type'],
        "P": conv['Nominal power (kW)'] / 1000,  # Convert kW to MW
        "efficiency": efficiency,
        "efficiency curve": row['efficiency curve'],
        "droop_curve": droop_curve,
        'converter_catalogue': tmp_cc,
        "conv_rank": tmp_cc['Nominal power (kW)'].idxmin(),
        'stand_by_loss' : conv['Stand-by losses (W)'] / 1e6
    }
    net.converter.loc[len(net.converter)] = new_row


def _calculate_efficiency(row: pd.Series, conv: pd.Series = None) -> np.ndarray:
    """
    Calculates the efficiency curve for a converter.

    Args:
        row (pd.Series): Row containing converter data.
        conv (pd.Series, optional): Row containing converter catalogue data.

    Returns:
        np.ndarray: Efficiency curve as a numpy array.
    """
    if row['Efficiency curve'] == 'user-defined':
        eff = np.array(literal_eval(row['efficiency curve']))
        e = eff[:, 1].astype('float') / 100
        p = eff[:, 0] / 100 * (row['Nominal Maximum power (kW)'] if conv is None else conv['Nominal power (kW)'])
        return np.vstack((p, e)).T
    else:
        eff_str = conv['Efficiency curve [Xi;Yi], i={1,2,3,4}, \nwith X= Factor of Nominal Power (%), Y=Efficiency (%)']
        eff = np.array(literal_eval('[' + eff_str.replace(';', ',') + ']'))
        e = eff[:, 1].astype('float') / 100
        p = eff[:, 0] / 100 * conv['Nominal power (kW)']
        return np.vstack((p, e)).T


def _calculate_droop_curve(row: pd.Series, converter_default: pd.DataFrame) -> np.ndarray:
    """
    Calculates the droop curve for a converter.

    Args:
        row (pd.Series): Row containing converter data.
        converter_default (pd.DataFrame): DataFrame containing default converter data.

    Returns:
        np.ndarray: Droop curve as a numpy array.
    """
    if 'user-defined' in row["Droop curve"]:
        return np.array(literal_eval(row['droop_curve']))
    else:
        str_dc = converter_default.loc[
            converter_default['Converter type'] == row['type'],
            'Default Droop curve'
        ].values[0]
        return np.array(literal_eval('[' + str_dc.replace(';', ',') + ']'))
