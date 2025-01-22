import pandas as pd
import pandapower as pp
from ast import literal_eval
import numpy as np
import openpyxl
from read_cable_catalogue import read_cable_catalogue, process_cable_catalogue
from reading_utilities import find_cable_info,read_UC_Definition
from topology_utilities import separate_subnetworks,sorting_network,merge_networks



def create_DC_network(path, path_cable_catalogue, return_buses=False):
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


    xl_file = pd.ExcelFile(path)
    Uc=read_UC_Definition(xl_file)
    # Parse the necessary sheets from the Excel file
    line_data = xl_file.parse('Lines')
    node_data = xl_file.parse('Assets Nodes')
    converter_data = xl_file.parse('Converters')

    # Read and process the cable catalogue
    cable_catalogue = read_cable_catalogue(path_cable_catalogue)
    cable_info = Uc['Conductor parameters']
    cable_catalogue = process_cable_catalogue(cable_catalogue, cable_info)
    
    # Create buses for each node
    for _, row in node_data.iterrows():
        bus = pp.create_bus(net,index=row['Node number'], vn_kv=row['Operating nominal voltage (V)']/1000, name=f"Bus {row['Node number']}")
        if row['Component type'].replace(' ','').lower()=='AC Grid'.replace(' ','').lower():
            pp.create_ext_grid(net, bus=bus, vm_pu=1.0)
        elif row['Component type'].replace(' ','').lower()=='DC Load'.replace(' ','').lower():    
            pp.create_load(net,
                           bus=bus,
                           p_mw=row['Maximum power (kW)'] / 1000,  # Convert kW to MW
                           q_mvar=0)
        elif row['Component type'].replace(' ','').lower()=='AC Load'.replace(' ','').lower():    
            pp.create_load(net,
                           bus=bus,
                           p_mw=row['Maximum power (kW)'] / 1000,  # Convert kW to MW
                           q_mvar=0)
        elif row['Component type'].replace(' ','').lower()=='Storage'.lower():  
            pp.create_storage(net,
                              bus=bus,
                              p_mw=row['Maximum power (kW)'] / 1000,  # Max active power, Convert kW to MW
                              max_e_mwh=row['Capacity (kWh)'] / 1000,  # Max energy capacity in MWh
                              soc_percent=50)  # Assuming initial state of charge at 50%
        elif row['Component type'].replace(' ','').lower()=='PV'.lower():  
            pp.create_sgen(net,
                           bus=bus,
                           p_mw=row['Maximum power (kW)'] / 1000)  # Active power in MW
        if (not np.isnan(row['Node number for directly linked converter'])) and (row['Node number for directly linked converter'] not in list(net.bus.index)):
            bus = pp.create_bus(net,index=row['Node number for directly linked converter'], 
                                vn_kv=0, 
                                name=f"Bus {row['Node number for directly linked converter']}")
    # Create lines based on the line data
    for _, row in line_data.iterrows():
        if row['Node_i'] not in list(net.bus.index):
            bus = pp.create_bus(net,index=row['Node_i'], 
                                vn_kv=0, 
                                name=f"Bus {row['Node_i']}")
        if row['Node_j'] not in list(net.bus.index):
            bus = pp.create_bus(net,index=row['Node_j'], 
                                vn_kv=0, 
                                name=f"Bus {row['Node_j']}")
            
        if 'Rij_ohm_km' in row.keys():
            pp.create_line_from_parameters(net,
                                           from_bus=int(row['Node_i']),
                                           to_bus=int(row['Node_j']),
                                           length_km=row['Line length (m)']/1000,
                                           r_ohm_per_km=row['Rij_ohm_km'],
                                           x_ohm_per_km=row['Xij_ohm_km'],
                                           c_nf_per_km=row['Cij_ohm_km'],
                                           max_i_ka=row['Imax_ij_kA'])
        else:
            cable = cable_catalogue.iloc[-1]
            pp.create_line_from_parameters(net,
                                           from_bus=int(row['Node_i']),
                                           to_bus=int(row['Node_j']),
                                           length_km=row['Line length (m)']/1000,
                                           r_ohm_per_km=cable['R'] * 1000,
                                           x_ohm_per_km=1e-20,
                                           c_nf_per_km=0,
                                           max_i_ka=cable['Imax'] / 1000,
                                           cable_rank=len(cable_catalogue) - 1)
    
    L=separate_subnetworks(net)
    for n in L:
        if sum(np.isclose(n.bus.vn_kv.values,0)):
            n.bus.loc[np.isclose(n.bus.vn_kv.values,0),'vn_kv'] = n.bus.loc[~np.isclose(n.bus.vn_kv.values,0),'vn_kv'].iloc[0]
    net=merge_networks(L)

    net.converter = pd.DataFrame(columns=['name','from_bus','to_bus','type','P','efficiency',"droop_curve"])
    converter_data=converter_data.rename(columns={"Converter name":"name",
                                                  "Node_i number\nAsset side / Lower Voltage Bus side " : 'from Asset/Node to Bus/Node',
                                                  "Node_j number\nDC Grid side / Higher Voltage Bus side":'to_bus/node ',
                                                  "Converter type":"type",
                                                  "Nominal power (kW)":'Nominal Maximum power (kW)',
                                                  "Efficiency curve if user-efined [Xi;Yi], i={1,2,3,4}, \nwith X= Factor of Nominal Power (%), Y=Efficiency (%)": 'efficiency curve',
                                                  "Voltage level V_i (V)\nAsset side / Lower Voltage Bus side " : "V_i",
                                                  "Voltage level V_j(V)\nDC Grid side / Higher Voltage Bus side":"V_j"})

    for _, row in converter_data.iterrows():
        if row['from Asset/Node to Bus/Node'] not in list(net.bus.index):
            if np.isnan(row["V_i"]):
                bus = pp.create_bus(net,index=row['from Asset/Node to Bus/Node'], 
                                    vn_kv=100, 
                                    name=f"Bus {row['from Asset/Node to Bus/Node']}")
            else :
                bus = pp.create_bus(net,index=row['from Asset/Node to Bus/Node'], 
                                    vn_kv=row["V_i"]/1000, 
                                    name=f"Bus {row['from Asset/Node to Bus/Node']}")
        else :
            if np.isnan(row["V_i"]):
                net.bus.loc[row['from Asset/Node to Bus/Node'],"vn_kv"]=100
            else :
                net.bus.loc[row['from Asset/Node to Bus/Node'],"vn_kv"]=row["V_i"]/1000
            

        if row['to_bus/node '] not in list(net.bus.index):
            bus = pp.create_bus(net,index=row['to_bus/node '], 
                                vn_kv=row["V_j"]/1000, 
                                name=f"Bus {row['to_bus/node ']}")
        else :
            net.bus.loc[row['to_bus/node '],"vn_kv"]=row["V_j"]/1000
        
        if row['Efficiency curve'] == 'user-defined':
            eff=np.array(literal_eval(row['efficiency curve']))
            e=eff[:,1].astype('float')/100
            p=eff[:,0]/100*row['Nominal Maximum power (kW)']
            efficiency=np.vstack((p,e)).T

        if 'user-defined' in row["Droop curve"]:
            dc=np.array(literal_eval(row['Droop curve if user-efined']))
        else:
            dc=np.array([[111,111],[111,111],[111,111]])


        new_row = {
            "name": row['name'],
            "from_bus": row['from Asset/Node to Bus/Node'],
            "to_bus": row['to_bus/node '],
            "type": row['type'],
            "P": row['Nominal Maximum power (kW)'] / 1000,  # Convert kW to MW
            "efficiency": efficiency,
            "droop_curve" : dc
        }
        net.converter.loc[len(net.converter)] = new_row
    
    
    return net, cable_catalogue
