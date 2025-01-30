import pandapower as pp
from loadflow_utilities import perform_dc_load_flow,perform_load_flow_with_sizing
from plot_utilities import plot_network_with_plotly
import copy
import math as m 

def worst_case_storage_sizing(net,cable_catalogue,use_case):
    net_storage = copy.deepcopy(net)

    load_prct=use_case['Worst case scenario 1 for sizing of Storage DC/DC converter ']['Loads power factor (%)']
    load_exp_prct=use_case['Sizing factor']['Load expansion factor (%)']
    pv_prct=use_case['Worst case scenario 1 for sizing of Storage DC/DC converter ']['PV power factor (%)']
    ev_prct=use_case['Worst case scenario 1 for sizing of Storage DC/DC converter ']['EV Charging station consumption (%)']
    ac_grid=use_case['Worst case scenario 1 for sizing of Storage DC/DC converter ']['AC grid']
    storage_duration=use_case['Worst case scenario 1 for sizing of Storage DC/DC converter ']['Storage duration (hours)']

    net_storage.load['p_mw']=net_storage.load['p_mw']*(load_prct/100)*(load_exp_prct/100)
    net_storage.sgen['p_mw']=net_storage.sgen['p_mw']*pv_prct/100
    net_storage.storage.loc[['EV' in x for x in net_storage.storage['name']], 'p_mw']=net_storage.storage.loc[['EV' in x for x in net_storage.storage['name']], 'p_mw']*ev_prct/100

    net_storage.ext_grid['in_service']=False
    net_storage.storage.loc[['Battery' in x for x in net_storage.storage['name']],'in_service']=False
    battery_spec={}
    for i,row in net_storage.storage.loc[['Battery' in x for x in net_storage.storage['name']]].iterrows():
        pp.create_ext_grid(net_storage, bus=row['bus'], vm_pu=1.0)
        net_storage=perform_load_flow_with_sizing(net_storage, cable_catalogue, use_case)
        battery_nominal_power=5*m.ceil(net_storage.res_ext_grid.loc[net_storage.ext_grid['in_service']==True,'p_mw'].values[0]*1000/5)

        battery_spec[net_storage.storage.loc[i,'name']]={"power" : battery_nominal_power,
                                                         "energy" : battery_nominal_power*storage_duration}
    
    return net_storage,battery_spec

def write_battery_spec(net,battery_spec):
    for bat_name in battery_spec.keys():
        net.storage.loc[net.storage['name']==bat_name,'p_mw']=abs(battery_spec[bat_name]['power']/1000)
        net.storage.loc[net.storage['name']==bat_name,'max_e_mwh']=abs(battery_spec[bat_name]['energy']/1000)

    
def worst_case2(net,cable_catalogue,use_case):
    load_prct=use_case['Worst case scenario 2 for sizing of cables and  PDU DC/DC,  DC/AC, PV DC/DC and EV DC/DC converters ']['Loads power factor (%)']
    load_exp_prct=use_case['Sizing factor']['Load expansion factor (%)']
    pv_prct=use_case['Worst case scenario 2 for sizing of cables and  PDU DC/DC,  DC/AC, PV DC/DC and EV DC/DC converters ']['PV power factor (%)']
    ev_prct=use_case['Worst case scenario 2 for sizing of cables and  PDU DC/DC,  DC/AC, PV DC/DC and EV DC/DC converters ']['EV Charging station consumption (%)']
    ac_grid=use_case['Worst case scenario 2 for sizing of cables and  PDU DC/DC,  DC/AC, PV DC/DC and EV DC/DC converters ']['AC grid']
    storage_prct=use_case['Worst case scenario 2 for sizing of cables and  PDU DC/DC,  DC/AC, PV DC/DC and EV DC/DC converters ']['Storage power contribution (%)']

    net_2 = copy.deepcopy(net)

    net_2.load['p_mw']=net_2.load['p_mw']*(load_prct/100)*(load_exp_prct/100)
    net_2.sgen['p_mw']=net_2.sgen['p_mw']*pv_prct/100
    net_2.storage.loc[['EV' in x for x in net_2.storage['name']], 'p_mw']=net_2.storage.loc[['EV' in x for x in net_2.storage['name']], 'p_mw']*ev_prct/100
    net_2.storage.loc[['Battery' in x for x in net_2.storage['name']], 'p_mw']=abs(net_2.storage.loc[['Battery' in x for x in net_2.storage['name']], 'p_mw'])*storage_prct/100*-1

    net_2=perform_load_flow_with_sizing(net_2, cable_catalogue, use_case)
    return net_2
        
def worst_case3(net,cable_catalogue,use_case):
    load_prct=use_case['Worst case scenario 3 for sizing cables and AC/DC converter']['Loads power factor (%)']
    load_exp_prct=use_case['Sizing factor']['Load expansion factor (%)']
    pv_prct=use_case['Worst case scenario 3 for sizing cables and AC/DC converter']['PV power factor (%)']
    ev_prct=use_case['Worst case scenario 3 for sizing cables and AC/DC converter']['EV Charging station consumption (%)']
    ac_grid=use_case['Worst case scenario 3 for sizing cables and AC/DC converter']['AC grid']
    storage_prct=use_case['Worst case scenario 3 for sizing cables and AC/DC converter']['Storage power contribution (%)']

    net_3 = copy.deepcopy(net)

    net_3.load['p_mw']=net_3.load['p_mw']*(load_prct/100)*(load_exp_prct/100)
    net_3.sgen['p_mw']=net_3.sgen['p_mw']*pv_prct/100
    net_3.storage.loc[['EV' in x for x in net_3.storage['name']], 'p_mw']=net_3.storage.loc[['EV' in x for x in net_3.storage['name']], 'p_mw']*ev_prct/100
    net_3.storage.loc[['Battery' in x for x in net_3.storage['name']], 'p_mw']=net_3.storage.loc[['Battery' in x for x in net_3.storage['name']], 'p_mw']*storage_prct/100*-1

    net_3=perform_load_flow_with_sizing(net_3, cable_catalogue, use_case)
    
    return net_3


def test_worst_case_storage(net,use_case):
    net_storage = copy.deepcopy(net)

    load_prct=use_case['Worst case scenario 1 for sizing of Storage DC/DC converter ']['Loads power factor (%)']
    load_exp_prct=use_case['Sizing factor']['Load expansion factor (%)']
    pv_prct=use_case['Worst case scenario 1 for sizing of Storage DC/DC converter ']['PV power factor (%)']
    ev_prct=use_case['Worst case scenario 1 for sizing of Storage DC/DC converter ']['EV Charging station consumption (%)']
    ac_grid=use_case['Worst case scenario 1 for sizing of Storage DC/DC converter ']['AC grid']
    storage_duration=use_case['Worst case scenario 1 for sizing of Storage DC/DC converter ']['Storage duration (hours)']

    net_storage.load['p_mw']=net_storage.load['p_mw']*(load_prct/100)*(load_exp_prct/100)
    net_storage.sgen['p_mw']=net_storage.sgen['p_mw']*pv_prct/100
    net_storage.storage.loc[['EV' in x for x in net_storage.storage['name']], 'p_mw']=net_storage.storage.loc[['EV' in x for x in net_storage.storage['name']], 'p_mw']*ev_prct/100

    net_storage=perform_dc_load_flow(net_storage)
    plot_network_with_plotly(net_storage)
    return net_storage

    
def test_worst_case2(net,use_case):
    load_prct=use_case['Worst case scenario 2 for sizing of cables and  PDU DC/DC,  DC/AC, PV DC/DC and EV DC/DC converters ']['Loads power factor (%)']
    load_exp_prct=use_case['Sizing factor']['Load expansion factor (%)']
    pv_prct=use_case['Worst case scenario 2 for sizing of cables and  PDU DC/DC,  DC/AC, PV DC/DC and EV DC/DC converters ']['PV power factor (%)']
    ev_prct=use_case['Worst case scenario 2 for sizing of cables and  PDU DC/DC,  DC/AC, PV DC/DC and EV DC/DC converters ']['EV Charging station consumption (%)']
    ac_grid=use_case['Worst case scenario 2 for sizing of cables and  PDU DC/DC,  DC/AC, PV DC/DC and EV DC/DC converters ']['AC grid']
    storage_prct=use_case['Worst case scenario 2 for sizing of cables and  PDU DC/DC,  DC/AC, PV DC/DC and EV DC/DC converters ']['Storage power contribution (%)']

    net_2 = copy.deepcopy(net)

    net_2.load['p_mw']=net_2.load['p_mw']*(load_prct/100)*(load_exp_prct/100)
    net_2.sgen['p_mw']=net_2.sgen['p_mw']*pv_prct/100
    net_2.storage.loc[['EV' in x for x in net_2.storage['name']], 'p_mw']=net_2.storage.loc[['EV' in x for x in net_2.storage['name']], 'p_mw']*ev_prct/100
    net_2.storage.loc[['Battery' in x for x in net_2.storage['name']], 'p_mw']=abs(net_2.storage.loc[['Battery' in x for x in net_2.storage['name']], 'p_mw'])*storage_prct/100*-1

    net_2=perform_dc_load_flow(net_2)
    plot_network_with_plotly(net_2)
    return net_2
        
def test_worst_case3(net,use_case):
    load_prct=use_case['Worst case scenario 3 for sizing cables and AC/DC converter']['Loads power factor (%)']
    load_exp_prct=use_case['Sizing factor']['Load expansion factor (%)']
    pv_prct=use_case['Worst case scenario 3 for sizing cables and AC/DC converter']['PV power factor (%)']
    ev_prct=use_case['Worst case scenario 3 for sizing cables and AC/DC converter']['EV Charging station consumption (%)']
    ac_grid=use_case['Worst case scenario 3 for sizing cables and AC/DC converter']['AC grid']
    storage_prct=use_case['Worst case scenario 3 for sizing cables and AC/DC converter']['Storage power contribution (%)']

    net_3 = copy.deepcopy(net)

    net_3.load['p_mw']=net_3.load['p_mw']*(load_prct/100)*(load_exp_prct/100)
    net_3.sgen['p_mw']=net_3.sgen['p_mw']*pv_prct/100
    net_3.storage.loc[['EV' in x for x in net_3.storage['name']], 'p_mw']=net_3.storage.loc[['EV' in x for x in net_3.storage['name']], 'p_mw']*ev_prct/100
    net_3.storage.loc[['Battery' in x for x in net_3.storage['name']], 'p_mw']=net_3.storage.loc[['Battery' in x for x in net_3.storage['name']], 'p_mw']*storage_prct/100*-1

    net_3=perform_dc_load_flow(net_3)
    plot_network_with_plotly(net_3)
    
    return net_3



def keep_greater_network(net_1,net_2):
    big_net = copy.deepcopy(net_1)
    for i in list(big_net.converter.index):
        if net_1.converter.loc[i,"P"]<net_2.converter.loc[i,"P"]:
            big_net.converter.loc[i]=net_2.converter.loc[i]
        else : 
            big_net.converter.loc[i]=net_1.converter.loc[i]

    for i in list(big_net.line.index):
        if net_1.line.loc[i,"max_i_ka"]<net_2.line.loc[i,"max_i_ka"]:
            big_net.line.loc[i]=net_2.line.loc[i]
        else : 
            big_net.line.loc[i]=net_1.line.loc[i]
    big_net_save = copy.deepcopy(big_net)
    return big_net_save

def process_worst_case_sizing(net,cable_catalogue,use_case):
    net_storage,battery_spec=worst_case_storage_sizing(net,cable_catalogue,use_case)
    print(net_storage.line.loc[2])
    write_battery_spec(net,battery_spec)
    net_2 = worst_case2(net,cable_catalogue,use_case)
    print(net_2.line.loc[2])
    net_3 = worst_case3(net,cable_catalogue,use_case)
    print(net_3.line.loc[2])
    big_net = keep_greater_network(keep_greater_network(net_2,net_storage),net_3)
    return big_net

def test_sized_network(big_net,use_case):
    net_1=test_worst_case_storage(big_net,use_case)
    net_2=test_worst_case2(big_net,use_case)
    net_3=test_worst_case3(big_net,use_case)
