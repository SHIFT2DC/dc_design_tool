import pandapower as pp
import numpy as np
import pandas as pd
import copy
from topology_utilities import separate_subnetworks,sorting_network,merge_networks
import math as m

def clean_network(net,original_net):

    net.converter=original_net.converter
    net.res_converter=original_net.res_converter
    # net.res_converter=pd.DataFrame(columns=["p_mw","loading (%)"])
    # for i, row in net.converter.iterrows():
    #     b=row.from_bus
    #     l=net.load.loc[net.load.bus==b]
    #     if len(l)>0:
    #         net.res_converter.loc[len(net.res_converter)]={"p_mw":l.p_mw.values[0],"loading (%)":l.p_mw.values[0]/row.P*100}
    #     else:
    #         net.res_converter.loc[len(net.res_converter)]={"p_mw":None,"loading (%)":None}


    del_load=['Load of net' in str(x) for x in net.load.name.values]
    net.load.drop(net.load.loc[del_load].index,inplace=True)
    net.res_load.drop(net.res_load.loc[del_load].index,inplace=True)

    del_ext_grid=['Converter emulation' in str(x) for x in net.ext_grid.name.values]
    net.ext_grid.drop(net.ext_grid.loc[del_ext_grid].index,inplace=True)
    net.res_ext_grid.drop(net.res_ext_grid.loc[del_ext_grid].index,inplace=True)
    
    return net

def New_LF(net):
    subnetwork_list = separate_subnetworks(net)
    dic_of_subs=sorting_network(net, subnetwork_list)
    #print(dic_of_subs)
    loadflowed_sub=[]
    net.res_converter=pd.DataFrame(data=np.empty((len(net.converter),3)),columns=["p_mw","loading (%)",'pl_mw'])
    while not all(elem in loadflowed_sub for elem in dic_of_subs.keys()):
        for n in list(set(list(dic_of_subs.keys()))-set(loadflowed_sub)):
            if all(elem in loadflowed_sub for elem in [x[0] for x in dic_of_subs[n]['direct_downstream_network']]):
                tmp_net=dic_of_subs[n]['network']
                for c in dic_of_subs[n]['direct_upstream_network']:
                    b=[x[1] for x in dic_of_subs[c[0]]['direct_downstream_network'] if x[0]==n][0]
                    pp.create_ext_grid(tmp_net,bus=b,vm_pu=1.0,name='Converter emulation')
                # print(tmp_net)
                # print('>> Loadflow of network '+ str(n))
                pp.runpp(tmp_net)
                dic_of_subs[n]['network']=tmp_net
                loadflowed_sub.append(n)
                for c in dic_of_subs[n]['direct_upstream_network']:
                    up_net=dic_of_subs[c[0]]['network']
                    p=tmp_net.res_ext_grid.p_mw.values[0]
                    converter=net.converter.loc[net.converter.name==c[2]]
                    eff = np.interp(abs(p), converter.efficiency.values[0][:, 0] / 1000, converter.efficiency.values[0][:, 1])
                    pp.create_load(up_net,
                            bus=c[1],
                            p_mw=p*eff*int(p<0)+p/eff*int(p>0),  # Convert kW to MW
                            q_mvar=0,name='Load of net '+ str(n))
                    dic_of_subs[c[0]]['network']=up_net
                    net.res_converter.loc[net.converter.name==c[2],'p_mw']=p*eff*int(p<0)+p/eff*int(p>0)
                    net.res_converter.loc[net.converter.name==c[2],'loading (%)']=p/net.converter.loc[net.converter.name==c[2],'P']*100
                    net.res_converter.loc[net.converter.name==c[2],'pl_mw']=p-(p*eff*int(p<0)+p/eff*int(p>0))

    net_res=merge_networks([dic_of_subs[n]['network'] for n in dic_of_subs.keys()])
    net=clean_network(net_res,net)
    return net



def new_optimisation_cable_size(net, cable_catalogue):
    """
    Optimizes the cable sizes in the DC network based on load flow results.

    Parameters:
    net (pandapowerNet): The pandapower network object.
    cable_catalogue (pd.DataFrame): The cable catalogue data.

    Returns:
    pandapowerNet: The network object with optimized cable sizes.
    """
    print(net)
    net = New_LF(net)
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
            #print(idx_new_cable)
            new_cable = cable_catalogue.loc[idx_new_cable]
            net.line.r_ohm_per_km.loc[i] = new_cable['R'] * 1000
            net.line.max_i_ka.loc[i] = new_cable['Imax'] / 1000
            net.line.cable_rank.loc[i] = idx_new_cable
        try :
            net = New_LF(net)
        except :
            load_flow_converge = False
            break

    # Revert to former cable sizes if load flow does not converge
    if not load_flow_converge:
        for i in range(len(net.line)):
            new_cable = cable_catalogue.loc[former_cable_ranks[i]]
            net.line.r_ohm_per_km.loc[i] = new_cable['R'] * 1000
            net.line.max_i_ka.loc[i] = new_cable['Imax'] / 1000
            net.line.cable_rank.loc[i] = former_cable_ranks[i]
        net = New_LF(net)
    
    return net
