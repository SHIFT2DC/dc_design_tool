import pandapower as pp
import numpy as np
import pandas as pd
import copy
from topology_utilities import separate_subnetworks,sorting_network,merge_networks,find_lines_between_given_line_and_ext_grid
import math as m
from ast import literal_eval

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

def LF_DC(net):
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
                # print(tmp_net.line)
                # print('>> Loadflow of network '+ str(n))
                pp.runpp(tmp_net)
                # print(tmp_net.res_line)
                # print("*************************************")
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



def LF_sizing(net,cable_catalogue,use_case):
    if use_case['Project details']['Ecosystem']=='ODCA':
        min_v=0.92
        max_v=1.08
    elif use_case['Project details']['Ecosystem']=='CurrentOS':
        min_v=0.98
        max_v=1.02 
    else :
        min_v=0.95
        max_v=1.05


    net=LF_DC(net)
    for i in net.converter.index:
        if net.converter.loc[i,'conv_rank'] is not None:
            tmp_cc=net.converter.loc[i,'converter_catalogue']

            if (tmp_cc['Nominal power (kW)']>(net.res_converter.loc[i,'p_mw']*1000)).values.any():
                filtered_tmp_cc = tmp_cc[tmp_cc['Nominal power (kW)'] > abs(net.res_converter.loc[i,'p_mw']*1000)]
                new_c = filtered_tmp_cc.loc[filtered_tmp_cc['Nominal power (kW)'].idxmin()]

                if net.converter.loc[i,'efficiency'] == 'user-defined':
                    eff=net.converter.loc[i,'efficiency'] 
                    p_previous = eff[:, 0]*100/net.converter.loc[i,'P']
                    p = p_previous/100*new_c['Nominal power (kW)']
                    efficiency = np.vstack((p, e)).T
                else :
                    eff_str=new_c['Efficiency curve [Xi;Yi], i={1,2,3,4}, \nwith X= Factor of Nominal Power (%), Y=Efficiency (%)']
                    eff=np.array(literal_eval('['+eff_str.replace(';',',')+']'))
                    e = eff[:, 1].astype('float')/100
                    p = eff[:, 0]/100*new_c['Nominal power (kW)']
                    efficiency = np.vstack((p, e)).T
                net.converter.loc[i,'conv_rank']=filtered_tmp_cc['Nominal power (kW)'].idxmin()
                print(efficiency)
                net.converter.at[i, 'efficiency']=efficiency
                net.converter.loc[i, 'P']=new_c['Nominal power (kW)']/1000
            else :
                new_c = tmp_cc.loc[tmp_cc['Nominal power (kW)'].idxmax()]
                if net.converter.loc[i,'efficiency'] == 'user-defined':
                    eff=net.converter.loc[i,'efficiency'] 
                    p_previous = eff[:, 0]*100/net.converter.loc[i,'P']
                    p = p_previous/100*new_c['Nominal power (kW)']
                    efficiency = np.vstack((p, e)).T
                else :
                    eff_str=new_c['Efficiency curve [Xi;Yi], i={1,2,3,4}, \nwith X= Factor of Nominal Power (%), Y=Efficiency (%)']
                    eff=np.array(literal_eval('['+eff_str.replace(';',',')+']'))
                    e = eff[:, 1].astype('float')/100
                    p = eff[:, 0]/100*new_c['Nominal power (kW)']
                    efficiency = np.vstack((p, e)).T
                net.converter.loc[i,'conv_rank']=filtered_tmp_cc['Nominal power (kW)'].idxmin()
                net.converter.at[i, 'efficiency']=efficiency
                net.converter.loc[i, 'P']=new_c['Nominal power (kW)']/1000


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
                # print(tmp_net.line)
                # print('>> Loadflow of network '+ str(n))
                pp.runpp(tmp_net)
                # print(tmp_net.res_line)
                # print("*************************************")
                tmp_net.res_line.i_ka.sort_values()
                if len(tmp_net.res_line)>0:
                    for i in tmp_net.res_line.i_ka.sort_values(ascending=False).index:
                        optimal=False
                        while (not optimal):
                            if tmp_net.res_line.loc[i,'p_from_mw']>0:
                                tension_of_interest = 'vm_to_pu'
                            else:
                                tension_of_interest = 'vm_from_pu'
                            idx_new_cable = tmp_net.line.loc[i,"cable_rank"]
                            idx_cable=idx_new_cable
                            load_flow_converge=True
                            lines_beetween=find_lines_between_given_line_and_ext_grid(tmp_net,i)
                            while ((tmp_net.res_line.loc[i,"loading_percent"]<100) 
                                    and (tmp_net.res_line.loc[i,tension_of_interest]<max_v)
                                    and (tmp_net.res_line.loc[i,tension_of_interest]>min_v)
                                    and (idx_new_cable>=1)
                                    and (load_flow_converge)):
                                idx_cable = idx_new_cable
                                idx_new_cable = idx_cable -1
                                new_cable = cable_catalogue.loc[idx_new_cable]
                                tmp_net.line.r_ohm_per_km.loc[i] = new_cable['R'] * 1000
                                tmp_net.line.max_i_ka.loc[i] = new_cable['Imax'] / 1000
                                tmp_net.line.cable_rank.loc[i] = idx_new_cable
                                try :
                                    pp.runpp(tmp_net)
                                except :
                                    load_flow_converge = False

                            if not ((tmp_net.res_line.loc[i,"loading_percent"]<100) 
                                    and (tmp_net.res_line.loc[i,tension_of_interest]<max_v)
                                    and (tmp_net.res_line.loc[i,tension_of_interest]>min_v)
                                    and (load_flow_converge)):
                                new_cable = cable_catalogue.loc[idx_cable]
                                tmp_net.line.r_ohm_per_km.loc[i] = new_cable['R'] * 1000
                                tmp_net.line.max_i_ka.loc[i] = new_cable['Imax'] / 1000
                                tmp_net.line.cable_rank.loc[i] = idx_cable
                            pp.runpp(tmp_net)

                            optimal=True
                            for l in lines_beetween:
                                cable_rank_beetween=tmp_net.line.loc[l,'cable_rank']
                                print(cable_rank_beetween,tmp_net.line.loc[i,'cable_rank'])
                                if cable_rank_beetween<tmp_net.line.loc[i,'cable_rank']:
                                    new_cable = cable_catalogue.loc[cable_rank_beetween+1]
                                    tmp_net.line.r_ohm_per_km.loc[l] = new_cable['R'] * 1000
                                    tmp_net.line.max_i_ka.loc[l] = new_cable['Imax'] / 1000
                                    tmp_net.line.cable_rank.loc[l] = cable_rank_beetween+1

                                    new_cable = cable_catalogue.loc[cable_rank_beetween+1]
                                    tmp_net.line.r_ohm_per_km.loc[i] = new_cable['R'] * 1000
                                    tmp_net.line.max_i_ka.loc[i] = new_cable['Imax'] / 1000
                                    tmp_net.line.cable_rank.loc[i] = cable_rank_beetween+1
                                    optimal=False
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