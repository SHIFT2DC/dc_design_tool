import pandapower as pp
import pandapower.topology as top

def separate_subnetworks(net):
    # Find all the connected subnetworks
    subnetworks = top.connected_components(top.create_nxgraph(net,include_trafos=False))
    
    # Create a list to store the individual subnetworks
    subnetwork_list = []
    
    # Iterate over each subnetwork and create a new network for each
    for subnetwork in subnetworks:
        sub_net = pp.select_subnet(net, subnetwork)
        subnetwork_list.append(sub_net)
    
    return subnetwork_list


def sorting_network(net, subneworks):
    dic_of_subs={}
    for n in range(len(subneworks)):
        subn=subneworks[n]
        
        dic_of_subs[n]={'network':subn,'direct_connect_network':[]}
        for b in subn.bus.index:
            if b in net.converter.from_bus.values:
                for i,bus_to_find in enumerate(net.converter.loc[net.converter.from_bus==b].to_bus.values):
                    for idx_tmp_sub in range(len(subneworks)):
                        tmp_sub=subneworks[idx_tmp_sub]
                        if bus_to_find in tmp_sub.bus.index:
                            dic_of_subs[n]['direct_connect_network'].append([idx_tmp_sub,bus_to_find,net.converter.loc[net.converter.from_bus==b].name.values[i]])
            if b in net.converter.to_bus.values:
                for i,bus_to_find in enumerate(net.converter.loc[net.converter.to_bus==b].from_bus.values):
                    for idx_tmp_sub in range(len(subneworks)):
                        tmp_sub=subneworks[idx_tmp_sub]
                        if bus_to_find in tmp_sub.bus.index:
                            dic_of_subs[n]['direct_connect_network'].append([idx_tmp_sub,bus_to_find,net.converter.loc[net.converter.to_bus==b].name.values[i]])
    
    network_dict=find_upndownstream_networks(dic_of_subs)

    return network_dict

def find_upndownstream_networks(network_dict):
    def is_upstream(network_id, visited):
        # Si le réseau a une ext_grid, c'est un réseau amont
        if len(network_dict[network_id]['network'].ext_grid)!=0:
            return True
        # Marquer le réseau actuel comme visité
        visited.add(network_id)
        # Vérifier récursivement les réseaux connectés
        for connection in network_dict[network_id]['direct_connect_network']:
            connected_network = connection[0]
            if connected_network not in visited:
                if is_upstream(connected_network, visited):
                    return True
        return False

    for network_id, network_data in network_dict.items():
        direct_connect_network = network_data['direct_connect_network']
        direct_upstream_network = []
        direct_downstream_network = []

        for connection in direct_connect_network:
            connected_network = connection[0]
            if is_upstream(connected_network, set([network_id])):
                direct_upstream_network.append(connection)
            else:
                direct_downstream_network.append(connection)

        network_data['direct_upstream_network'] = direct_upstream_network
        network_data['direct_downstream_network'] = direct_downstream_network
    
    for network_id, network_data in network_dict.items():
        del network_data['direct_connect_network']
    return network_dict

def merge_networks(nets):
    # Créer un réseau vide pour contenir le réseau fusionné
    merged_net = pp.create_empty_network()
    for net in nets:
        merged_net=pp.merge_nets(merged_net, net, validate=False, std_prio_on_net1=True)
    return merged_net
