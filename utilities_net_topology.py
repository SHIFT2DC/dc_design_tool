import pandapower as pp
import pandapower.topology as top
import networkx as nx


def separate_subnetworks(net):
    """
    Separates the network into connected subnetworks.

    Args:
        net (pandapowerNet): The pandapower network to be separated.

    Returns:
        list: A list of pandapower subnetworks.
    """
    # Create a NetworkX graph from the pandapower network
    graph = top.create_nxgraph(net, include_trafos=True)
    
    # Find all connected subnetworks
    subnetworks = top.connected_components(graph)
    
    # Create a list to store individual subnetworks
    subnetwork_list = []
    
    # Iterate over each subnetwork and create a new network for each
    for subnetwork in subnetworks:
        sub_net = pp.select_subnet(net, subnetwork)
        subnetwork_list.append(sub_net)
    
    return subnetwork_list


def sorting_network(net, subnetworks):
    """
    Sorts the subnetworks based on their direct connections.

    Args:
        net (pandapowerNet): The original pandapower network.
        subnetworks (list): List of subnetworks to be sorted.

    Returns:
        dict: A dictionary containing information about upstream and downstream connections.
    """
    dic_of_subs = {}
    
    # Iterate over each subnetwork
    for n in range(len(subnetworks)):
        subn = subnetworks[n]
        
        # Initialize the dictionary for this subnetwork
        dic_of_subs[n] = {'network': subn, 'direct_connect_network': []}
        
        # Check direct connections via converters
        for b in subn.bus.index:
            if b in net.converter.from_bus.values:
                for i, bus_to_find in enumerate(net.converter.loc[net.converter.from_bus == b].to_bus.values):
                    for idx_tmp_sub in range(len(subnetworks)):
                        tmp_sub = subnetworks[idx_tmp_sub]
                        if bus_to_find in tmp_sub.bus.index:
                            dic_of_subs[n]['direct_connect_network'].append([idx_tmp_sub, bus_to_find, net.converter.loc[net.converter.from_bus == b].name.values[i]])
            
            if b in net.converter.to_bus.values:
                for i, bus_to_find in enumerate(net.converter.loc[net.converter.to_bus == b].from_bus.values):
                    for idx_tmp_sub in range(len(subnetworks)):
                        tmp_sub = subnetworks[idx_tmp_sub]
                        if bus_to_find in tmp_sub.bus.index:
                            dic_of_subs[n]['direct_connect_network'].append([idx_tmp_sub, bus_to_find, net.converter.loc[net.converter.to_bus == b].name.values[i]])
    
    # Find upstream and downstream networks
    network_dict = find_upndownstream_networks(dic_of_subs)

    return network_dict


def find_upndownstream_networks(network_dict):
    """
    Determines the upstream and downstream networks for each subnetwork.

    Args:
        network_dict (dict): Dictionary containing information about direct connections.

    Returns:
        dict: Updated dictionary with upstream and downstream networks.
    """
    def is_upstream(network_id, visited):
        """
        Checks if a network is upstream.

        Args:
            network_id (int): The ID of the network.
            visited (set): Set of already visited networks.

        Returns:
            bool: True if the network is upstream, False otherwise.
        """
        # If the network has an ext_grid, it is an upstream network
        if len(network_dict[network_id]['network'].ext_grid.loc[network_dict[network_id]['network'].ext_grid['in_service']==True]) != 0:
            return True
        # Mark the current network as visited
        visited.add(network_id)
        # Recursively check connected networks
        for connection in network_dict[network_id]['direct_connect_network']:
            connected_network = connection[0]
            if connected_network not in visited:
                if is_upstream(connected_network, visited):
                    return True
        return False

    # Iterate over each network to determine upstream and downstream connections
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
    
    # Remove the 'direct_connect_network' key as it is no longer needed
    for network_id, network_data in network_dict.items():
        del network_data['direct_connect_network']
    
    return network_dict


def merge_networks(nets):
    """
    Merges multiple pandapower networks into a single network.

    Args:
        nets (list): List of pandapower networks to be merged.

    Returns:
        pandapowerNet: The merged network.
    """
    # Create an empty network to contain the merged network
    merged_net = pp.create_empty_network()
    
    # Merge each network into the empty network
    for net in nets:
        merged_net = pp.merge_nets(merged_net, net, validate=False, std_prio_on_net1=True)
    
    return merged_net


def find_lines_between_given_line_and_ext_grid(net, line_id):
    """
    Finds the lines between a given line and the external grid.

    Args:
        net (pandapowerNet): The pandapower network.
        line_id (int): The ID of the given line.

    Returns:
        list: List of indices of the lines between the given line and the external grid.
    """
    # Create a NetworkX graph from the pandapower network
    grid_graph = pp.topology.create_nxgraph(net)

    # Find the node connected to the external grid
    ext_grid_node = net.ext_grid.loc[net.ext_grid['in_service'] == True].bus.values[0]

    # Find the nodes connected to the given line
    line = net.line.loc[line_id]
    from_node = line['from_bus']
    to_node = line['to_bus']

    # Find all paths from the line nodes to the external grid node
    paths_from = nx.shortest_path(grid_graph, source=from_node, target=ext_grid_node)
    paths_to = nx.shortest_path(grid_graph, source=to_node, target=ext_grid_node)

    # Choose the shortest path
    if len(paths_from) > len(paths_to):
        short_path = paths_to
    else:
        short_path = paths_from

    # Extract the lines from the shortest path
    lines = [[short_path[i], short_path[i+1]] for i in range(len(short_path)-1)]
    lines_index = []
    
    # Find the indices of the corresponding lines
    for line in lines:
        lines_index.append(net.line.loc[((net.line.from_bus == line[1]) & (net.line.to_bus == line[0])) |
                                        ((net.line.from_bus == line[0]) & (net.line.to_bus == line[1]))].index[0])
    
    return lines_index


def get_bus_distances(net):
    """
    Compute electrical distances from the slack bus using the shortest path.
    Returns a dictionary {bus_id: distance}.
    """
    # Create graph representation of the grid
    grid_graph = pp.topology.create_nxgraph(net)

    # Identify the slack bus
    slack_bus = net.ext_grid.bus.values[0]

    # Compute distances
    distances = nx.single_source_shortest_path_length(grid_graph, slack_bus)

    return distances
