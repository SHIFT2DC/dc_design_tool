import matplotlib.pyplot as plt
import pandapower.plotting.plotly as pplotly
from pandas import Series
import pandas as pd
import pandapower as pp
import copy
import plotly.graph_objects as go


def plot_voltage(net):
    """
    Plots the voltage profiles of the buses in the given pandapower network.

    Parameters:
    net (pandapowerNet): The pandapower network object.
    """
    # Print a message indicating the start of the plotting process
    print(">Plot voltage profiles")
    
    # Adjust bus indices to start from 1 for better readability
    bus_indices = net.res_bus.index + 1
    
    # Extract the voltage magnitudes in per unit (pu) from the network results
    bus_voltages = net.res_bus.vm_pu
    
    # Enable interactive mode for plotting
    plt.ion()
    
    # Create a new figure with specified size
    plt.figure(figsize=(10, 6))
    
    # Plot the voltage values with markers and lines
    plt.plot(range(len(bus_voltages.values)), bus_voltages.values, marker='o', linestyle='-', color='b')
    
    # Set the title and labels for the plot
    plt.title("Voltage Profiles")
    plt.xlabel("Bus Index")
    plt.ylabel("Voltage (pu)")
    
    # Enable grid for better visualization
    plt.grid(True)
    
    # Calculate the spread of voltage values to set y-axis limits dynamically
    spread = max(bus_voltages) - min(bus_voltages)
    plt.ylim(min(bus_voltages) - 0.1 * spread, max(bus_voltages) + 0.1 * spread)
    
    # Set x-axis ticks and labels to match bus indices
    plt.xticks(ticks=range(len(bus_voltages.values)), labels=bus_indices)
    
    # Display the plot
    plt.show()


def plot_network_with_plotly(net, file_name):
    net_plot = copy.deepcopy(net)
    # Add equivalent transformers for visualization
    for _, row in net_plot.converter.iterrows():
        if row.type != 'ILC':
            V1 = net_plot.bus.vn_kv.loc[net_plot.bus.index == row.from_bus].values[0]
            V2 = net_plot.bus.vn_kv.loc[net_plot.bus.index == row.to_bus].values[0]

            if V1 > V2:
                pp.create_transformer_from_parameters(net_plot, hv_bus=row.from_bus, lv_bus=row.to_bus,
                                                      sn_mva=row.P, vn_hv_kv=V1, vn_lv_kv=V2,
                                                      vkr_percent=0, vk_percent=0, pfe_kw=0,i0_percent=0)
            else:
                pp.create_transformer_from_parameters(net_plot, hv_bus=row.to_bus, lv_bus=row.from_bus,
                                                      sn_mva=row.P, vn_hv_kv=V2, vn_lv_kv=V1,
                                                      vkr_percent=0, vk_percent=0, pfe_kw=0, i0_percent=0)

    # Create coordinates
    pplotly.create_generic_coordinates(net_plot, mg=None, library='igraph', respect_switches=True,
                                       trafo_length_km=0.0000001, geodata_table='bus_geodata', buses=None, overwrite=True)

    # Create base figure
    fig = pplotly.simple_plotly(net_plot, auto_open=False)

    # Line trace
    '''line_trace = pplotly.create_line_trace(net_plot, cmap="jet", cmap_vals=net_plot.res_line.loading_percent, width=4.0,
                                           infofunc=(Series(index=net.line.index,
                                                            data=[f'I : {row.i_ka*1000:.1f} A <br>loading : {row.loading_percent:.1f} % <br> cable_rank : {net.line.loc[i,"cable_rank"]} % <br> section : {net.line.loc[i,"section"]}' for i, row in net.res_line.iterrows()]
                                                            )))'''
    line_trace = pplotly.create_line_trace(net_plot, cmap="jet", cmap_vals=net_plot.res_line.loading_percent, width=4.0,
                                           infofunc=(Series(index=net.line.index,
                                                            data=[f'Line from {net.line.loc[i, "from_bus"]} bus to bus {net.line.loc[i, "to_bus"]} <br>'
                                                                  f'Section: {net.line.loc[i, "section"]} mm² <br>'
                                                                  f'Cable rank: {net.line.loc[i, "cable_rank"]} <br>'
                                                                  f'Current: {row.i_ka * 1000:.1f} A <br>'
                                                                  f'Power from bus {net.line.loc[i, "from_bus"]}: {row.p_from_mw * 1000:.3f} kW <br>'
                                                                  f'Power to bus {net.line.loc[i, "to_bus"]}: {row.p_to_mw * 1000:.3f} kW <br>'
                                                                  f'Losses: {row.pl_mw * 1000:.5f} kW <br>'
                                                                  f'Loading: {row.loading_percent:.1f} % '
                                                                  for i, row in net.res_line.iterrows()])))
    '''bus_trace = pplotly.create_bus_trace(net_plot, cmap="jet_r", cmap_vals=net_plot.res_bus.vm_pu, size=10,
                                         infofunc=(Series(index=net.bus.index,
                                                          data=[s1 + s2 for s1, s2 in zip(net.bus.index.astype(str), net.res_bus.vm_pu.apply(lambda x: f'<br> V : {x:.3f} <br>').values)]
                                                          )))'''
    bus_trace = pplotly.create_bus_trace(net_plot, cmap="jet_r", cmap_vals=net_plot.res_bus.vm_pu, size=10,
                                         infofunc=(Series(index=net.bus.index,
                                                          data=[f'Bus {s1} <br>'
                                                                f'Voltage: {s2:.3f} pu <br>'
                                                                f'Power: {net.res_bus.p_mw.loc[int(s1)]*1000:.2f} kW'
                                                                for s1, s2 in
                                                                zip(net.bus.index.astype(str), net.res_bus.vm_pu)])))
    ''' trafo_trace = pplotly.create_trafo_trace(net_plot, color='black', width=15, infofunc=(Series(index=net.converter.index,
                                             data=[f'P : {row.p_mw*1000:.1f} kW <br>loading : {net.res_converter.loc[i,"loading (%)"]:.1f} % <br> conv_rank : {net.converter.loc[i,"conv_rank"]}' for i, row in net.res_converter.iterrows()]
                                                          )))'''
    trafo_trace = pplotly.create_trafo_trace(net_plot, color='black', width=15,
                                             infofunc=(Series(index=net.converter.index,
                                                              data=[f'Converter {net.converter.loc[i, "name"]} from bus {net.converter.loc[i, "from_bus"]} to bus {net.converter.loc[i, "to_bus"]} <br>'
                                                                    f'Installed Power: {net.converter.loc[i, "P"]*1000:.1f} kW <br>'
                                                                    f'Conv rank: {net.converter.loc[i, "conv_rank"]} <br>'
                                                                    f'Power: {row.p_mw * 1000:.1f} kW <br>'
                                                                    f'Losses: {row.pl_mw * 1000:.1f} kW <br>'
                                                                    f'Loading: {net.res_converter.loc[i, "loading (%)"]:.1f} %'
                                                                    for i, row in net.res_converter.iterrows()])))

    # Add text for bus numbers
    bus_text = go.Scatter(
        x=[net_plot.bus_geodata.x.loc[i] for i in net_plot.bus.index],
        y=[net_plot.bus_geodata.y.loc[i]+0.1 for i in net_plot.bus.index],
        mode="text",
        text=[str(i) for i in net_plot.bus.index],
        textposition="top right",
        showlegend=False
    )

    # Add text for line sections
    line_text = go.Scatter(
        x=[(net_plot.bus_geodata.x.loc[net.line.loc[i, "from_bus"]] + net_plot.bus_geodata.x.loc[
            net.line.loc[i, "to_bus"]]) / 2 for i in net.line.index],
        y=[(net_plot.bus_geodata.y.loc[net.line.loc[i, "from_bus"]] + net_plot.bus_geodata.y.loc[
            net.line.loc[i, "to_bus"]]) / 2 + 0.1 for i in net.line.index],
        mode="text",
        text=[f'{net.line.loc[i, "section"]} mm²' for i in net.line.index],
        textposition="top center",
        showlegend=False
    )

    # Add text for converter installed power
    converter_text = go.Scatter(
        x=[(net_plot.bus_geodata.x.loc[net.converter.loc[i, "from_bus"]] + net_plot.bus_geodata.x.loc[
            net.converter.loc[i, "to_bus"]]) / 2 for i in net.converter.index],
        y=[(net_plot.bus_geodata.y.loc[net.converter.loc[i, "from_bus"]] + net_plot.bus_geodata.y.loc[
            net.converter.loc[i, "to_bus"]]) / 2 + 0.2 for i in net.converter.index],
        mode="text",
        text=[f'{net.converter.loc[i, "P"] * 1000:.1f} kW' for i in net.converter.index],
        textposition="bottom center",
        showlegend=False
    )

    # Draw plot with legend
    fig = pplotly.draw_traces(line_trace + trafo_trace + bus_trace + [bus_text] + [line_text] + [converter_text],
                              figsize=2, aspectratio=(20, 10),
                              filename=file_name, auto_open=False, showlegend=False)

    fig.show()
