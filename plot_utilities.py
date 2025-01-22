import matplotlib.pyplot as plt
import pandapower.plotting.plotly as pplotly
from pandas import Series
import pandas as pd
import pandapower as pp
import copy

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


def plot_network_with_plotly(net):
    net_plot = copy.deepcopy(net)
    for _,row in net_plot.converter.iterrows():
        if row.type != 'ILC':
            V1=net_plot.bus.vn_kv.loc[net_plot.bus.index==row.from_bus].values[0]
            V2=net_plot.bus.vn_kv.loc[net_plot.bus.index==row.to_bus].values[0]

            if V1>V2:
                pp.create_transformer_from_parameters(net_plot,hv_bus=row.from_bus,lv_bus=row.to_bus,
                                                      sn_mva=row.P,vn_hv_kv=V1,vn_lv_kv=V2,
                                                      vkr_percent=0,vk_percent=0,pfe_kw=0,i0_percent=0)
            else:
                pp.create_transformer_from_parameters(net_plot,hv_bus=row.to_bus,lv_bus=row.from_bus,
                                                      sn_mva=row.P,vn_hv_kv=V2,vn_lv_kv=V1,
                                                      vkr_percent=0,vk_percent=0,pfe_kw=0,i0_percent=0)

    # Créer une figure plotly
    pplotly.create_generic_coordinates(net_plot, mg=None, library='igraph', respect_switches=True, trafo_length_km=0.0000001 ,geodata_table='bus_geodata', buses=None, overwrite=True)
    fig = pplotly.simple_plotly(net_plot)
    line_trace=pplotly.create_line_trace(net_plot,cmap="jet",cmap_vals=net_plot.res_line.loading_percent,width=4.0,
                                         infofunc=(Series(index=net.line.index,
                                                          data=[s1 + s2 for s1, s2 in zip(net.res_line.i_ka.astype(str), net.res_line.loading_percent.apply(lambda x : f'<br> loading  : {x:.1f} % <br>').values)]
                                                          )))
    bus_trace=pplotly.create_bus_trace(net_plot,cmap="jet_r",cmap_vals=net_plot.res_bus.vm_pu, size=10,
                                       infofunc=(Series(index=net.bus.index,
                                                          data=[s1 + s2 for s1, s2 in zip(net.bus.index.astype(str), net.res_bus.vm_pu.apply(lambda x : f'<br> V : {x:.3f} <br>').values)]
                                                          )))
    
    trafo_trace=pplotly.create_trafo_trace(net_plot,color='black',width=15)
    fig = pplotly.draw_traces(line_trace+trafo_trace+bus_trace,
                              figsize=2, aspectratio=(20, 10),
                              filename='NetworkEcartLEPlot.html', auto_open=True,showlegend=False)
    fig.show()