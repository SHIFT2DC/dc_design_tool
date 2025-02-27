import matplotlib.pyplot as plt
import pandapower.plotting.plotly as pplotly
from pandas import Series
import pandas as pd
import pandapower as pp
import copy
import plotly.graph_objects as go
import plotly.express as px


def plot_voltage(net):
    """
    Plots the voltage profiles of the buses in the given pandapower network.

    Parameters:
    net (pandapowerNet): The pandapower network object.
    """
    bus_indices = net.bus.index
    bus_voltages = net.res_bus.vm_pu

    plt.figure(figsize=(10, 6))
    plt.plot(bus_indices, bus_voltages, marker='o', linestyle='None', color='b', label="Voltage (p.u.)")

    plt.title("Voltage Profiles")
    plt.xlabel("Bus Index")
    plt.ylabel("Voltage (p.u.)")
    plt.grid(True)

    # Ensure x-axis limits cover all bus indices
    plt.xlim(min(bus_indices) - 1, max(bus_indices) + 1)  # Add some margin

    # Set dynamic y-axis limits with a fixed margin
    margin = 0.02  # Add a 2% margin for better visualization
    plt.ylim(min(bus_voltages) - margin, max(bus_voltages) + margin)

    # Ensure x-ticks match bus indices and rotate if necessary
    plt.xticks(ticks=bus_indices, labels=bus_indices, rotation=45)

    plt.legend()
    plt.show()


def plot_network_evaluation_results_with_plotly(net, node_data, file_name):
    net_plot = copy.deepcopy(net)
    # Add equivalent transformers for visualization
    for _, row in net_plot.converter.iterrows():
        if row.type != 'ILC':
            v1 = net_plot.bus.vn_kv.loc[net_plot.bus.index == row.from_bus].values[0]
            v2 = net_plot.bus.vn_kv.loc[net_plot.bus.index == row.to_bus].values[0]

            if v1 > v2:
                pp.create_transformer_from_parameters(
                    net_plot,
                    hv_bus=row.from_bus,
                    lv_bus=row.to_bus,
                    sn_mva=row.P,
                    vn_hv_kv=v1,
                    vn_lv_kv=v2,
                    vkr_percent=0,
                    vk_percent=0,
                    pfe_kw=0,
                    i0_percent=0
                )
            else:
                pp.create_transformer_from_parameters(
                    net_plot,
                    hv_bus=row.to_bus,
                    lv_bus=row.from_bus,
                    sn_mva=row.P,
                    vn_hv_kv=v2,
                    vn_lv_kv=v1,
                    vkr_percent=0,
                    vk_percent=0,
                    pfe_kw=0,
                    i0_percent=0
                )

    # Create coordinates
    pplotly.create_generic_coordinates(
        net_plot,
        mg=None,
        library='igraph',
        respect_switches=True,
        trafo_length_km=0.0000001,
        geodata_table='bus_geodata',
        buses=None,
        overwrite=True
    )

    # Create base figure
    fig = pplotly.simple_plotly(net_plot, auto_open=False)

    # Line trace
    line_trace = pplotly.create_line_trace(
        net_plot,
        cmap="jet",
        cmap_vals=net_plot.res_line.loading_percent,
        width=4.0,
        cbar_title="Line Loading (%)",
        cmin=5,
        cmax=100,
        infofunc=(Series(index=net.line.index,
                         data=[f'Line from {net.line.loc[i, "from_bus"]} bus to bus {net.line.loc[i, "to_bus"]} <br>'
                               f'Section: {net.line.loc[i, "section"]} mm² <br>'
                               f'Cable rank: {net.line.loc[i, "cable_rank"]} <br>'
                               f'Current: {row.i_ka * 1000:.1f} A <br>'
                               f'Power from bus {net.line.loc[i, "from_bus"]}: {row.p_from_mw * 1000:.3f} kW <br>'
                               f'Power to bus {net.line.loc[i, "to_bus"]}: {row.p_to_mw * 1000:.3f} kW <br>'
                               f'Losses: {row.pl_mw * 1000:.3f} kW <br>'
                               f'Loading: {row.loading_percent:.1f} % '
                               for i, row in net.res_line.iterrows()]))
    )

    # Bus trace
    bus_trace = pplotly.create_bus_trace(
        net_plot,
        cmap="plasma_r",
        cmap_vals=net_plot.res_bus.vm_pu,
        size=10,
        cbar_title="Bus Voltage (p.u.)",
        cmin=0.9,
        cmax=1.1,
        infofunc=(Series(index=net.bus.index,
                         data=[f'Bus {s1} <br>'
                               f'Voltage: {s2:.3f} pu <br>'
                               f'Power: {net.res_bus.p_mw.loc[int(s1)]*1000:.3f} kW'
                               for s1, s2 in zip(net.bus.index.astype(str), net.res_bus.vm_pu)]))
    )

    # Conv trace
    trafo_trace = pplotly.create_trafo_trace(
        net_plot,
        color='black',
        width=15,
        infofunc=(Series(index=net.converter.index,
                         data=[f'Converter {net.converter.loc[i, "name"]} from bus {net.converter.loc[i, "from_bus"]} to bus {net.converter.loc[i, "to_bus"]} <br>'
                               f'Installed Power: {net.converter.loc[i, "P"]*1000:.1f} kW <br>'
                               f'Conv rank: {net.converter.loc[i, "conv_rank"]} <br>'
                               f'Power: {row.p_mw * 1000:.3f} kW <br>'
                               f'Losses: {row.pl_mw * 1000:.3f} kW <br>'
                               f'Loading: {net.res_converter.loc[i, "loading (%)"]:.1f} %'
                               for i, row in net.res_converter.iterrows()]))
    )

    # Add text for bus numbers & asset types
    bus_text = go.Scatter(
        x=[net_plot.bus_geodata.x.loc[i] for i in net_plot.bus.index],
        y=[net_plot.bus_geodata.y.loc[i]+0.1 for i in net_plot.bus.index],
        mode="text",
        text=[
            f"{i} ({node_data.loc[node_data['Node number'] == i, 'Component type'].str.replace(' ', '').str.lower().values[0]})"
            if i in node_data['Node number'].values else str(i)
            for i in net_plot.bus.index
        ],
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
    fig = pplotly.draw_traces(
        line_trace + trafo_trace + bus_trace + [bus_text] + [line_text] + [converter_text],
        figsize=2,
        aspectratio=(20, 10),
        filename=file_name,
        auto_open=False,
        showlegend=False
    )

    fig.show()


def plot_network_sizing_results_with_plotly(net, node_data, file_name):
    net_plot = copy.deepcopy(net)
    # Add equivalent transformers for visualization
    for _, row in net_plot.converter.iterrows():
        if row.type != 'ILC':
            v1 = net_plot.bus.vn_kv.loc[net_plot.bus.index == row.from_bus].values[0]
            v2 = net_plot.bus.vn_kv.loc[net_plot.bus.index == row.to_bus].values[0]

            if v1 > v2:
                pp.create_transformer_from_parameters(
                    net_plot,
                    hv_bus=row.from_bus,
                    lv_bus=row.to_bus,
                    sn_mva=row.P,
                    vn_hv_kv=v1,
                    vn_lv_kv=v2,
                    vkr_percent=0,
                    vk_percent=0,
                    pfe_kw=0,
                    i0_percent=0
                )
            else:
                pp.create_transformer_from_parameters(
                    net_plot,
                    hv_bus=row.to_bus,
                    lv_bus=row.from_bus,
                    sn_mva=row.P,
                    vn_hv_kv=v2,
                    vn_lv_kv=v1,
                    vkr_percent=0,
                    vk_percent=0,
                    pfe_kw=0,
                    i0_percent=0
                )

    # Create coordinates
    pplotly.create_generic_coordinates(
        net_plot,
        mg=None,
        library='igraph',
        respect_switches=True,
        trafo_length_km=0.0000001,
        geodata_table='bus_geodata',
        buses=None,
        overwrite=True
    )

    # Create base figure
    fig = pplotly.simple_plotly(net_plot, auto_open=False)

    # Line trace
    line_trace = pplotly.create_line_trace(
        net_plot,
        # cmap="jet",
        # cmap_vals=net_plot.line.section,
        color='brown',
        width=4.0,
        # cbar_title="Line section (mm²)",
        infofunc=(Series(index=net.line.index,
                         data=[f'Line from {net.line.loc[i, "from_bus"]} bus to bus {net.line.loc[i, "to_bus"]} <br>'
                               f'Section: {net.line.loc[i, "section"]} mm²'
                               for i, row in net.res_line.iterrows()]))
    )

    # Bus trace
    bus_trace = pplotly.create_bus_trace(
        net_plot,
        color='blue',
        size=10,
        infofunc=(Series(index=net.bus.index,
                         data=[f'Bus {s1}'
                               for s1, s2 in zip(net.bus.index.astype(str), net.res_bus.vm_pu)]))
    )

    # Conv trace
    trafo_trace = pplotly.create_trafo_trace(
        net_plot,
        color='black',
        width=15,
        infofunc=(Series(index=net.converter.index,
                         data=[f'Converter {net.converter.loc[i, "name"]} from bus {net.converter.loc[i, "from_bus"]} to bus {net.converter.loc[i, "to_bus"]} <br>'
                               f'Installed Power: {net.converter.loc[i, "P"]*1000:.1f} kW'
                               for i, row in net.res_converter.iterrows()]))
    )
    # Rename legend
    for trace in trafo_trace:
        trace.update(name="converters")

    # Add text for bus numbers & asset types
    bus_text = go.Scatter(
        x=[net_plot.bus_geodata.x.loc[i] for i in net_plot.bus.index],
        y=[net_plot.bus_geodata.y.loc[i]+0.1 for i in net_plot.bus.index],
        mode="text",
        text=[
            f"{i} ({node_data.loc[node_data['Node number'] == i, 'Component type'].str.replace(' ', '').str.lower().values[0]})"
            if i in node_data['Node number'].values else str(i)
            for i in net_plot.bus.index
        ],
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
    fig = pplotly.draw_traces(
        line_trace + trafo_trace + bus_trace + [bus_text] + [line_text] + [converter_text],
        figsize=2,
        aspectratio=(20, 10),
        filename=file_name,
        auto_open=True,
        showlegend=True
    )

    fig.show()


def save_sizing_results_to_excel(net, node_data, file_name):
    # Create the "line sizing" sheet
    line_sizing_data = {
        "Line Index": net.line.index,
        "From Bus": net.line['from_bus'],
        "To Bus": net.line['to_bus'],
        "Section (mm²)": net.line['section']
    }
    line_sizing_df = pd.DataFrame(line_sizing_data)

    # Create the "converter sizing" sheet
    converter_sizing_data = {
        "Converter Index": net.converter.index,
        "Converter Name": net.converter['name'],
        "From Bus": net.converter['from_bus'],
        "To Bus": net.converter['to_bus'],
        "Nominal Power Installed (kW)": net.converter['P'] * 1000  # Convert to kW
    }
    converter_sizing_df = pd.DataFrame(converter_sizing_data)

    # Save both DataFrames to an Excel file
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        line_sizing_df.to_excel(writer, sheet_name='Line Sizing', index=False)
        converter_sizing_df.to_excel(writer, sheet_name='Converter Sizing', index=False)

    print(f"Sizing results saved to {file_name}")


def plot_bus_voltage_heatmap(net, scenario_name):
    bus_indices = list(map(str, net.bus.index))  # Convert to strings for categorical x-axis

    fig = px.bar(
        x=bus_indices,
        y=net.res_bus.vm_pu,
        color=net.res_bus.vm_pu,
        labels={'x': 'Bus Index', 'y': 'Voltage (p.u.)'},
        title=rf'Bus Voltage Levels in scenario {scenario_name.split(" ")[3]}',
    )

    # Force all x-ticks to be displayed
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(bus_indices))),
            ticktext=bus_indices,
            showticklabels=True
        ),
        bargap=0.1
    )
    fig.show()
    fig.write_html(rf'output_voltage_bars_scenario_{scenario_name.split(" ")[3]}.html')
