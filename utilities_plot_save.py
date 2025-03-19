import matplotlib.pyplot as plt
import pandapower.plotting.plotly as pplotly
from pandas import Series
import pandas as pd
import pandapower as pp
import copy
import plotly.graph_objects as go
import plotly.express as px
import os

# Define the output directory
output_dir = "output"
# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


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
                               f'Length: {net.line.loc[i, "length_km"]*1000} m <br>'
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
                               f'Length: {net.line.loc[i, "length_km"]*1000} m <br>'
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
        auto_open=False,
        showlegend=True
    )

    fig.show()


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
    fig.write_html(os.path.join(output_dir, rf'output_voltage_bars_scenario_{scenario_name.split(" ")[3]}.html'))


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


def plot_efficiency_kpi(efficiency_results):
    labels = ['Efficiency Ratio', 'Total Consumed Energy (MWh)', 'Total Generated Energy (MWh)',
              'Total Losses in Cables (MWh)', 'Total Losses in Converters (MWh)']
    values = [efficiency_results[0], efficiency_results[1], efficiency_results[2],
              efficiency_results[3], efficiency_results[4]]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot the quantities on the primary y-axis
    ax1.bar(labels[1:], values[1:], color=['green', 'orange', 'red', 'purple'])
    # ax1.set_xlabel('KPI Type')
    ax1.set_ylabel('Energy (MWh)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Rotate the xticks to 45 degrees
    plt.xticks(rotation=45, ha='right')

    # Create a secondary y-axis for the Efficiency Ratio
    ax2 = ax1.twinx()
    ax2.bar(labels[:1], [values[0]], color='blue', alpha=0.6)  # Bar for Efficiency Ratio
    ax2.set_ylabel('Efficiency Ratio', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Adding title and adjusting layout
    plt.title('Efficiency KPI')
    plt.tight_layout()

    # Save or show the plot
    plt.savefig(os.path.join(output_dir, 'output_kpis_results_efficiency.png'))
    plt.close()


def plot_economic_kpi(economic_results):
    # Extract CAPEX details
    total_capex_keur = economic_results[0]
    capex_details = economic_results[1]
    converters_capex = capex_details['Converters CAPEX (kEUR)']
    cables_capex = capex_details['Cables CAPEX (kEUR)']

    # Data for pie chart
    labels = ['Converters CAPEX', 'Cables CAPEX']
    sizes = [converters_capex, cables_capex]
    colors = ['lightblue', 'lightgreen']

    # Create a pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'})
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Add title
    plt.title(f"CAPEX Distribution (Total CAPEX: {total_capex_keur} kEUR)")

    # Show or save the plot
    plt.savefig(os.path.join(output_dir, 'output_kpis_results_economic.png'))
    plt.close()


def plot_environmental_kpi(environmental_results):
    labels = ['Total Weight (kg)', 'Total Lifecycle Emissions (kg CO2)']
    values = [environmental_results[0], environmental_results[2]]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['brown', 'green'])
    plt.title('Environmental KPI')
    plt.xlabel('KPI Type')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'output_kpis_results_environmental.png'))
    plt.close()


def save_kpis_results_to_excel(file_path, efficiency_results, economic_results, environmental_results):
    # Plot each KPI type
    # plot_efficiency_kpi(efficiency_results)
    # plot_economic_kpi(economic_results)
    # plot_environmental_kpi(environmental_results)

    # Helper function to replace None with "Not Available"
    def safe_value(value):
        return value if value is not None else "Not Available"

    # Determine the correct label for energy savings
    energy_savings_mwh = efficiency_results[5]
    if energy_savings_mwh is not None:
        energy_savings_label_mwh = 'Energy Savings (DC more efficient) (MWh)' if energy_savings_mwh >= 0 else 'Extra Energy (AC more efficient) (MWh)'
        energy_savings_label_percent = 'Energy Savings (DC more efficient) (%)' if energy_savings_mwh >= 0 else 'Extra Energy (AC more efficient) (%)'
    else:
        energy_savings_label_mwh = 'Energy Savings / Extra Energy (MWh)'
        energy_savings_label_percent = 'Energy Savings / Extra Energy (%)'

    # Efficiency KPI results
    efficiency_df = pd.DataFrame({
        'Efficiency Ratio': [efficiency_results[0]],
        'Total Consumed Energy (MWh)': [efficiency_results[1]],
        'Total Generated Energy (MWh)': [efficiency_results[2]],
        'Total Losses in Cables (MWh)': [efficiency_results[3]],
        'Total Losses in Converters (MWh)': [efficiency_results[4]],
        energy_savings_label_mwh: [safe_value(efficiency_results[5])],
        energy_savings_label_percent: [safe_value(efficiency_results[6])]
    })

    # Determine the correct label for CAPEX difference
    capex_difference_keur = economic_results[2]
    if capex_difference_keur is not None:
        capex_difference_label_keur = 'CAPEX Savings (AC more expensive) (KEUR)' if capex_difference_keur >= 0 else 'Extra CAPEX (DC more expensive) (KEUR)'
        capex_difference_label_percent = 'CAPEX Savings (AC more expensive) (%)' if capex_difference_keur >= 0 else 'Extra CAPEX (DC more expensive) (%)'
    else:
        capex_difference_label_keur = 'CAPEX Savings / Extra CAPEX (KEUR)'
        capex_difference_label_percent = 'CAPEX Savings / Extra CAPEX (%)'

    # Economic KPI results - split Capex Details into separate columns
    capex_details = economic_results[1]
    opex_details = economic_results[5]
    economic_df = pd.DataFrame({
        'Total CAPEX (KEUR)': [economic_results[0]],
        capex_difference_label_keur: [safe_value(economic_results[2])],
        capex_difference_label_percent: [safe_value(economic_results[3])],
        'Converters CAPEX (KEUR)': [capex_details['Converters CAPEX (kEUR)']],
        'Cables CAPEX (KEUR)': [capex_details['Cables CAPEX (kEUR)']],
        'Total OPEX (KEUR)': [economic_results[4]],
        'Converters OPEX (KEUR)': [opex_details['Converters OPEX (kEUR)']],
        'Cables OPEX (KEUR)': [opex_details['Cables OPEX (kEUR)']],
    })

    # Add individual converter details (each converter in a separate column)
    for converter, cost in capex_details['Details']['Converters'].items():
        economic_df[f'Converter {converter} CAPEX (KEUR)'] = [cost]

    # Add individual cable line details (each cable line in a separate column)
    for cable_line, cost in capex_details['Details']['Cables'].items():
        economic_df[f'Cable {cable_line} CAPEX (KEUR)'] = [cost]

    # Environmental KPI results - split Weight Details into separate columns
    weight_details = environmental_results[1]
    environmental_df = pd.DataFrame({
        'Total Weight (kg)': [environmental_results[0]],
        'Total Lifecycle Emissions (kg CO2)': [environmental_results[2]]
    })

    # Add individual converter details (each converter in a separate column)
    for converter, weight in weight_details['Details']['Converters'].items():
        environmental_df[f'Converter {converter} Weight (kg)'] = [weight]

    # Add individual cable line details (each converter in a separate column)
    for cable_line, weight in weight_details['Details']['Cables'].items():
        environmental_df[f'Cable {cable_line} Weight (kg)'] = [weight]

    # Create an Excel writer
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        # Save each dataframe to a different sheet
        efficiency_df.to_excel(writer, sheet_name='Efficiency', index=False)
        economic_df.to_excel(writer, sheet_name='Economic', index=False)
        environmental_df.to_excel(writer, sheet_name='Environmental', index=False)

    print(f"KPIs have been saved to {file_path}")



