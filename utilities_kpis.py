import pandapower as pp
import numpy as np
import pandas as pd


def calculate_efficiency_kpi(net_snapshots, timestep_hours):

    # Calculate efficiency ration of the network
    efficiency_ratio, total_consumed_energy_mwh, total_generated_energy_mwh = calculate_efficiency_ratio(
        net_snapshots, timestep_hours)

    # Calculate losses in cables and converters
    total_losses_cables_mwh, total_losses_converters_mwh = calculate_losses(net_snapshots, timestep_hours)

    return (efficiency_ratio, total_consumed_energy_mwh, total_generated_energy_mwh,
            total_losses_cables_mwh, total_losses_converters_mwh)


def calculate_efficiency_ratio(net_snapshots, timestep_hours):
    total_generated_energy_mwh = 0
    total_consumed_energy_mwh = 0

    for net in net_snapshots:
        # Sum of generated power from sgens (e.g., PV) and discharging storage (negative for discharging)
        total_generated_energy_mwh += net.res_sgen["p_mw"].sum() * timestep_hours
        total_generated_energy_mwh += abs(net.res_storage[net.res_storage["p_mw"] < 0]["p_mw"].sum()) * timestep_hours

        # Sum of consumed power from loads and charging storage (positive for charging)
        total_consumed_energy_mwh += net.res_load["p_mw"].sum() * timestep_hours
        total_consumed_energy_mwh += net.res_storage[net.res_storage["p_mw"] > 0]["p_mw"].sum() * timestep_hours

        # Include slack bus power: if positive --> supplying power (generation), if negative --> consuming power (load)
        if "res_ext_grid" in net:
            slack_power = net.res_ext_grid["p_mw"].sum() * timestep_hours  # Power exchange with external grid
            if slack_power > 0:
                total_generated_energy_mwh += slack_power
            else:
                total_consumed_energy_mwh += abs(slack_power)

    efficiency_ratio = total_consumed_energy_mwh / total_generated_energy_mwh if total_generated_energy_mwh > 0 else 0
    return efficiency_ratio*100, total_consumed_energy_mwh, total_generated_energy_mwh


def calculate_losses(net_snapshots, timestep_hours):
    total_losses_cables_mwh = 0
    total_losses_converters_mwh = 0

    for net in net_snapshots:
        total_losses_cables_mwh += net.res_line["pl_mw"].sum() * timestep_hours
        total_losses_converters_mwh += net.res_converter["pl_mw"].sum() * timestep_hours

    return total_losses_cables_mwh, total_losses_converters_mwh


def calculate_total_investment_cost_cables_converters_kpi(net, use_case, path_converter_catalogue,
                                                          default_cost_converter_per_kw=2,
                                                          default_cost_cable_al_per_m=0.001,
                                                          default_cost_cable_cu_per_m=0.002):

    # --- Compute Converter CAPEX ---
    total_converter_cost_keur, capex_converters_keur_dict = (
        calculate_investment_cost_converters(net, use_case, path_converter_catalogue, default_cost_converter_per_kw))

    # --- Compute Cable CAPEX ---
    total_cable_cost_keur, capex_cables_keur_dict = (
        calculate_investment_cost_cables(net, use_case, default_cost_cable_al_per_m, default_cost_cable_cu_per_m))

    # --- Compute Total CAPEX ---
    total_capex_keur = total_converter_cost_keur + total_cable_cost_keur

    capex_details = {
        "Total CAPEX (kEUR)": total_capex_keur,
        "Converters CAPEX (kEUR)": total_converter_cost_keur,
        "Cables CAPEX (kEUR)": total_cable_cost_keur,
        "Details": {
            "Converters": capex_converters_keur_dict,
            "Cables": capex_cables_keur_dict
        }
    }

    return total_capex_keur, capex_details


def calculate_investment_cost_converters(net, use_case, path_converter_catalogue, default_cost_per_kw=2):
    """
    Calculate the total investment cost for converters in the network.

    Parameters:
        net (pandapowerNet): The DC network model.
        use_case (dict): Dictionary containing UC definition data.
        path_converter_catalogue (str): Path to the converter catalogue file.
        default_cost_per_kw (float): Default cost per kW in kEUR/kW if no matching converter is found.

    Returns:
        total_converter_cost_keur (float): Total investment cost in kEUR.
        capex_converters_keur_dict (dict): Dictionary storing capex for each converter.
    """
    total_converter_cost_keur = 0
    capex_converters_keur_dict = {}

    # Read and filter the converter catalogue
    converter_catalogue = pd.ExcelFile(path_converter_catalogue).parse('Converters')
    converter_catalogue = converter_catalogue.loc[
        converter_catalogue['Ecosystem'] == use_case['Project details']['Ecosystem']
        ]

    for index, row in net.converter.iterrows():
        converter_name = row["name"]
        nominal_power_kw = row["P"] * 1000  # Convert MW to kW
        converter_type = row["type"]

        tmp_catalogue = row["converter_catalogue"]
        converter_cost = None

        # If tmp_catalogue is a DataFrame, check if it's empty
        if isinstance(tmp_catalogue, pd.DataFrame):
            if not tmp_catalogue.empty:
                # Filter the catalogue for the installed converter
                matching_row = tmp_catalogue.loc[tmp_catalogue['Nominal power (kW)'] == nominal_power_kw]
                if not matching_row.empty:
                    converter_cost = matching_row['Approximate cost (kEUR)'].values[0]

        if converter_cost is None:  # If not found, search in the global converter catalogue
            matching_row = converter_catalogue[
                (converter_catalogue['Converter type'] == converter_type) &
                (converter_catalogue['Nominal power (kW)'] == nominal_power_kw)
                ]
            if not matching_row.empty:
                converter_cost = matching_row['Approximate cost (kEUR)'].values[0]

        if converter_cost is None:  # If still not found, use the default cost
            converter_cost = nominal_power_kw * default_cost_per_kw

        capex_converters_keur_dict[converter_name] = converter_cost
        total_converter_cost_keur += converter_cost

    return total_converter_cost_keur, capex_converters_keur_dict


def calculate_investment_cost_cables(net, use_case, default_cost_cable_al_per_m=0.001,
                                     default_cost_cable_cu_per_m=0.002):
    """
    Calculate the total investment cost for cables in the network.

    Parameters:
        net (pandapowerNet): The DC network model.
        use_case (dict): Dictionary containing UC definition data.
        default_cost_cable_al_per_m (float): Default cost per meter for aluminum cables in kEUR/m.
        default_cost_cable_cu_per_m (float): Default cost per meter for copper cables in kEUR/m.

    Returns:
        total_cable_cost_keur (float): Total investment cost for cables in kEUR.
        capex_cables_keur_dict (dict): Dictionary storing capex for each cable.
    """
    total_cable_cost_keur = 0
    capex_cables_keur_dict = {}

    cable_info = use_case['Conductor parameters']
    cable_type = cable_info['Material ']  # Default to Aluminum if type is not specified

    for index, row in net.line.iterrows():
        cable_name = f"line {row['from_bus']} - {row['to_bus']}"
        cable_length_m = row["length_km"] * 1000

        # Determine cost per meter based on material type
        if "Copper" in cable_type:
            cost_per_m = default_cost_cable_cu_per_m
        else:
            cost_per_m = default_cost_cable_al_per_m

        # Calculate cost
        cable_cost = cable_length_m * cost_per_m
        capex_cables_keur_dict[cable_name] = cable_cost
        total_cable_cost_keur += cable_cost

    return total_cable_cost_keur, capex_cables_keur_dict


def calculate_total_weight_cables_converters_kpi(net, use_case, path_converter_catalogue,
                                                 default_weight_converter_kg_per_kw=1,
                                                 default_weight_cable_al_per_m=1,
                                                 default_weight_cable_cu_per_m=2):

    # --- Compute Converter weight ---
    total_weight_converter_kg, weight_converters_kg_dict = (
        calculate_weight_converters(net, use_case, path_converter_catalogue, default_weight_converter_kg_per_kw))

    # --- Compute Cable weight ---
    total_weight_cable_kg, weight_cables_kg_dict = (
        calculate_weight_cables(net, use_case, default_weight_cable_al_per_m, default_weight_cable_cu_per_m))

    # --- Compute Total weight ---
    total_weight_kg = total_weight_converter_kg + total_weight_cable_kg

    weight_details = {
        "Total Weight (kg)": total_weight_kg,
        "Converters Weight (kg)": total_weight_converter_kg,
        "Cables Weight (kg)": total_weight_cable_kg,
        "Details": {
            "Converters": weight_converters_kg_dict,
            "Cables": weight_cables_kg_dict
        }
    }

    return total_weight_kg, weight_details


def calculate_weight_converters(net, use_case, path_converter_catalogue, default_weight_converter_kg_per_kw=1):
    """
    Calculate the total weight of converters in the network.

    Parameters:
        net (pandapowerNet): The DC network model.
        use_case (dict): Dictionary containing UC definition data.
        path_converter_catalogue (str): Path to the converter catalogue file.
        default_weight_converter_kg_per_kw (float): Default weight per kW if no catalogue data is available.

    Returns:
        total_weight_converter_kg (float): Total weight of converters in kg.
        weight_converters_kg_dict (dict): Dictionary with individual converter weights.
    """

    weight_converters_kg_dict = {}
    total_weight_converter_kg = 0

    # Read and filter the converter catalogue
    converter_catalogue = pd.ExcelFile(path_converter_catalogue).parse('Converters')
    converter_catalogue = converter_catalogue.loc[
        converter_catalogue['Ecosystem'] == use_case['Project details']['Ecosystem']
        ]

    for index, row in net.converter.iterrows():
        converter_name = row["name"]
        nominal_power_kw = row["P"] * 1000  # Convert MW to kW
        converter_type = row["type"]

        tmp_catalogue = row["converter_catalogue"]
        converter_weight = None

        # If tmp_catalogue is a DataFrame, check if it's empty
        if isinstance(tmp_catalogue, pd.DataFrame):
            if not tmp_catalogue.empty:
                # Filter the catalogue for the installed converter
                matching_row = tmp_catalogue.loc[tmp_catalogue['Nominal power (kW)'] == nominal_power_kw]
                if not matching_row.empty:
                    converter_weight = matching_row['Weight (kg)'].values[0]

        if converter_weight is None:  # If not found, search in the global converter catalogue
            matching_row = converter_catalogue[
                    (converter_catalogue['Converter type'] == converter_type) &
                    (converter_catalogue['Nominal power (kW)'] == nominal_power_kw)
                    ]
            if not matching_row.empty:
                converter_weight = matching_row['Weight (kg)'].values[0]

        if converter_weight is None:  # If still not found, use the default cost
            print(f"WARNING: Converter {converter_name} not found in catalogue. Using default weight.")
            converter_weight = nominal_power_kw * default_weight_converter_kg_per_kw

        weight_converters_kg_dict[converter_name] = converter_weight
        total_weight_converter_kg += converter_weight

    return total_weight_converter_kg, weight_converters_kg_dict


def calculate_weight_cables(net, use_case, default_weight_cable_al_per_m=1, default_weight_cable_cu_per_m=2):
    """
    Calculate the weight of cables in the network.

    Parameters:
        net (pandapowerNet): The DC network model.
        use_case (dict): Dictionary containing UC definition data.
        default_weight_cable_al_per_m (float): Default weight per meter for aluminum cables in kEUR/m.
        default_weight_cable_cu_per_m (float): Default weight per meter for copper cables in kEUR/m.

    Returns:
        total_weight_cable_kg (float): Total weight of converters in kg.
        weight_cables_kg_dict (dict): Dictionary with individual converter weights.
    """

    weight_cables_kg_dict = {}
    total_weight_cable_kg = 0

    cable_info = use_case['Conductor parameters']
    cable_type = cable_info['Material ']  # Default to Aluminum if type is not specified

    for index, row in net.line.iterrows():
        cable_name = f"line {row['from_bus']} - {row['to_bus']}"
        cable_length_m = row["length_km"] * 1000

        # Determine weight per meter based on material type
        if "Copper" in cable_type:
            weight_per_m = default_weight_cable_cu_per_m
        else:
            weight_per_m = default_weight_cable_al_per_m

        # Calculate weight
        cable_weight = cable_length_m * weight_per_m
        weight_cables_kg_dict[cable_name] = cable_weight
        total_weight_cable_kg += cable_weight

    return total_weight_cable_kg, weight_cables_kg_dict


def calculate_lifecycle_emissions_kpi(
        net,
        use_case,
        path_converter_catalogue,
        default_weight_converter_kg_per_kw=1.0,
        default_weight_cable_al_per_m=1.0,
        default_weight_cable_cu_per_m=2.0,
        emission_factor_converter_kg_co2_per_kg=5.0,  # Default CO2 emission factor for converters
        emission_factor_cable_kg_co2_per_kg=3.0  # Default CO2 emission factor for cables
):
    """
    Compute the lifecycle CO2 emissions for converters and cables.
    Note: The way we currently define the lifecycle CO₂ emissions considers only the embodied emissions—that is,
    the emissions associated with the manufacturing and installation of converters and cables, based on their weight.

    Parameters:
    - net: Pandapower network
    - use_case: Use case description
    - path_converter_catalogue: Path to the converter catalogue
    - default_weight_converter_kg_per_kw: Default weight of converters per kW
    - default_weight_cable_al_per_m: Default weight of aluminum cables per meter
    - default_weight_cable_cu_per_m: Default weight of copper cables per meter
    - emission_factor_converter_kg_co2_per_kg: Emission factor for converters (kg CO2 per kg)
    - emission_factor_cable_kg_co2_per_kg: Emission factor for cables (kg CO2 per kg)

    Returns:
    - Total lifecycle CO2 emissions for converters and cables.
    """

    # --- Compute Converter weight ---
    total_weight_converter_kg, _ = calculate_weight_converters(
        net, use_case, path_converter_catalogue, default_weight_converter_kg_per_kw
    )

    # --- Compute Cable weight ---
    total_weight_cable_kg, _ = calculate_weight_cables(
        net, use_case, default_weight_cable_al_per_m, default_weight_cable_cu_per_m
    )

    # --- Compute lifecycle emissions ---
    total_emissions_converter_kg_co2 = total_weight_converter_kg * emission_factor_converter_kg_co2_per_kg
    total_emissions_cable_kg_co2 = total_weight_cable_kg * emission_factor_cable_kg_co2_per_kg

    total_lifecycle_emissions_kg_co2 = total_emissions_converter_kg_co2 + total_emissions_cable_kg_co2

    return total_lifecycle_emissions_kg_co2


def calculate_energy_savings(dc_total_energy_mwh, ac_total_energy_mwh):
    """
    Calculate the energy savings or extra energy when using a DC grid.
    Args:
        dc_total_energy_mwh: Total energy produced in the DC grid (MWh).
        ac_total_energy_mwh: Total energy produced in the equivalent AC grid (MWh).

    Returns:
        tuple: (absolute energy savings in MWh - positive if DC is more efficient, negative if AC is more efficient,
        percentage savings relative to AC energy)

    """
    if ac_total_energy_mwh is None or pd.isna(ac_total_energy_mwh):  # Check if missing
        return None, None  # Return None values if AC data is unavailable

    energy_savings_mwh = ac_total_energy_mwh - dc_total_energy_mwh
    energy_savings_percentage = (energy_savings_mwh / ac_total_energy_mwh) * 100 if ac_total_energy_mwh != 0 else 0
    return energy_savings_mwh, energy_savings_percentage


def calculate_total_capex_difference(dc_total_capex_keur, ac_total_capex_keur):
    """
     Calculate the total CAPEX difference between the DC and AC grids.
    Args:
        dc_total_capex_keur: Total capital expenditure for the DC grid (k€).
        ac_total_capex_keur: Total capital expenditure for the AC equivalent grid (k€).

    Returns:
        tuple: (absolute CAPEX difference in k€ - positive if AC is more expensive, negative if DC is more expensive,
        percentage difference relative to AC CAPEX).

    """
    if ac_total_capex_keur is None or pd.isna(ac_total_capex_keur):  # Check if missing
        return None, None  # Return None values if AC data is unavailable

    capex_difference_keur = ac_total_capex_keur - dc_total_capex_keur
    capex_difference_percentage = (capex_difference_keur / ac_total_capex_keur) * 100 if ac_total_capex_keur != 0 else 0
    return capex_difference_keur, capex_difference_percentage
