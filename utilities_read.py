import openpyxl
import numpy as np
import pandas as pd


def read_uc_definition(xl_file):
    """
    Read the 'UC Definition' sheet from the provided Excel file.

    """
    df = xl_file.parse('UC Definition').rename(columns={"Project details": "param", "Unnamed: 1": "val"})
    uc_definition = {}

    i_start = 0
    uc_definition['Project details'] = {df.iloc[i_start+i].param: df.iloc[i_start+i].val for i in range(4)}
    
    i_start = list(df.loc[df['param'] == 'Grid architecture and inputs'].index)[0]+1
    uc_definition['Grid architecture and inputs'] = {df.iloc[i_start+i].param: df.iloc[i_start+i].val for i in range(10)}
    
    i_start = list(df.loc[df['param'] == 'Conductor parameters'].index)[0]+1
    uc_definition['Conductor parameters'] = {df.iloc[i_start+i].param: df.iloc[i_start+i].val for i in range(3)}
    
    i_start = list(df.loc[df['param'] == 'Sizing factors'].index)[0]+1
    uc_definition['Sizing factor'] = {df.iloc[i_start+i].param: df.iloc[i_start+i].val for i in range(4)}
    
    i_start = list(df.loc[df['param'] == 'Worst case scenario 1 for sizing of Storage DC/DC converter '].index)[0]+1
    uc_definition['Worst case scenario 1 for sizing of Storage DC/DC converter '] = {df.iloc[i_start+i].param: df.iloc[i_start+i].val for i in range(5)}
    
    i_start = list(df.loc[df['param'] == 'Worst case scenario 2 for sizing of cables and  PDU DC/DC,  DC/AC, PV DC/DC and EV DC/DC converters '].index)[0]+1
    uc_definition['Worst case scenario 2 for sizing of cables and  PDU DC/DC,  DC/AC, PV DC/DC and EV DC/DC converters '] = {df.iloc[i_start+i].param: df.iloc[i_start+i].val for i in range(5)}
    
    i_start = list(df.loc[df['param'] == 'Worst case scenario 3 for sizing cables and AC/DC converter'].index)[0]+1
    uc_definition['Worst case scenario 3 for sizing cables and AC/DC converter'] = {df.iloc[i_start+i].param: df.iloc[i_start+i].val for i in range(5)}
    
    i_start = list(df.loc[df['param'] == 'Parameters for annual simulations'].index)[0]+1
    uc_definition['Parameters for annual simulations'] = {df.iloc[i_start+i].param: df.iloc[i_start+i].val for i in range(2)}

    i_start = list(df.loc[df['param'] == 'Parameters of equivalent AC grid for KPIs comparison'].index)[0]+1
    uc_definition['Parameters of equivalent AC grid for KPIs comparison'] = {df.iloc[i_start + i].param: df.iloc[i_start + i].val for i in range(5)}
    return uc_definition


def read_cable_catalogue(path_catalogue):
    """
    Reads the cable catalogue from the specified Excel file.

    Parameters:
    path_catalogue (str): The path to the Excel file containing the cable catalogue.

    Returns:
    pd.DataFrame: The cable catalogue data.
    """
    # Load the Excel file
    xl_cat = pd.ExcelFile(path_catalogue)
    # Parse the 'catalogue' sheet
    cable_catalogue = xl_cat.parse('catalogue')
    return cable_catalogue


def process_cable_catalogue(catalogue, cable_info):
    """
    Processes the cable catalogue based on the provided cable information.

    Parameters:
    catalogue (pd.DataFrame): The cable catalogue data.
    cable_info (pd.DataFrame): The cable information data.

    Returns:
    pd.DataFrame: The processed cable catalogue.
    """
    # Extract temperature, material, and isolation type from cable_info
    op_temp = cable_info['Operating temperature (degrees Celsius)']
    mat = cable_info['Material ']
    isolation = cable_info['Isolation ']
    if 'Cu' in mat:
        mat = 'Cu'
    elif 'Al' in mat:
        mat = 'Al'

    if 'PVC' in isolation:
        isolation = 'PVC'
    elif 'XLPE' in isolation:
        isolation = 'XLPE'

    # Filter the catalogue based on material and isolation type
    catalogue = catalogue.loc[catalogue['materiaux'].str.lower() == mat.lower()]
    catalogue = catalogue.loc[catalogue['isolation'].str.lower() == isolation.lower()]

    # Calculate the resistance (R) at the given temperature
    catalogue['R'] = catalogue['Coef'] * (1 + catalogue['Const_r'] * (op_temp - 20))

    # Calculate the maximum current (Imax) based on the temperature and resistance
    catalogue['Imax'] = np.sqrt((catalogue['Tcond'] - op_temp) / (catalogue['Const_isol'] * catalogue['R']))

    # Sort the catalogue by maximum current (Imax)
    catalogue = catalogue.sort_values(by=['Imax'])

    # Reset the index of the DataFrame
    return catalogue.reset_index(drop=True)
