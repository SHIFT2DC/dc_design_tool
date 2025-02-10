import pandas as pd
import numpy as np


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
    print(cable_info)
    # Extract temperature, material, and isolation type from cable_info
    T = cable_info['Operating temperature (degrees Celsius)']
    mat = cable_info['Material ']
    isolation = cable_info['Isolation ']
    if 'Cu' in mat:
        mat ='Cu'
    elif 'Al' in mat:
        mat ='Al'

    if 'PVC' in isolation:
        isolation = 'PVC'
    elif 'XLPE' in isolation:
        isolation = 'XLPE'
        
    # Filter the catalogue based on material and isolation type
    catalogue = catalogue.loc[catalogue['materiaux'].str.lower() == mat.lower()]
    catalogue = catalogue.loc[catalogue['isolation'].str.lower() == isolation.lower()]

    # Calculate the resistance (R) at the given temperature
    catalogue['R'] = catalogue['Coef'] * (1 + catalogue['Const_r'] * (T - 20))

    # Calculate the maximum current (Imax) based on the temperature and resistance
    catalogue['Imax'] = np.sqrt((catalogue['Tcond'] - T) / (catalogue['Const_isol'] * catalogue['R']))

    # Sort the catalogue by maximum current (Imax)
    catalogue = catalogue.sort_values(by=['Imax'])

    # Reset the index of the DataFrame
    return catalogue.reset_index(drop=True)
