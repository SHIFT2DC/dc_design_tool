import openpyxl
import numpy as np
import pandas as pd


def read_UC_Definition(xl_file):
    df = xl_file.parse('UC Definition').rename(columns={"Project details": "param", "Unnamed: 1": "val"})
    UC_Definition = {}

    i_start = 0
    UC_Definition['Project details'] = {df.iloc[i_start+i].param: df.iloc[i_start+i].val for i in range(4)}
    
    i_start = list(df.loc[df['param'] == 'Grid architecture and inputs'].index)[0]+1
    UC_Definition['Grid architecture and inputs'] = {df.iloc[i_start+i].param: df.iloc[i_start+i].val for i in range(10)}
    
    i_start = list(df.loc[df['param'] == 'Conductor parameters'].index)[0]+1
    UC_Definition['Conductor parameters'] = {df.iloc[i_start+i].param: df.iloc[i_start+i].val for i in range(3)}
    
    i_start = list(df.loc[df['param'] == 'Sizing factors'].index)[0]+1
    UC_Definition['Sizing factor'] = {df.iloc[i_start+i].param: df.iloc[i_start+i].val for i in range(4)}
    
    i_start = list(df.loc[df['param'] == 'Worst case scenario 1 for sizing of Storage DC/DC converter '].index)[0]+1
    UC_Definition['Worst case scenario 1 for sizing of Storage DC/DC converter '] = {df.iloc[i_start+i].param: df.iloc[i_start+i].val for i in range(5)}
    
    i_start = list(df.loc[df['param'] == 'Worst case scenario 2 for sizing of cables and  PDU DC/DC,  DC/AC, PV DC/DC and EV DC/DC converters '].index)[0]+1
    UC_Definition['Worst case scenario 2 for sizing of cables and  PDU DC/DC,  DC/AC, PV DC/DC and EV DC/DC converters '] =  {df.iloc[i_start+i].param: df.iloc[i_start+i].val for i in range(5)}
    
    i_start = list(df.loc[df['param'] == 'Worst case scenario 3 for sizing cables and AC/DC converter'].index)[0]+1
    UC_Definition['Worst case scenario 3 for sizing cables and AC/DC converter'] = {df.iloc[i_start+i].param: df.iloc[i_start+i].val for i in range(5)}
    
    i_start = list(df.loc[df['param'] == 'Parameters for annual simulations'].index)[0]+1
    UC_Definition['Parameters for annual simulations'] = {df.iloc[i_start+i].param: df.iloc[i_start+i].val for i in range(2)}
    return UC_Definition

