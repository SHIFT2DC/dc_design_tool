import openpyxl
import numpy as np
import pandas as pd


def read_UC_Definition(xl_file):
    df=xl_file.parse('UC Definition').rename(columns={"Project details": "param", "Unnamed: 1": "val"})
    UC_Definition={}
    UC_Definition['Project details']={df.iloc[0].param:df.iloc[0].val,
                                      df.iloc[1].param:df.iloc[1].val,
                                      df.iloc[2].param:df.iloc[2].val,
                                      df.iloc[3].param:df.iloc[3].val}
    
    UC_Definition['Grid architecture and inputs']={ df.iloc[7].param:df.iloc[7].val,
                                                    df.iloc[8].param:df.iloc[8].val,
                                                    df.iloc[9].param:df.iloc[9].val,
                                                    df.iloc[10].param:df.iloc[10].val,
                                                    df.iloc[11].param:df.iloc[11].val,
                                                    df.iloc[12].param:df.iloc[12].val,
                                                    df.iloc[13].param:df.iloc[13].val,
                                                    df.iloc[14].param:df.iloc[14].val,
                                                    df.iloc[15].param:df.iloc[15].val,
                                                    df.iloc[16].param:df.iloc[16].val}
    
    UC_Definition['Conductor parameters']={ df.iloc[19].param:df.iloc[19].val,
                                            df.iloc[20].param:df.iloc[20].val,
                                            df.iloc[21].param:df.iloc[21].val}
    
    UC_Definition['Converter details']={df.iloc[26].param:df.iloc[26].val}

    UC_Definition['Sizing factor']={    df.iloc[29].param:df.iloc[29].val,
                                        df.iloc[30].param:df.iloc[30].val,
                                        df.iloc[31].param:df.iloc[31].val,
                                        df.iloc[32].param:df.iloc[32].val}
    
    UC_Definition['Worst case scenario 1 for sizing of Storage DC/DC converter ']={
                                        df.iloc[35].param:df.iloc[35].val,
                                        df.iloc[36].param:df.iloc[36].val,
                                        df.iloc[37].param:df.iloc[37].val,
                                        df.iloc[38].param:df.iloc[38].val,
                                        df.iloc[39].param:df.iloc[39].val}
    
    UC_Definition['Worst case scenario 2 for sizing of cables and  PDU DC/DC,  DC/AC, PV DC/DC and EV DC/DC converters ']={
                                        df.iloc[43].param:df.iloc[43].val,
                                        df.iloc[44].param:df.iloc[44].val,
                                        df.iloc[45].param:df.iloc[45].val,
                                        df.iloc[46].param:df.iloc[46].val,
                                        df.iloc[47].param:df.iloc[47].val}
    
    UC_Definition['Worst case scenario 3 for sizing cables and AC/DC converter']={
                                        df.iloc[50].param:df.iloc[50].val,
                                        df.iloc[51].param:df.iloc[51].val,
                                        df.iloc[52].param:df.iloc[52].val,
                                        df.iloc[53].param:df.iloc[53].val,
                                        df.iloc[54].param:df.iloc[54].val}
    return UC_Definition
