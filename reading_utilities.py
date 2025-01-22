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
                                            df.iloc[21].param:df.iloc[21].val,
                                            df.iloc[22].param:df.iloc[22].val,
                                            df.iloc[23].param:df.iloc[23].val}
    UC_Definition['Converter details']={df.iloc[26].param:df.iloc[26].val,
                                        df.iloc[27].param:df.iloc[27].val,
                                        df.iloc[28].param:df.iloc[28].val}
    UC_Definition['Worst case definition for Storage DC/DC converter sizing ']={
                                        df.iloc[31].param:df.iloc[31].val,
                                        df.iloc[32].param:df.iloc[32].val,
                                        df.iloc[33].param:df.iloc[33].val,
                                        df.iloc[34].param:df.iloc[34].val}
    UC_Definition['Worst case definition for AC/DC, DC/AC, PDU DC/DC, PV DC/DC and EV DC/DC converters sizing ']={
                                        df.iloc[37].param:df.iloc[37].val,
                                        df.iloc[38].param:df.iloc[38].val,
                                        df.iloc[39].param:df.iloc[39].val,
                                        df.iloc[40].param:df.iloc[40].val}
    return UC_Definition
