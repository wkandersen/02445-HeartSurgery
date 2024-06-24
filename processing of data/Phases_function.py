import pandas as pd
from ... import Anæstesihændelse_processed
from ... import OE1O_processed
def Phases(IDno,Dataframe): # Har indtil videre brugt AHM_processed i stedet for Dataframe
    operationsdatoer = pd.unique(OE1O_processed[OE1O_processed['IDno'] == IDno]['Operationsdato'].dt.date)
    Hændelsestidspunkt = Anæstesihændelse_processed[Anæstesihændelse_processed[Anæstesihændelse_processed['IDno'] == IDno]['Timestamp'].dt.date.isin(operationsdatoer)]

    Induktion = Hændelsestidspunkt[Hændelsestidspunkt['Hændelse'] == 'Induktion']['Timestamp']
    Bypass_start = Hændelsestidspunkt[Hændelsestidspunkt['Hændelse'] == 'CV Bypass start']['Timestamp']
    Aorta_tang_på = Hændelsestidspunkt[Hændelsestidspunkt['Hændelse'] == 'Aorta tang på']['Timestamp']
    Aorta_tang_af = Hændelsestidspunkt[Hændelsestidspunkt['Hændelse'] == 'Aorta tang af']['Timestamp']
    Bypass_slut = Hændelsestidspunkt[Hændelsestidspunkt['Hændelse'] == 'CV Bypass slut']['Timestamp']
    Stop_indsaml = Hændelsestidspunkt[Hændelsestidspunkt['Hændelse'] == 'Stop Data Indsamling']['Timestamp']
    AHM_ID = Dataframe[Dataframe['IDno'] == IDno]
    AHM_ID = AHM_ID.set_index(['Timestamp'])

    Phase_1 = AHM_ID.loc[Induktion.iloc[0]:(Bypass_start).iloc[0]]
    Phase_1 = Phase_1.drop(Phase_1.index[-1])

    Phase_2 = AHM_ID.loc[Bypass_start.iloc[0]:(Aorta_tang_på).iloc[0]]
    Phase_2 = Phase_2.drop(Phase_2.index[-1])

    Phase_3 = AHM_ID.loc[Aorta_tang_på.iloc[0]:(Aorta_tang_af).iloc[0]]
    Phase_3 = Phase_3.drop(Phase_3.index[-1])

    Phase_4 = AHM_ID.loc[Aorta_tang_af.iloc[0]:(Bypass_slut).iloc[0]]
    Phase_4 = Phase_4.drop(Phase_4.index[-1])

    Phase_5 = AHM_ID.loc[Bypass_slut.iloc[0]:(Stop_indsaml).iloc[0]]
    return Phase_1, Phase_2,Phase_3,Phase_4,Phase_5