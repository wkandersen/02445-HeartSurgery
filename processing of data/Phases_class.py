import pandas as pd
from ... import Anæstesihændelse_processed
from ... import OE1O_processed

class Phases():  
    Anæstesihændelse_processed = Anæstesihændelse_processed
    OE1O_processed = OE1O_processed

    def __init__(self, IDno, Dataframe):
        self.IDno = IDno
        self.Dataframe = Dataframe
        # self.Anæstesihændelse_processed = Anæstesihændelse_processed
        # self.OE1O_processed = OE1O_processed

    def Dates_op(self):
        operationsdatoer = pd.unique(self.OE1O_processed[self.OE1O_processed['IDno'] == self.IDno]['Operationsdato'].dt.date)
        Hændelsetidspunkt = self.Anæstesihændelse_processed[self.Anæstesihændelse_processed[self.Anæstesihændelse_processed['IDno'] == self.ID]['Timestamp'].dt.date.isin(operationsdatoer)]
        return Hændelsetidspunkt

    def Hændelser(self):
        Tidspunkt = self.Dates_op()
        Induktion = Tidspunkt[Tidspunkt['Hændelse'] == 'Induktion']['Timestamp']
        Bypass_start = Tidspunkt[Tidspunkt['Hændelse'] == 'CV Bypass start']['Timestamp']
        Aorta_tang_på = Tidspunkt[Tidspunkt['Hændelse'] == 'Aorta tang på']['Timestamp']
        Aorta_tang_af = Tidspunkt[Tidspunkt['Hændelse'] == 'Aorta tang af']['Timestamp']
        Bypass_slut = Tidspunkt[Tidspunkt['Hændelse'] == 'CV Bypass slut']['Timestamp']
        Stop_indsaml = Tidspunkt[Tidspunkt['Hændelse'] == 'Stop Data Indsamling']['Timestamp']
        return Induktion, Bypass_start, Aorta_tang_på, Aorta_tang_af, Bypass_slut, Stop_indsaml
    
    def Dataframe_set(self):
        AHM_ID = self.Dataframe[self.Dataframe['IDno'] == self.IDno]
        AHM_ID = AHM_ID.set_index(['Timestamp'])
        return AHM_ID
    
    def All_phases(self): # Fjerner sidste index også i phase 5
        phases = {}
        hændelser = self.Hændelser()
        for i in range(len(hændelser) - 1):
            start_phase, end_phase, AHM_ID = hændelser[i], hændelser[i + 1], hændelser[-1]
            phase_data = AHM_ID.loc[start_phase.iloc[0]:end_phase.iloc[0]]
            phases[f"Phase_{i+1}"] = phase_data.drop(phase_data.index[-1])
        return phases

    def Phase_1(self): # Induktion til Bypass start
        Induktion, Bypass_start, AHM_ID = self.Hændelser()[0], self.Hændelser()[1], self.Dataframe_set()
        Phase_1 = AHM_ID.loc[Induktion.iloc[0]:(Bypass_start).iloc[0]]
        Phase_1 = Phase_1.drop(Phase_1.index[-1])
        return Phase_1

    def Phase_2(self): # Bypass start til Aorta tang på
        Aorta_tang_på, Bypass_start, AHM_ID = self.Hændelser()[2], self.Hændelser()[1], self.Dataframe_set()
        Phase_2 = AHM_ID.loc[Bypass_start.iloc[0]:(Aorta_tang_på).iloc[0]]
        Phase_2 = Phase_2.drop(Phase_2.index[-1])
        return Phase_2
    
    def Phase_3(self): # Aorta tang på til Aorta tang af
        Aorta_tang_på, Aorta_tang_af, AHM_ID = self.Hændelser()[2], self.Hændelser()[3], self.Dataframe_set()
        Phase_3 = AHM_ID.loc[Aorta_tang_på.iloc[0]:(Aorta_tang_af).iloc[0]]
        Phase_3 = Phase_3.drop(Phase_3.index[-1])
        return Phase_3

    def Phase_4(self): # Aorta tang af til Bypass slut
        Bypass_slut, Aorta_tang_af, AHM_ID = self.Hændelser()[4], self.Hændelser()[3], self.Dataframe_set()
        Phase_4 = AHM_ID.loc[Aorta_tang_af.iloc[0]:(Bypass_slut).iloc[0]]
        Phase_4 = Phase_4.drop(Phase_4.index[-1])
        return Phase_4

    def Phase_5(self): # Bypass slut til stop data indsamling
        Bypass_slut, Stop_indsaml, AHM_ID = self.Hændelser()[4], self.Hændelser()[5], self.Dataframe_set()
        Phase_5 = AHM_ID.loc[Bypass_slut.iloc[0]:(Stop_indsaml).iloc[0]]
        return Phase_5