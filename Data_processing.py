import pandas as pd
class Data_processing():    
    def Dates_heart_op(IDno):
        operationsdatoer = pd.unique(OE1O_processed[OE1O_processed['IDno'] == IDno]['Operationsdato'].dt.date)
        Hændelsestidspunkt = Anæstesihændelse_processed[Anæstesihændelse_processed[Anæstesihændelse_processed['IDno'] == ID]['Timestamp'].dt.date.isin(operationsdatoer)]

    def Phases(IDno):
        Dates_heart_op(IDno) # Vi skal bruge det vi fandt i Dates
        Induktion = Hændelsestidspunkt[Hændelsestidspunkt['Hændelse'] == 'Induktion']['Timestamp']
        Bypass_start = Hændelsestidspunkt[Hændelsestidspunkt['Hændelse'] == 'CV Bypass start']['Timestamp']
        Aorta_tang_på = Hændelsestidspunkt[Hændelsestidspunkt['Hændelse'] == 'Aorta tang på']['Timestamp']
        Aorta_tang_af = Hændelsestidspunkt[Hændelsestidspunkt['Hændelse'] == 'Aorta tang af']['Timestamp']
        Bypass_slut = Hændelsestidspunkt[Hændelsestidspunkt['Hændelse'] == 'CV Bypass slut']['Timestamp']
        Stop_indsaml = Hændelsestidspunkt[Hændelsestidspunkt['Hændelse'] == 'Stop Data Indsamling']['Timestamp']
        AHM_ID = AHM_processed[AHM_processed['IDno'] == IDno]
        AHM_ID = AHM_ID.set_index(['Timestamp'])

    def Phase_1(IDno): #Induction to Bypass start
        Phases(IDno) # Vi skal bruge det vi fandt i Phases
        Phase_1 = AHM_ID.loc[Induktion.iloc[0]:(Bypass_start).iloc[0]]
        Phase_1 = Induk.drop(Induk.index[-1])
        return Phase_1

    def Phase_2(IDno):
        Phases(IDno) # Vi skal bruge det vi fandt i Phases
        Phase_2 = AHM_ID.loc[Bypass_start.iloc[0]:(Aorta_tang_på).iloc[0]]
        Phase_2 = Phase_2.drop(Phase_2.index[-1])
        return Phase_2
    
    def Phase_3(Idno):
        Phases(IDno) # Vi skal bruge det vi fandt i Phases
        Phase_3 = AHM_ID.loc[Aorta_tang_på.iloc[0]:(Aorta_tang_af).iloc[0]]
        Phase_3 = Phase_3.drop(Phase_3.index[-1])
        return Phase_3

    def Phase_4(IDno): #Induction to Bypass start
        Phases(IDno) # Vi skal bruge det vi fandt i Phases
        Phase_4 = AHM_ID.loc[Aorta_tang_af.iloc[0]:(Bypass_slut).iloc[0]]
        Phase_4 = Phase_4.drop(Phase_4.index[-1])
        return Phase_4

    def Phase_5(IDno): #Induction to Bypass start
        Phases(IDno) # Vi skal bruge det vi fandt i Phases
        Phase_5 = AHM_ID.loc[Bypass_slut.iloc[0]:(Stop_indsaml).iloc[0]]
        return Phase_5