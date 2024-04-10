def Dates_heart_op(IDno):
    operationsdatoer = pd.unique(OE1O_processed[OE1O_processed['IDno'] == IDno]['Operationsdato'].dt.date)
    Hændelsestidspunkt = Anæstesihændelse_processed[Anæstesihændelse_processed[Anæstesihændelse_processed['IDno'] == ID]['Timestamp'].dt.date.isin(operationsdatoer)]
    return Hændelsestidspunkt

def Inddeling(IDno):
    Dates_heart_op(IDno)
    Induktion = Hændelsestidspunkt[Hændelsestidspunkt['Hændelse'] == 'Induktion']['Timestamp']
    Bypass_start = Hændelsestidspunkt[Hændelsestidspunkt['Hændelse'] == 'CV Bypass start']['Timestamp']
    Aorta_tang_på = Hændelsestidspunkt[Hændelsestidspunkt['Hændelse'] == 'Aorta tang på']['Timestamp']
    Aorta_tang_af = Hændelsestidspunkt[Hændelsestidspunkt['Hændelse'] == 'Aorta tang af']['Timestamp']
    Bypass_slut = Hændelsestidspunkt[Hændelsestidspunkt['Hændelse'] == 'CV Bypass slut']['Timestamp']
    Stop_indsaml = Hændelsestidspunkt[Hændelsestidspunkt['Hændelse'] == 'Stop Data Indsamling']['Timestamp']

def Induktion_timestamps(IDno):
    AHM_ID = AHM_processed[AHM_processed['IDno'] == IDno]