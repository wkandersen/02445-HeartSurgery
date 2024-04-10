def Dates_heart_op(IDno):
    operationsdatoer = pd.unique(OE1O_processed[OE1O_processed['IDno'] == IDno]['Operationsdato'].dt.date)
    H?ndelsestidspunkt = An?stesih?ndelse_processed[An?stesih?ndelse_processed[An?stesih?ndelse_processed['IDno'] == ID]['Timestamp'].dt.date.isin(operationsdatoer)]
    return H?ndelsestidspunkt

def Inddeling(IDno):
    Dates_heart_op(IDno)
    Induktion = H?ndelsestidspunkt[H?ndelsestidspunkt['H?ndelse'] == 'Induktion']['Timestamp']
    Bypass_start = H?ndelsestidspunkt[H?ndelsestidspunkt['H?ndelse'] == 'CV Bypass start']['Timestamp']
    Aorta_tang_p? = H?ndelsestidspunkt[H?ndelsestidspunkt['H?ndelse'] == 'Aorta tang p?']['Timestamp']
    Aorta_tang_af = H?ndelsestidspunkt[H?ndelsestidspunkt['H?ndelse'] == 'Aorta tang af']['Timestamp']
    Bypass_slut = H?ndelsestidspunkt[H?ndelsestidspunkt['H?ndelse'] == 'CV Bypass slut']['Timestamp']
    Stop_indsaml = H?ndelsestidspunkt[H?ndelsestidspunkt['H?ndelse'] == 'Stop Data Indsamling']['Timestamp']

def Induktion_timestamps(IDno):
    AHM_ID = AHM_processed[AHM_processed['IDno'] == IDno]