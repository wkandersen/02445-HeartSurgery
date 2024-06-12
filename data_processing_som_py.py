import pandas as pd
import numpy as np

titles = ['ADT' ,  'AHM' ,  'AoR' ,  'OE1O' , 'Anæstesidata' , 'Anæstesihændelse' ,  'Blødning' ,  'ca_vitale_værdier' ,   'Dialyse' ,  'EKG_data' ,  'Ekkokardiografi' ,  'Højde' ,  'intellispace' ,  'ITA' ,  'KAG_data' ,  'Lab_svar' ,  'medicin' ,  'Pleuradræn' ,  'Populationen' ,  'Problemliste' , 'Respirator_data' ,  'Samlet_udskillelse' ,  'Spiromitri' ,  'Total_indgift' ,  'Urin' ,  'Vægt' ,  'viewpoint' ]

for title in titles:
    parquet_file = f'Nye_parquet_files\{title}.parquet.gzip'
    df = pd.read_parquet(parquet_file)
    globals()[title] = df

    if title == 'Anæstesihændelse':
        globals()[title]=globals()[title].rename(columns={"Hændelsestidspunkt": "Timestamp"})
        globals()[title]['Timestamp'] = pd.to_datetime(globals()[title]['Timestamp'])

    elif title == 'Populationen':
        globals()[title]['Indlæggelsestidspunkt']=pd.to_datetime(globals()[title]['Indlæggelsestidspunkt'])
        globals()[title]['Udskrivelsestidspunkt']=pd.to_datetime(globals()[title]['Udskrivelsestidspunkt'])
        globals()[title]['Operationsdato']=pd.to_datetime(globals()[title]['Operationsdato'])
        globals()[title]['Dødsdato']=pd.to_datetime(globals()[title]['Dødsdato'])

        dead30 = []
        dead90 = []
        dead365 = []
        for i in range(len(globals()[title]["IDno"])):
            if globals()[title]["Død inden for 1 år af operation"][i] == "Nej":
                dead30.append(np.nan)
                dead90.append(np.nan)
                dead365.append(np.nan)
                continue
            else:
                difference = ((pd.to_datetime(globals()[title]["Dødsdato"][i]).date()-globals()[title]["Operationsdato"][i].date()).days)
                if difference <31:
                    dead30.append(difference)
                    dead90.append(np.nan)
                    dead365.append(np.nan)
                elif 31<= difference < 90:
                    dead30.append(np.nan)
                    dead90.append(difference)
                    dead365.append(np.nan)
                else:
                    dead30.append(np.nan)
                    dead90.append(np.nan)
                    dead365.append(difference)
                
        globals()[title]["dead30"]=dead30
        globals()[title]["dead90"]=dead90
        globals()[title]["dead365"]=dead365

        dead_days = []

        for i in range(len(globals()[title]["IDno"])):
            if globals()[title]["Død inden for 1 år af operation"][i] == "Nej":
                dead_days.append(0)
                continue
            else:
                difference = ((pd.to_datetime(globals()[title]["Dødsdato"][i]).date()-globals()[title]["Operationsdato"][i].date()).days)
                if difference <31:
                    dead_days.append(1)
                elif 31<= difference < 90:
                    dead_days.append(2)
                else:
                    dead_days.append(3)
                
        globals()[title]["dead_days"]=dead_days




    elif title == 'AHM':
        globals()[title]['Enhed'] = globals()[title]['Enhed'].fillna('Ikke angivet')
        globals()[title] = globals()[title].dropna(subset=['Værdi'])
        globals()[title]=globals()[title].rename(columns={ 'Målingsdatotid' :  'Timestamp' })
        globals()[title]['Timestamp'] = pd.to_datetime(globals()[title]['Timestamp'])

        globals()[title]['Værdi'] = pd.to_numeric(globals()[title]['Værdi'], errors='coerce')

        globals()[title] = globals()[title].groupby(['Timestamp','IDno','Skema navn', 'Målingsnavn']).Værdi.mean().reset_index()

        globals()[title] = globals()[title].pivot(index =['IDno','Timestamp','Skema navn'], columns=['Målingsnavn'],values='Værdi')

        globals()[title].reset_index(inplace=True)

    elif title == 'OE10':
        globals()[title]['Operationsdato']=pd.to_datetime(globals()[title]['Operationsdato'])

    elif title == 'AoR':
        for i in range(len(globals()[title])):
            value = globals()[title]['Genstande per uge'].iloc[i]
            if pd.isnull(value):
                continue 
            elif isinstance(value, str):
                if '-' in value:
                    start, end = map(int, value.split('-'))
                    globals()[title].loc[i, 'Genstande per uge'] = (start + end) / 2
                else:
                    globals()[title].loc[i, 'Genstande per uge'] = float(value)
            elif isinstance(value, int):
                globals()[title].loc[i, 'Genstande per uge'] = float(value)
            else:
                globals()[title].loc[i, 'Genstande per uge'] = float(value) 

        rygerstatuser_rank = {
            'Storryger': 0,
            'Hver dag': 1,
            'Lejlighedsvis ryger': 2,
            'Nogle dage': 3,
            'Ryger, aktuel status ukendt': 4,
            'Tidligere': 5,
            'Udsat for passiv rygning - aldrig været ryger': 6,
            'Aldrig': 7,
            'Ukendt': 8,
            'Aldrig vurderet': 9
        }

        data = []

        ## Finder gennemsnit for genstande, og værste rygerstatus
        for IDno in globals()[title]['IDno'].unique():
            listen = globals()[title][globals()[title]['IDno'] == IDno]
            
            genstande = np.nan
            max_rank = 9 
            
            ## Udregn mean Genstande per uge 
            if not listen['Genstande per uge'].isnull().all():
                genstande = listen['Genstande per uge'].mean()
            
            ## Find værste rygerstatus opgivet
            for rygerstatus in listen['Rygning']:
                rank = rygerstatuser_rank.get(rygerstatus, 9)  # Default rank is 9
                if rank < max_rank:
                    max_rank = rank
                    worst_rygerstatus = rygerstatus
            
            data.append([genstande, worst_rygerstatus, IDno])

        globals()[title] = pd.DataFrame(data, columns=['Genstande per uge', 'Rygning', 'IDno'])

    # elif title == 'Lab_svar':
    #     globals()[title] = globals()[title].rename(columns={ 'V2' : 'What is being measured' ,  'V3' :  'Timestamp' ,  'V4' :  'Results' })
    #     globals()[title] = globals()[title].dropna(subset=['Results'])
    #     globals()[title]['Results'] = pd.to_numeric(globals()[title]['Results'], errors='coerce')
    #     globals()[title] = globals()[title].groupby(['Timestamp','IDno','What is being measured']).Results.mean().reset_index()
    #     globals()[title] = globals()[title].pivot(index = ['IDno','Timestamp'], columns=['What is being measured'],values='Results')
    #     globals()[title].reset_index(inplace=True)
    #     globals()[title]['Timestamp'] = pd.to_datetime(globals()[title]['Timestamp'])
    
    elif title == 'Lab_svar':
        globals()[title] = globals()[title].rename(columns={ 'Best./ord. navn' : 'What is being measured' ,  'Resultatdatotid' :  'Timestamp' ,  'Resultatværdi' :  'Results' })
        globals()[title] = globals()[title].dropna(subset=['Results'])
        globals()[title]['Results'] = pd.to_numeric(globals()[title]['Results'], errors='coerce')
        globals()[title] = globals()[title].groupby(['Timestamp','IDno','What is being measured']).Results.mean().reset_index()
        globals()[title] = globals()[title].pivot(index = ['IDno','Timestamp'], columns=['What is being measured'],values='Results')
        globals()[title].reset_index(inplace=True)
        globals()[title]['Timestamp'] = pd.to_datetime(globals()[title]['Timestamp'])

    print(f'loaded dataframe {parquet_file} as {title}')