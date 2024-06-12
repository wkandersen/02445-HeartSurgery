# Importer data

import pandas as pd
import numpy as np
from Basic_functions import divide_by_ID
import os

titles = ['ADT' ,  'AHM' ,  'AoR' ,  'OE1O' , 'Anæstesidata' , 'Anæstesihændelse' ,  'Blødning' ,  'ca_vitale_værdier' ,   'Dialyse' ,  'EKG_data' ,  'Ekkokardiografi' ,  'Højde' ,  'intellispace' ,  'ITA' ,  'KAG_data' ,  'Lab_svar' ,  'medicin' ,  'Pleuradræn' ,  'Populationen' ,  'Problemliste' , 'Respirator_data' ,  'Samlet_udskillelse' ,  'Spiromitri' ,  'Total_indgift' ,  'Urin' ,  'Vægt' ,  'viewpoint' ]

for title in titles:
    parquet_file = f'Processed_parquet_real\{title}.parquet.gzip'
    df = pd.read_parquet(parquet_file)
    globals()[title] = df

# Checks and converts timestamp to the right
dataframes = {
    'ADT': ADT,
    'AHM': AHM,
    'AoR': AoR,
    'OE1O': OE1O,
    'Anæstesidata': Anæstesidata,
    'Anæstesihændelse': Anæstesihændelse,
    'Blødning': Blødning,
    'ca_vitale_værdier': ca_vitale_værdier,
    'Dialyse': Dialyse,
    'EKG_data': EKG_data,
    'Ekkokardiografi': Ekkokardiografi,
    'Højde': Højde,
    'intellispace': intellispace,
    'ITA': ITA,
    'KAG_data': KAG_data,
    'Lab_svar': Lab_svar,
    'medicin': medicin,
    'Pleuradræn': Pleuradræn,
    'Populationen': Populationen,
    'Problemliste': Problemliste,
    'Respirator_data': Respirator_data,
    'Samlet_udskillelse': Samlet_udskillelse,
    'Spiromitri': Spiromitri,
    'Total_indgift': Total_indgift,
    'Urin': Urin,
    'Vægt': Vægt,
    'viewpoint': viewpoint
}

def check_timestamp_header_and_type(df_dict):
    results = []
    for name, df in df_dict.items():
        if 'Timestamp' in df.columns:
            timestamp_type = df['Timestamp'].dtype
            if timestamp_type == np.dtype('O'):
                try:
                    # Try parsing with the default format
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
                except Exception as e_default:
                    print(f"Default parsing failed for DataFrame '{name}' with error: {e_default}")
                    try:
                        # Try parsing with dayfirst=True
                        df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, dayfirst=True)
                    except Exception as e_dayfirst:
                        print(f"Parsing with dayfirst=True failed for DataFrame '{name}' with error: {e_dayfirst}")
                        try:
                            # If parsing fails, you can specify a custom format or handle mixed formats
                            # Example: df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y', utc=True)
                            df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, errors='coerce')
                            if df['Timestamp'].isnull().any():
                                raise ValueError("Some dates could not be parsed and resulted in NaT values.")
                        except Exception as e_custom:
                            print(f"Custom parsing failed for DataFrame '{name}' with error: {e_custom}")
                            results.append((name, False, None))
                            continue
                timestamp_type = df['Timestamp'].dtype  # Update type after conversion
            results.append((name, True, timestamp_type))
        else:
            results.append((name, False, None))
    return results

# Check each DataFrame in the dictionary for the 'Timestamp' header and its type
results = check_timestamp_header_and_type(dataframes)

Populationen_drop = Populationen.drop(columns=['Dødsdato','Død inden for 1 år af operation','dead30','dead90','dead365','dead_days'])
# dropper de kolonner der kan havde noget at gøre med y-matricen

# Assuming Populationen and Anæstesihændelse are DataFrames
IDs_listx = Populationen['IDno'].unique()

no_induktion = []
no_stop = []
nothing = []
more_induktion = []
more_stop = []
for ID in IDs_listx:
    # Filter the rows where ID matches and Hændelse is 'Induktion' or 'Stop Data Indsamling'
    event_rows_induk = Anæstesihændelse[(Anæstesihændelse['IDno'] == ID) & (Anæstesihændelse['Hændelse'] == 'Induktion')]
    event_rows_stop = Anæstesihændelse[(Anæstesihændelse['IDno'] == ID) & (Anæstesihændelse['Hændelse'] == 'Stop Data Indsamling')]
    
    # Check for more than one row
    if  len(event_rows_induk) > 1:
        more_induktion.append(ID)
    if len(event_rows_stop) > 1:
        more_stop.append(ID)

    # If it has nothing
    if event_rows_induk.empty and event_rows_stop.empty:
        nothing.append(ID)
    
    # Add ID if it doesn't have either 'Induktion' or 'Stop Data Indsamling'
    if event_rows_induk.empty:
        if ID not in nothing:
            no_induktion.append(ID)
    if event_rows_stop.empty:
        if ID not in nothing:
            no_stop.append(ID)

from Basic_functions import divide_by_ID
import os

def save_dataframes_by_ID(IDs, dataframes, save_path):
    os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist
    for ID in IDs:
        concatenated_df = pd.concat([divide_by_ID(ID, df) for df in dataframes],
                                    ignore_index=True).sort_values('Timestamp', na_position='last').reset_index(drop=True)
        
        first_induktion_index = concatenated_df[concatenated_df['Hændelse'] == 'Induktion'].index.min()

        # Select all rows before the first occurrence of 'Induktion'
        data_before_induktion = concatenated_df.loc[:first_induktion_index - 1]

        rest = concatenated_df.loc[first_induktion_index:]

        # Check if either 'Genstande per uge' or 'Rygning' columns have any values
        events_with_values = concatenated_df[(concatenated_df['Genstande per uge'].notna()) | (concatenated_df['Rygning'].notna())]

        # Remove events_with_values from rest
        rest = rest.drop(events_with_values.index)

        # Concatenate data_before_induktion and events_with_values
        pre_op = pd.concat([data_before_induktion, events_with_values], ignore_index=True)

        # Find the index of the first occurrence of 'Induktion' in ID1
        concatenated_df = rest
        first_induktion_index = concatenated_df[concatenated_df['Hændelse'] == 'Induktion'].index.min()

        # Find the index of the first occurrence of 'Stop Data Indsamling' in ID1
        first_stop_index = concatenated_df[concatenated_df['Hændelse'] == 'Stop Data Indsamling'].index.min()
        rest = concatenated_df.loc[first_stop_index+1:]

        # Get the data between the two events
        Operation = concatenated_df.loc[first_induktion_index:first_stop_index]

        Post_op = rest

        file_name_pre = os.path.join(save_path, f"{ID.replace(' ','_')}.pre.parquet.gzip")
        file_name_op = os.path.join(save_path, f"{ID.replace(' ','_')}.op.parquet.gzip")
        file_name_post = os.path.join(save_path, f"{ID.replace(' ','_')}.post.parquet.gzip")

        pre_op.to_parquet(file_name_pre,compression='gzip')
        print(f"Saved dataframe for {ID} as {file_name_pre}")
        Operation.to_parquet(file_name_op,compression='gzip')
        print(f"Saved dataframe for {ID} as {file_name_op}")
        Post_op.to_parquet(file_name_post,compression='gzip')
        print(f"Saved dataframe for {ID} as {file_name_post}")


IDs_listx = (np.unique(Populationen['IDno']))  # List of IDs for which you want to save dataframes
IDs_listx = [ID for ID in IDs_listx if ID not in no_induktion and ID not in no_stop and ID not in nothing and ID not in more_induktion and ID not in more_stop]

IDs_list = IDs_listx[2076:]
dataframes = [AHM, AoR, OE1O, Anæstesidata, Anæstesihændelse, Blødning,
              ca_vitale_værdier, Dialyse, EKG_data, Ekkokardiografi,
              Højde, KAG_data, Lab_svar, medicin,
              Pleuradræn, Populationen_drop, Problemliste, Respirator_data]

#ITA er fjernet, da vi har indlæggelsestid i Population
#Intellispace fjernet for nu, da den er meget notebaseret og kan tilføjes senere hvis nødvendigt

save_path = "x_matricer"  # Specify your save path here
save_dataframes_by_ID(IDs_list, dataframes, save_path)

##Tjek hvor vi er nået til
IDs_listx = (np.unique(Populationen['IDno']))  # List of IDs for which you want to save dataframes
IDs_listx = [ID for ID in IDs_listx if ID not in no_induktion and ID not in no_stop and ID not in nothing and ID not in more_induktion and ID not in more_stop]

IDs_list = IDs_listx[2076]
IDs_list
