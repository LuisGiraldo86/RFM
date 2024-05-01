"""
Module to store auxilary functions
"""
import os
import urllib.request
import zipfile

import pandas as pd

def fetch_web_data(http_address:str, path_folder:str)->None:

    if not os.path.isdir(path_folder):
        os.makedirs(path_folder)
    
    zip_path = os.path.join(path_folder, 'sales.zip')
    urllib.request.urlretrieve(http_address, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as file:
        file.extractall(path_folder)
    
    pass

def load_dataset(path_to_data:str)->pd.DataFrame:

    files_list = [file for file in os.listdir(path_to_data) if file.split('.')[-1]=='csv']

    df = pd.DataFrame()
    
    for file in files_list:
        df_temp = pd.read_csv(file)
        df = pd.concat([df, df_temp], axis=1)

    return df
