"""
Module to store auxilary functions
"""

import os
from urllib.request import urlretrieve
import tarfile

import pandas as pd

def fetch_web_data(http_address:str, path_folder:str)->None:

    """
    Function to fetch sales data from a web repository.

    Parameters
    ----------
    http_address: str
        link pointing to the .tgz file with the sales data

    path_folder: str
        path to the folder where the data will be stored.

    Return
    ------
    None
    """

    if not os.path.isdir(path_folder):
        os.makedirs(path_folder)
    
    zip_path = os.path.join(path_folder, 'sales.zip')
    urlretrieve(http_address, zip_path)

    with tarfile.open(zip_path) as file:
        file.extractall(path_folder)
    
    pass

def load_dataset(path_to_data:str)->pd.DataFrame:

    """
    Function to load the data into a pandas dataframe.

    Parameter
    ---------
    path_to_data: str
        path to the folder where the .csv files are stored

    Return
    ------
    pandas.DataFrame
    """

    files_list = [file for file in os.listdir(path_to_data) if file.split('.')[-1]=='csv']

    df = pd.DataFrame()
    
    for file in files_list:
        df_temp = pd.read_csv(os.path.join(path_to_data, file))
        df = pd.concat([df, df_temp], axis=0)

    return df[df['Country']=='United Kingdom'].reset_index(drop=True)

def raw_rfm(data:pd.DataFrame)->pd.DataFrame:

    df = data.groupby(by='identity__Customer ID', as_index=False).agg(
        frequency= pd.NamedAgg(column='identity__Invoice', aggfunc='nunique'),
        recency  = pd.NamedAgg(column='recency__recency', aggfunc='min'),
        monetary = pd.NamedAgg(column='spending__spending', aggfunc='sum')
    )

    df.columns = [col.split('__')[-1] if '__' in col else col for col in df.columns]

    return df