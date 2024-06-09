import os
import sys
import json

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from plotly.colors import n_colors

from rfm_clustering.helpers import fetch_web_data, load_dataset
from rfm_clustering.rfm_steps import rfm_by_operation, raw_rfm, encoder_by_quartiles, category_encoder


# get current working directory
path_to_wd = os.getcwd()

# append path to local modules for streamlit use
sys.path.append(
    os.path.join(path_to_wd, 'rfm_clustering')
)

def axes_style3d(bgcolor = "rgb(204, 204, 204)",
                 gridcolor="rgb(150, 150, 150)", 
                 zeroline=False): 
    return dict(showbackground =True,
                backgroundcolor=bgcolor,
                gridcolor=gridcolor,
                zeroline=False)

def compute_rfm(df:pd.DataFrame, date:pd.Timestamp)->pd.DataFrame:

    df_1 = rfm_by_operation(df, date)

    del df

    df_2 = raw_rfm(df_1)

    del df_1

    df_3 = encoder_by_quartiles(df_2)

    df_4 = category_encoder(df_3)

    del df_3

    return df_4.merge(df_2, on='Customer ID')

def execute_main():

    st.set_page_config(layout="wide")

    WEB_PATH = 'https://raw.githubusercontent.com/LuisGiraldo86/auxiliarData/main/RFM/retails.tgz'
    FOLDER_PATH = os.path.join(path_to_wd, 'data')

    fetch_web_data(WEB_PATH, FOLDER_PATH)

    df = load_dataset(FOLDER_PATH)

    df_rfm = compute_rfm(df=df, date=pd.Timestamp(2013,1,10))

    st.title('RFM analysis of e-commerce sales')

    with st.container(border=True):
        st.write('Distribution of customer in levels and sublevels.')

        df_grouped = df_rfm.groupby(by=['scale', 'subscale'], as_index=False).agg(
            Total= pd.NamedAgg(column='Customer ID', aggfunc='count')
        )

        fig = px.treemap(
            df_grouped, 
            path                   =['scale', 'subscale'], 
            values                 ='Total', 
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
        st.plotly_chart(fig)

    with st.container(border=True):
        st.write('Customers clustering')

        my_axes =  axes_style3d() 
        fig = px.scatter_3d(
            df_rfm, 
            x                      ='frequency', 
            y                      ='recency', 
            z                      ='monetary', 
            color                  ='scale', 
            opacity                =0.9, 
            color_discrete_sequence=px.colors.qualitative.Pastel, 
            symbol                 ='scale'
        )
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.update_layout(scene=dict(xaxis=my_axes, 
                             yaxis=my_axes, 
                             zaxis=my_axes))
        fig.update_traces(marker_size = 5)
        st.plotly_chart(fig)


    pass

if __name__=="__main__":

    execute_main()