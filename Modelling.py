import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error
from prophet import Prophet

import numpy as np

def train_test_split(dataframe,Date_column,forecasted_steps):
    # try:
    st.markdown("<h3 style='text-align: center; color: gray;'>Modelling and Forecasted Result </h3>", unsafe_allow_html=True)
    try:
        
        dataframe[Date_column] = pd.to_datetime(dataframe[Date_column]).dt.date
    except Exception:
        dataframe[Date_column] = pd.to_datetime(dataframe[Date_column])
    dataframe = dataframe.sort_values(by=Date_column)
    tab1, tab2 = st.columns(2)
    with tab1:
        st.info(f"Availabel Sales data is from {dataframe[Date_column].min()} ------ To ------ {dataframe[Date_column].max()}")

    with tab2:
        forecasted_data = pd.DataFrame(pd.date_range(dataframe[Date_column].max(), periods=forecasted_steps+1, freq='W-MON')).rename_axis(['week']).reset_index(drop=True)
        forecasted_data.columns = [Date_column]
        forecasted_data[Date_column] = pd.to_datetime(forecasted_data[Date_column]).dt.date
        forecasted_data = forecasted_data.iloc[1:,:]
        st.warning(f"Forecasted Sales data is from {forecasted_data[Date_column].min()} ------ To ------ {forecasted_data[Date_column].max()}")

    return dataframe,forecasted_data

def fbprophet_model(train_df,test_df,Target_column,Date_column):
    train_df = train_df.rename(columns={Date_column: 'ds',
                        Target_column: 'y'})
    test_df = test_df.rename(columns={Date_column: 'ds'})
    train_df = train_df[['ds','y']]
    my_model = Prophet(weekly_seasonality=True,yearly_seasonality=True, seasonality_prior_scale=0.07)
    my_model.fit(train_df)
    future_dates = test_df[['ds']]
    forecast = my_model.predict(future_dates)
    forecast = forecast[['ds','yhat']]
 


    
    st.markdown("<h3 style='text-align: center; color: gray;'>Final Prediction Chart </h3>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df['ds'], y=train_df[ 'y'],
                    mode='lines',
                    name='Historical Sales'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                    mode='lines',
                    name='Forecasted Sales'))

    fig.update_layout(   
                            title=f'{Target_column} Data',
                            xaxis_title=Target_column,
                            yaxis_title=Date_column,  

                            yaxis=dict(                         
                            showticklabels=True,
                            linecolor='rgb(204, 204, 204)',
                            linewidth=2,
                            ticks='outside',  ),

                            xaxis=dict(                         
                            showticklabels=True,
                            linecolor='rgb(204, 204, 204)',
                            linewidth=2,
                            ticks='outside',  ),


                            plot_bgcolor='white',
                            showlegend=True,)

    st.plotly_chart(fig, use_container_width = True)
    final_df = train_df.append(forecast)
    final_df = final_df.rename(columns={
                        'ds':Date_column,
                        'y':Target_column,
                        "yhat":'Forecasted_Sales'
    })
    final_df = final_df.fillna(0)
    tab1, tab2 = st.columns(2)
    with tab2 :
        st.markdown("<h3 style='text-align: left; color: gray;'>Final DataFrame </h3>", unsafe_allow_html=True)
        st.dataframe(final_df)
    with tab1:
        @st.cache
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(final_df)

        st.download_button(
            label="Download forecast Result as CSV",
            data=csv,
            file_name='forecast_Result.csv',
            mime='text/csv',
            )
    return train_df
