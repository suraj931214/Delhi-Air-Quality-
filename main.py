import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.ar_model import AutoReg

import streamlit as st

def wrangle(filepath):
    df=pd.read_csv(filepath)
    df.drop(columns=['co', 'no', 'no2', 'o3', 'so2','pm10', 'nh3'],inplace=True)
    df=df.set_index('date')


    return df

df=wrangle('data/delhi_aqi.csv')

menu=st.sidebar.radio(
    "Chose A Model",
    ['Home','LinearRegression','AutoRegression','ARIMA']
)
#####
if menu == "Home":
    st.title('Delhi Air Quality Index')
    st.header('Here we will show delhi air quality on the bases of PM2.5 data')

    #Data showcase
    st.text('let me first introduce my hourly data')
    st.dataframe(df.tail(26))

    # Box plot
    st.header('This is a Distribution of our pm2.5 data')
    fig,ax = plt.subplots(figsize=(15,6))
    df['pm2_5'].plot(kind='box',vert=False,title='Distribution of PM2.5 Reading',ax=ax)
    st.pyplot(fig)
    st.text('you can observe the outliers of delhi pollution ')
    ######

    #####
    st.header(' Create a time series plot of the PM 2.5 readings ')
    fig,ax=plt.subplots(figsize=(15,6))
    df['pm2_5'].plot(ax=ax,xlabel='timestamp',ylabel='pm2.5',title='pm2.5 vs time series')
    st.pyplot(fig)

    st.header('Weekly Time Series')
    st.subheader(' Plot the rolling average of the window size of "P2" readings of 168 (the number of hours in a week).')
    fig,ax=plt.subplots(figsize=(15,6))
    df['pm2_5'].rolling(168).mean().plot(ax=ax,xlabel='timestamp',ylabel='pm2.5',title='weekly rolling average time series')
    st.pyplot(fig)

    st.header('Monthly Time Series')
    st.subheader(' Plot the rolling average of the window size of "P2" readings of 720 (the number of hours in a month).')
    fig,ax=plt.subplots(figsize=(15,6))
    df['pm2_5'].rolling(720).mean().plot(ax=ax,xlabel='timestamp',ylabel='pm2.5',title='monthly rolling average time series')
    st.pyplot(fig)

    st.header('Yearly Time Series')
    st.subheader(' Plot the rolling average of the window size of "P2" readings of 8640 (the number of hours in a year).')
    fig,ax=plt.subplots(figsize=(15,6))
    df['pm2_5'].rolling(8640).mean().plot(ax=ax,xlabel='timestamp',ylabel='pm2.5',title='yearly rolling average time series')
    st.pyplot(fig)
    #####\

elif menu == 'LinearRegression':
    st.title('Delhi Air Quality Time Series')
    st.header('Model: LinearRegression')
    st.subheader('These predictions are on the bases of Linear Regression')
    # creating one more column duplicate column of pm2_5 just shift by one row
    df['pm2_5_shifted'] = df['pm2_5'].shift(1)

    # drop null value
    df.dropna(inplace=True)

    # building a model
    target='pm2_5'
    y=df[target]
    X=df.drop(columns=target)

    cutoff= int(len(df)*0.8)
    X_train,y_train=X[:cutoff],y[:cutoff]
    X_test,y_test=X[cutoff:],y[cutoff:]

    model=LinearRegression()
    model.fit(X_train,y_train)

    # training_mae = mean_absolute_error(y_train,model.predict(X_train))
    # test_mae = mean_absolute_error(y_test,model.predict(X_test))

    #######
    st.header('Model Prediction by column y_pred')
    df_pred_test = pd.DataFrame(
        {
            "y_test":y_test,
            "y_pred":model.predict(X_test)
        }
    )

    st.dataframe(df_pred_test.head())

    st.subheader('Time series Line plot')
    st.text('This is difference between model and real data plot')

    fig = px.line(df_pred_test,labels={"value":"PM2.5"})
    st.plotly_chart(fig)
    ######

elif menu == 'AutoRegression':
    y=wrangle('data/delhi_aqi.csv')

    st.title('Delhi Air Quality Time Series')
    st.header('Model: AutoRegression')
    st.subheader('These predictions are on the bases of Auto Regression')

    #create an acf plot
    st.header('This is an ACF plot')
    fig,ax= plt.subplots(figsize=(15,6))
    plot_acf(y,ax=ax)
    plt.xlabel('Lag [Hours]')
    plt.ylabel('correlation coefficient')
    st.pyplot(fig)



    pass

elif menu == 'ARIMA':
    pass