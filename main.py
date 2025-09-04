import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

import streamlit as st

def wrangle(filepath):
    df=pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df.drop(columns=['co', 'no', 'no2', 'o3', 'so2','pm10', 'nh3'],inplace=True)
    df=df.set_index('date')
    return df

df=wrangle('data/delhi_aqi.csv')
#print(df.info())
#print(df.head())

df_real = df.reset_index()
#print(df_real.head())

dfh=df_real
dfh['month'] = dfh['date'].dt.month
dfh['month_name'] = dfh['date'].dt.month_name()
#print(dfh.head())

month_df = dfh.groupby(['month', 'month_name'])['pm2_5'].mean().reset_index()
#print(month_df.head())


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

    st.header(' Create a time series plot of the PM 2.5 readings ')
    fig, ax = plt.subplots(figsize=(15, 6))
    df['pm2_5'].plot(ax=ax, xlabel='timestamp', ylabel='pm2.5', title='pm2.5 vs time series')
    st.pyplot(fig)

    # df = df.reset_index()
    # df['date'] = pd.to_datetime(df['date'])
    # df['month'] = df['date'].dt.month
    # df['month_name'] = df['date'].dt.month_name()
    # month_df = df.groupby(['month', 'month_name'])['pm2_5'].mean().reset_index()

    st.header('Monthly delhi pollution')
    st.subheader('Line Plot')
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(month_df['month_name'], month_df['pm2_5'], marker='o', color="blue", linewidth=2)
    ax.set_xlabel("Month")
    ax.set_ylabel("Average PM2.5")
    ax.set_title("Average PM2.5 by Month")
    plt.xticks(rotation=45)
    st.pyplot(fig)



    st.header('Monthly Delhi Pollution')
    st.subheader('Bar Plot')

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 8))

    bars = ax.bar(month_df['month_name'], month_df['pm2_5'], color="skyblue", edgecolor="black")

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.1f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )

    # Labels and formatting
    ax.set_xlabel("Month")
    ax.set_ylabel("Average PM2.5")
    ax.set_title("Average PM2.5 by Month")
    plt.xticks(rotation=45)


    st.pyplot(fig)

    ######

    #####
    # st.header('Weekly Time Series')
    # st.subheader(' Plot the rolling average of the window size of "P2" readings of 168 (the number of hours in a week).')
    # fig,ax=plt.subplots(figsize=(15,6))
    # df['pm2_5'].rolling(168).mean().plot(ax=ax,xlabel='timestamp',ylabel='pm2.5',title='weekly rolling average time series')
    # st.pyplot(fig)

    # st.header('Monthly delhi pollution')
    # st.subheader(
    #     ' Plot the rolling average of the window size of "P2" readings of 720 (the number of hours in a month).')
    # fig, ax = plt.subplots(figsize=(15, 6))
    # df['pm2_5'].rolling(720).mean().plot(ax=ax, xlabel='timestamp', ylabel='pm2.5',
    #                                      title='monthly rolling average time series')
    # st.pyplot(fig)


    # st.header('Yearly Time Series')
    # st.subheader(' Plot the rolling average of the window size of "P2" readings of 8640 (the number of hours in a year).')
    # fig,ax=plt.subplots(figsize=(15,6))
    # df['pm2_5'].rolling(8640).mean().plot(ax=ax,xlabel='timestamp',ylabel='pm2.5',title='yearly rolling average time series')
    # st.pyplot(fig)
    ########\

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
    def wrangle2(filepath):
        df = pd.read_csv(filepath)
        df.drop(columns=['co', 'no', 'no2', 'o3', 'so2', 'pm10', 'nh3'], inplace=True)
        return df


    df = wrangle2('data/delhi_aqi.csv')

    df['date'] = pd.to_datetime(df['date'])
    y1 = pd.Series(df['pm2_5'].values, index=df['date'])

    y = df['pm2_5']


    st.title('Delhi Air Quality Time Series')
    st.header('Model: AutoRegression')
    st.subheader('These predictions are on the bases of Auto Regression')

    #create an acf plot
    st.header('This is an ACF plot')
    fig,ax= plt.subplots(figsize=(15,6))
    plot_acf(y1,ax=ax)
    plt.xlabel('Lag [Hours]')
    plt.ylabel('correlation coefficient')
    st.pyplot(fig)

    # create an pacf plot
    st.header('This is an PACF plot')
    fig, ax = plt.subplots(figsize=(15, 6))
    plot_pacf(y1, ax=ax)
    plt.xlabel('Lag [Hours]')
    plt.ylabel('correlation coefficient')
    st.pyplot(fig)

    # Splitting the data into 95/5 ratio
    cutoff_test = int(len(y)*0.95)
    y_train = y.iloc[:cutoff_test]
    y_test = y.iloc[cutoff_test:]

    # Baseline Model
    y_pred_baseline=[y_train.mean()]*len(y_train)
    mae_baseline = mean_absolute_error(y_train,y_pred_baseline)

    st.subheader('This is Our Baseline Model Evaluation')
    # Or display nicely with metric
    st.metric(label="Mean PM2.5 Reading", value=round(y_train.mean(), 2))
    st.metric(label="Baseline MAE(error)", value=round(mae_baseline, 2))

    # Build Model
    model = AutoReg(y_train, lags=2).fit()

    y_pred = model.predict(start=2, end=len(y_train) - 1)

    y_pred = pd.Series(y_pred.values, index=df['date'][2:cutoff_test])

    training_mae = mean_absolute_error(y_train[2:], y_pred)

    st.subheader('This is Our Auto Regression Model Evaluation')
    st.metric(label="AutoReg MAE(error)",value=round(training_mae, 2))


elif menu == 'ARIMA':

    st.title('Delhi Air Quality Time Series')
    st.header('Model: ARIMA ')
    st.subheader('These predictions are on the bases of ARIMA walk forward validation with order(1,1,0)')

    y=df #series with pm2_5

    #create an acf plot
    st.header('This is an ACF plot')
    fig,ax= plt.subplots(figsize=(15,6))
    plot_acf(y,ax=ax)
    plt.xlabel('Lag [Hours]')
    plt.ylabel('correlation coefficient')
    st.pyplot(fig)

    #create an pacf plot
    st.header('This is an PACF plot')
    fig, ax = plt.subplots(figsize=(15, 6))
    plot_pacf(y, ax=ax)
    plt.xlabel('Lag [Hours]')
    plt.ylabel('correlation coefficient')
    st.pyplot(fig)

    #Splitting the data into 95/5 ratio
    cutoff_test = int(len(y)*0.95)
    y_train = y.iloc[:cutoff_test]
    y_test = y.iloc[cutoff_test:]

    # Baseline Model
    y_pred_baseline=[y_train.mean()]*len(y_train)
    mae_baseline = mean_absolute_error(y_train,y_pred_baseline)

    st.subheader('This is Our Baseline Model Evaluation')
    # Or display nicely with metric
    st.metric(label="Mean PM2.5 Reading", value=round(y_train.mean(), 2))
    st.metric(label="Baseline MAE(error)", value=round(mae_baseline, 2))

    # Build Model
    # y_pred_wfv = []
    # history = y_train.copy()
    # for i in range(len(y_test)):
    #     model = ARIMA(history, order=(1, 1, 0)).fit()
    #     next_pred = model.forecast()
    #     y_pred_wfv.append(next_pred.iloc[0])
    #     history = pd.concat([history, pd.Series([y_test.iloc[i]], index=[y_test.index[i]])])
    #
    # y_pred_wfv = pd.Series(y_pred_wfv, index=y_test.index)
    #
    # test_mae = mean_absolute_error(y_test, y_pred_wfv)
    test_mae=32.995433042208305

    st.subheader('This is Our ARIMA Model Evaluation')
    st.metric(label="ARIMA MAE(error) on test data",value=round(test_mae, 2))

    new_forecast= pd.read_csv('data/delhi_air_test_result.csv').set_index('Date')
    st.dataframe(new_forecast)


    ## create a plot
    st.header('This is a prediction  plot')
    fig,ax= plt.subplots(figsize=(15,10))
    # y1.plot(ax=ax)

    new_forecast["Actual PM2.5"].plot(ax=ax, label="Actual PM2.5", color="green", linewidth=2)
    new_forecast["Predicted PM2.5"].plot(ax=ax, label="Predicted PM2.5", color="orange", linewidth=2, linestyle="--")


    plt.xlabel('timestamp')
    plt.ylabel('PM2.5 values')
    st.pyplot(fig)

    st.subheader('this above table values and this graph show how well my model is able to predict actual values')
    st.text('Thanks')