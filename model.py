import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, StandardScaler, OneHotEncoder
from tensorflow.keras.models import load_model
from analysis import technical_analysis, indications

#Declare models
buy_sell_prediction_model = load_model("models\model_buy_sell_.h5")
price_prediction_model = load_model("models\model_price_model.h5")

def data_preprocessing(df, close, high, low,):
    #Load at least 90 data point from the returned historical data for analysis
    #df = df.iloc[-90:]
    #Conduct the necessary technical and indication calculations
    try:
        #Change the cloumn names to avoid conflict further down the pipeline
        df = df.rename(columns = {'OFR_OPEN':'Open', 'OFR_HIGH':'High', 'OFR_LOW':'Low', 'OFR_CLOSE':'Close'})
    except:
        pass
    technical_analysis(df, close, high, low,)
    indications(df)
    return df

def buy_sell_prediction(data, model): #Does all the buy sell indications used by the buy & sell MLs
    #Load preprocessed data
    try:
        df = data_preprocessing(data, data['Close'], data['High'], data['Low'])
    except:
        df = data_preprocessing(data, data['OFR_CLOSE'], data['OFR_HIGH'], data['OFR_LOW'])
    ohe = OneHotEncoder(categories = [['Buy', 'Hold', 'Sell']], sparse = False)#Define the targeted categories before conversion 
                                                                                #to a sutable form for the ML model
    #Define the parameter used for prediction
    X = np.array(df[['Action_Buy', 'Action_Hold', 'Action_Sell']])
    #Converting the targets to a form sutable for the ML to interpate diferent categories
    y = ohe.fit_transform(df[['Distinct_Action']])
    
    #Scales the data to make it easier for the ML to identify the desired patterns
    X = scale(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    #Feed the features to the general ML model to get the prediction
    results = model.predict(X).round(1)
    #Converting the targets to a form sutable for the ML to interpate diferent categories
    decoded = ohe.inverse_transform(results)
    return (decoded)#Return the prediction for the most current data point

def price_prediction(data, model):
    #Load preprocessed data
    try:
        df = data_preprocessing(data, data['Close'], data['High'], data['Low'])
    except:
        df = data_preprocessing(data, data['OFR_CLOSE'], data['OFR_HIGH'], data['OFR_LOW'])
    #Define the parameter used for prediction
    X = np.array(df[['Open', 'High', 'Low', 'P', 'R1', 'R2', 'S1', 'S2']])
    y = np.array(df[['Close']])
    
     #Scales the data to make it easier for the ML to identify the desired patterns
    X = scale(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1))
    
    #Feed the features to the general ML model to get the prediction
    results = model.predict(X).round(2)
    results = scaler.inverse_transform(results)#Convert the prediction array back to the respective price value
    return (results.round(2)) #Return current price prediction and round off to 2 decimal places