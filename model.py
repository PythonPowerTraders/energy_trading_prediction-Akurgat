import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, StandardScaler, OneHotEncoder
from tensorflow.keras.models import load_model
from analysis import technical_analysis, indications

#Declare models
buy_sell_prediction_model = load_model("models\model_buy_sell_.h5")
price_prediction_model = load_model("models\model_price_model.h5")

def buy_sell_prediction(df, model): #Does all the buy sell indications used by the buy & sell MLs
    #Load at least 90 data point from the returned historical data for analysis
    df = df.iloc[-90:]
    #Conduct the necessary technical and indication calculations
    technical_analysis(df, df['Close'], df['High'], df['Low'])
    indications(df)
    
    ohe = OneHotEncoder(categories = [['Buy', 'Hold', 'Sell']], sparse = False)#Define the targeted categories before conversion 
                                                                                #to a sutable form for the ML model
    #Define the parameter used for prediction and drop the unnecessary columns in the data
    X = np.array(df.drop(['Open', 'High', 'Low', 'Close', 'P', 'R1', 'R2', 'S1', 'S2', 'P_Past', '%K', '%D', 
                            'RSI', 'MACD', 'MACDS', 'MACDH','R1_Past', 'R2_Past', 'S1_Past', 'S2_Past', 
                              'General_Action', 'Distinct_Action'], 1))
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
    print (decoded[-1])#Return the prediction for the most current data point

def price_prediction(df, model):
    #Load at least 90 data point from the returned historical data for analysis
    df = df.iloc[-90:]
    technical_analysis(df, df['Close'], df['High'], df['Low'])#Conduct the necessary technical and indication calculations
    indications(df)
    
    #Define the parameter used for prediction and drop the unnecessary columns in the data
    X = np.array(df.drop(['Close', 'P_Past', 'R1_Past', 'R2_Past', 'S1_Past', 'S2_Past', '%K', '%D', 'RSI', 
                          'MACD', 'MACDS', 'MACDH', 'Action_Buy', 'Action_Hold', 'Action_Sell', 'General_Action', 
                          'Distinct_Action'], 1))
    y = np.array(df[['Close']])
    
     #Scales the data to make it easier for the ML to identify the desired patterns
    X = scale(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1))
    
    #Feed the features to the general ML model to get the prediction
    results = model.predict(X).round(2)
    results = scaler.inverse_transform(results)#Convert the prediction array back to the respective price value
    print (results[-1].round(2)) #Display current price prediction and round off to 2 decimal places