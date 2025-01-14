import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, StandardScaler, OneHotEncoder, LabelEncoder
from tensorflow.keras.models import load_model
from analysis import technical_analysis, indications
from tensorflow.keras.utils import to_categorical


#Declare models
buy_sell_prediction_model = load_model("models\model_buy_sell_.h5")
price_prediction_model = load_model("models\model_price_model.h5")

def data_preprocessing(df, close, high, low,):
    
    #Conduct the necessary technical and indication calculations
    technical_analysis(df, close, high, low,)
    indications(df)
    df.dropna(inplace = True)
    return df

def buy_sell_prediction(df, model): #Does all the buy sell indications used by the buy & sell MLs
    #Load preprocessed data
    training_window = 15
    df = df[['Action_Buy', 'Action_Hold', 'Action_Sell', 'Distinct_Action']]
   
    #Define the targeted categories before conversion to a sutable form for the ML model
    ohe = OneHotEncoder(categories = [['Buy', 'Hold', 'Sell']], sparse = False)
    #Define the parameter used for prediction
    X = np.array(df[['Action_Buy', 'Action_Hold', 'Action_Sell']])
    #Converting the targets to a form sutable for the ML to interpate diferent categories
    #y = ohe.fit_transform(df[['Distinct_Action']])

    le = LabelEncoder()
    le = le.fit(['Buy', 'Hold', 'Sell'])
    y = le.transform(df['Distinct_Action'])
    y = to_categorical(y)
    
    
    #Scales the data to make it easier for the ML to identify the desired patterns
    X = scale(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    
    #Feed the features to the general ML model to get the prediction
    results = model.predict(X).round(1)
    #Converting the targets to a form sutable for the ML to interpate diferent categories
    #clear
    # decoded = ohe.inverse_transform(results)
    decoded =le.inverse_transform(np.argmax(results.round(1), axis = 1))

    return decoded#Return the prediction for the most current data point

def price_prediction(df, model):

    #Define the parameter used for prediction
    X = np.array(df[['Open', 'High', 'Low', 'P', 'R1', 'R2', 'S1', 'S2']])
    y = np.array(df[['Close']])
    
     #Scales the data to make it easier for the ML to identify the desired patterns
    X = scale(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1))
    
    #Feed the features to the general ML model to get the prediction
    results = model.predict(X)
    results = scaler.inverse_transform(results)#Convert the prediction array back to the respective price value
    return (results.round(2)) #Return current price prediction and round off to 2 decimal places
