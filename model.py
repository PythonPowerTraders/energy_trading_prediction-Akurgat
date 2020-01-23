import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, StandardScaler, OneHotEncoder
from tensorflow.keras.models import load_model
from analysis import technical_analysis, indications

#Declare models
distinct_model = load_model("models\buy_sell_models\model-distinct.h5")
general_model = load_model("models\buy_sell_models\model-general.h5")
price_prediction_model = load_model("models\price_prediction_models\price_model.h5")


def buy_sell_regions(df, model): #Function uses the raw data to identify the general by and sell regions 
    #Load at least 90 data point from the returned historical data for analysis
    df = df.iloc[-90:]
    #Conduct the necessary technical and indication calculations
    technical_analysis(df, df['Close'], df['High'], df['Low'])
    indications(df)
    
    ohe = OneHotEncoder(categories = [['Buy', 'Hold', 'Sell']], sparse = False)#Define the targeted categories before conversion 
                                                                                #to a sutable form for the ML model
    #Define the parameter used for prediction and drop the unnecessary columns in the data
    X = np.array(df.drop(['Open', 'High', 'Low', 'Close', 'P', 'R1', 'R2', 'S1', 'S2', 'P_Past', 
                        'R1_Past', 'R2_Past', 'S1_Past', 'S2_Past', 'Action_Buy', 'Action_Hold', 
                        'Action_Sell', 'General_Action', 'Distinct_Action'], 1))
    #Converting the targets to a form sutable for the ML to interpate diferent categories
    y = ohe.fit_transform(df[['General_Action']])
    
    #Scales the data to make it easier for the ML to identify the desired patterns
    X = scale(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    #Feed the features to the general ML model to get the prediction
    results = model.predict(X).round(1)
    decoded = ohe.inverse_transform(results) #Convert the prediction array back to the respective categories
    return decoded.reshape(-1), df #Reshape the prediction array to a 1D matrix and return it together with the data used 
    before the prediction

def distinct_buy_sell(data, model): #Identifies distinct buy sell point in the data based on the buy sell regions predicted above
    #data is the dataframe returned from the previous model
    df = pd.DataFrame()#Creating a and feeding the predictions from the general model to it
    ohe = OneHotEncoder(categories = [['Buy', 'Hold', 'Sell']], sparse = False)#Define the targeted categories before 
    conversion to a sutable form for the ML model
    signals, data = buy_sell_regions(data, general_model)
    df['General_Action'] =  (signals)

    #Analysing predictions from the identified regions to develop suitable feature for the distinct model and saving 
    # them to the dataframe
    df.loc[((df['General_Action'] == 'Buy') & (df['General_Action'].shift(-3) == 'Sell')), 'Action_Buy'] = 1
    df.loc[((df['General_Action'] == 'Sell') & (df['General_Action'].shift(-3) == 'Buy')), 'Action_Sell'] = 1
    df.loc[((df['General_Action'] == 'Hold')), 'Action_Hold'] = 1
    df['Action_Buy'].fillna(0, inplace = True)
    df['Action_Sell'].fillna(0, inplace = True)
    df['Action_Hold'].fillna(0, inplace = True)
    
    #Dropping all columns that are no longer necessary for analysis, backtesting and prediction
    df.drop(['General_Action'], inplace = True, axis =1)
    #Converting th remaining features to a form suitable for prediction
    X = np.array(df)
    results = model.predict(X).round(1)#Feeding the features to the general ML model to get the prediction
    y = ohe.fit_transform(data[['Distinct_Action']])#Converting the targets to a form sutable for the 
                                                    #ML to interpate diferent categories
    decoded = ohe.inverse_transform(results)#Convert the prediction array back to the respective categories
    print (decoded[-1]) #Return the prediction for the most current data point

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