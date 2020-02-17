import logging

from trading_ig import IGService, IGStreamService
from trading_ig.config import config
from trading_ig.lightstreamer import Subscription

import pandas as pd
import numpy as np
import datetime as dt
from pandas import json_normalize
from sqlalchemy import create_engine
from model import data_preprocessing, buy_sell_prediction, buy_sell_prediction_model, price_prediction, price_prediction_model
from analysis import pivot_point
    
#Declare the db
db_engine = create_engine(r'sqlite://', echo = False)

def streaming_func(df, engine = db_engine):

    #Ensure the number of historical values is set to 200 set max for predictions
    prediction_df = df.iloc[-200:].copy()
    prediction_df = data_preprocessing(prediction_df, prediction_df['Close'], 
                                        prediction_df['High'], prediction_df['Low'])
    #Predict the trade action
    predicted_buy_or_sell = buy_sell_prediction(prediction_df, buy_sell_prediction_model) 
    #Predict the possible close price          
    predicted_price = price_prediction(prediction_df, price_prediction_model)
    #Display prediction
    print (f'Best Trading Action: {predicted_buy_or_sell[-1]}')
    #Added Recommended action from analysis in the app just a confirmation the model is doing the right thing
    print (f'Recommended Action: {prediction_df.Distinct_Action.iloc[-1]}')
    print (f'Possible Next Candle Closing Price: {predicted_price[-1]}')

    # Set saved data limit and append the current record to the db table. 
    # If the table structure has changed, replace the table
    df = df.iloc[-200:]
    try:
        df.to_sql('streaming_data', con = engine, if_exists = 'append')
    except:
        df.to_sql('streaming_data', con = engine, if_exists = 'replace')

    print (df[['UTM', 'Close', 'High', 'Low']].iloc[-1]) #Return most recent streamed values

def db_func(df, engine = db_engine):
    
    try:
        #Chceking if db exists. If so merge with current data set. If not, proceed
        past_data = pd.read_sql("select * from streaming_data;", con = engine, index_col = 'DateTime')
        df = pd.concat([past_data, df], axis = 0)
        df.sort_index(axis = 0, ascending = True, inplace = True)
        df = df.loc[~df.index.duplicated(keep='last')]
    except:
        pass
    
    #Check if merged datset has enough values for preprocessing and predictions. Min 25 records
    if df.shape[0] < 26:
        
        #If data set has less than 25 records, append the current record to the db table
        df.to_sql('streaming_data', con = engine, if_exists = 'append')
    elif df.shape[0] >= 25 and int(df["CONS_END"].iloc[-1]) != 0:
        #Run streaming prediction function
        streaming_func(df)
    
def on_prices_update(item_update):
    
    #Create a database to hold historical streaming data
    #Convert received data set from lightstream from json to  dataframe for processing
    df = json_normalize(item_update['values'])
    
    #Create datetime column and index current datetime the update was made
    df['DateTime'] = dt.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
    df.set_index("DateTime", inplace = True)#Setting time column as the index
    df = df.rename(columns = {'OFR_OPEN':'Open', 'OFR_HIGH':'High', 'OFR_LOW':'Low', 'OFR_CLOSE':'Close'})

    #Ensure all values received are numerical for calculations
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].apply(pd.to_numeric)
    
    #Run the streaming functionto preprocess and predict the streamed data
    db_func(df)
            
    
def main():
    
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)

    ig_service = IGService(config.username, config.password, config.api_key, config.acc_type)

    ig_stream_service = IGStreamService(ig_service)
    ig_session = ig_stream_service.create_session()
    # Ensure configured account is selected
    accounts = ig_session[u'accounts']
    for account in accounts:
        if account[u'accountId'] == config.acc_number:
            accountId = account[u'accountId']
            break
        else:
            print('Account not found: {0}'.format(config.acc_number))
            accountId = None
    ig_stream_service.connect(accountId)

    # Making a new Subscription in MERGE
    # https://labs.ig.com/streaming-api-reference
    subscription_prices = Subscription(mode="MERGE", items=['CHART:CC.D.NG.USS.IP:5MINUTE'],
                                       fields=["UTM" , "OFR_OPEN", "OFR_HIGH", "OFR_LOW", "OFR_CLOSE", "CONS_END"],)
    subscription_prices.addlistener(on_prices_update)

    # Registering the Subscription
    sub_key_prices = ig_stream_service.ls_client.subscribe(subscription_prices)

    input("{0:-^80}\n".format("HIT CR TO UNSUBSCRIBE AND DISCONNECT FROM \
    LIGHTSTREAMER"))

    ig_stream_service.disconnect()

if __name__ == '__main__':
    main()