import logging

from trading_ig import (IGService, IGStreamService)
from trading_ig.config import config
from trading_ig.lightstreamer import Subscription

import pandas as pd
import numpy as np
import datetime as dt
from pandas.io.json import json_normalize
from sqlalchemy import create_engine
from model import buy_sell_prediction, buy_sell_prediction_model, price_prediction, price_prediction_model
from analysis import pivot_point
    
def on_prices_update(item_update):
    #Create a database to hold historical streaming data
    engine = create_engine('sqlite:///streaming_db.db', echo = False)
    #Convert received data set from lightstream from json to  dataframe for processing
    df = json_normalize(item_update['values'])
    df['DateTime'] = dt.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")#Timestamping the live update
    df.set_index("DateTime", inplace = True)#Setting time column as the index
    
    df = df.apply(pd.to_numeric)#Ensure all values received are numerical for calculations

    try:
        #Chceking if db exists. If so merge with current data set. If not, proceed
        past_data = pd.read_sql("select * from streaming_data;", con = engine, index_col = 'DateTime')
        df = pd.concat([past_data, df], axis = 0)
    except:
        pass
    
    if df.shape[0] > 25: #Check if merged data set has enough valuse for preprocessing and predictions. Min 25 records
        
        predicted_buy_or_sell = buy_sell_prediction(df, buy_sell_prediction_model) #Predict the trade action
        predicted_price = price_prediction(df, price_prediction_model)#Predict the possible close price
        
        #Identifing the lenght of the predictions returned and normalizing with the original data set
        df_length = predicted_price.shape[0]
        df = df.iloc[-df_length:]

        #Add the predictions to the dataframe
        df['Predicted_Action'] = predicted_buy_or_sell 
        df['Predicted_Close_Price'] = predicted_price

        #Calculating pivot points from predicted price close price to determine stop losses
        pivot_point(df, df["Predicted_Close_Price"], df["OFR_HIGH"], df["OFR_LOW"]) 

        #Prepreocess the buy sell predictions based on the buy sell prediction. Was to be used for visualization
        df.loc[((df['Predicted_Action'] == 'Buy')), 'Predicted_Action_Buy'] = 1
        df.loc[((df['Predicted_Action'] == 'Sell')), 'Predicted_Action_Sell'] = 1
        df.loc[((df['Predicted_Action'] == 'Hold')), 'Predicted_Action_Hold'] = 1
        df['Predicted_Action_Buy'].fillna(0, inplace = True)
        df['Predicted_Action_Sell'].fillna(0, inplace = True)
        df['Predicted_Action_Hold'].fillna(0, inplace = True)

        #Save dataframe to db as table and replace existing table to keep the data set as recent as possible for accurate
        #future predictions.
        df = df.iloc[-90:]

        try:
            #only update most recent update to the db otherwise add and replace the whole table
            df.iloc[-1:].to_sql('streaming_data', con = engine, if_exists = 'append')
        except:
            df.to_sql('streaming_data', con = engine, if_exists = 'replace')
        
        
    else:
        #If data set has less than 25 records, append the current record to the db table
        df.to_sql('streaming_data', con = engine, if_exists = 'append')
        
    print (df.iloc[-1]) #Return most recent streamed values
    
def main():
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)

    ig_service = IGService(
         config.username, config.password, config.api_key, config.acc_type
    )

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
                                       fields=["UTM" , "OFR_OPEN", "OFR_HIGH", "OFR_LOW", "OFR_CLOSE"],)
    
    subscription_prices.addlistener(on_prices_update)
    

    # Registering the Subscription
    sub_key_prices = ig_stream_service.ls_client.subscribe(subscription_prices)
    

    input("{0:-^80}\n".format("HIT CR TO UNSUBSCRIBE AND DISCONNECT FROM \
    LIGHTSTREAMER"))

    ig_stream_service.disconnect()

if __name__ == '__main__':
    main()