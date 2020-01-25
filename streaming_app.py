import logging

from trading_ig import (IGService, IGStreamService)
from trading_ig.config import config
from trading_ig.lightstreamer import Subscription

import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
from sqlalchemy import create_engine
from model import buy_sell_prediction, buy_sell_prediction_model, price_prediction, price_prediction_model
    
def on_prices_update(item_update):
    #Create a database to hold historical streaming data
    engine = create_engine('sqlite:///streaming_db.db', echo = False)
    #Convert received data set from lightstream from json to  dataframe for processing
    df = json_normalize(item_update['values'])
    df.set_index("UTM", inplace = True)#Setting time column as the index
    
    try:
        #Chceking if db exists. If so merge with current data set. If not, proceed
        past_data = pd.read_sql("select * from streaming_data;", con = engine, index_col = 'UTM')
        df = pd.concat([past_data, df], axis = 0)
    except:
        pass
    
    df = df.apply(pd.to_numeric)#Ensure all values received are numerical for calculations
    
    if df.shape[0] > 25: #Check if merged datset has enough valuse for preprocessing and predictions. Min 25 records
        
        df = df.iloc[-90:]#Ensure the number of historical values is set to 90 set max to avoid overpopulating the db
        predicted_buy_or_sell = buy_sell_prediction(df, buy_sell_prediction_model) #Predict the trade action
        predicted_price = price_prediction(df, price_prediction_model)#Predict the possible close price
        
        #Save dataframe to db as table and replace existing table to keep the data set as recent as possible for accurate
        #future predictions.
        df.to_sql('streaming_data', con = engine, if_exists = 'replace') 
        
        #Display prediction
        print (f"Best Trading Action:{predicted_buy_or_sell}")
        print (f"Possible Closing Price:{predicted_price}")
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
            print('Account not found: {0}'.format(config.acc_number))S
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