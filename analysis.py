import pandas as pd

#Imports all the necessary attributes and columns from the obtained IG data
#stoch period & rsi period set buy default to 5 and 14 respectively but can be redefined when calling the function
def technical_analysis(df, close, high, low, stoch_period = 5, rsi_period = 14): 

    #Runs all the technical analysis calculations with one command
    pivot_point(df, close, high, low)    
    stochastic(df, close, high, low, stoch_period)
    rsi(df, close, rsi_period)
    macd(df, close)

    #Dropping all points with null values after running the technical analysis
    df.dropna(inplace = True)

def indications(df):

    #Runs all the indications necessary for the buy & sell functions with one command
    macd_analysis(df)
    rsi_analysis(df)
    stochastic_rsi_analysis(df)
    price_action(df)

    #Dropping all points with null values after running all the necessary indications
    df.dropna(inplace = True)

def pivot_point(df, close, high, low): 

    #Indications were are instrumental to the price prediction ML along side High, Low and Open values from the IG API to determine the Close price
    #Calculating all the necessary pivot point values from dataframe. Formular obtained from investorpedia
    
    P = (close + high + low) / 3
    R1 = (P * 2) - low
    R2 = P + (high - low)
    S1 = (P * 2) - high
    S2 = P - (high - low)
    
    #Creating columns for the pivot point values and adding the to the data
    df['P'] = P
    df['R1'] = R1
    df['R2'] = R2
    df['S1'] = S1
    df['S2'] = S2

def stochastic(df, close, high, low, stoch_period):
    
    #Getting the minimum and maximum low and high values respectively over the declared stoch period
    stoch_low = low.rolling(window = stoch_period).min()
    stoch_high = high.rolling(window = stoch_period).max()


    fast_k = 100 * ((close - stoch_low) / (stoch_high - stoch_low)) #Running the slow stochastic formular to get %K and %D values
    fast_d = fast_k.rolling(window = 3).mean() #Using 3 as the slow period and %D period
    slow_d = fast_d.rolling(window = 3).mean()

    #Adding the calulated %K and %D columns to the data
    df['%K'] = fast_d
    df['%D'] = slow_d

def rsi(df, close, rsi_period):

    change = close.diff(1) #Running close price difference through the data

    #Identifying the gains and losses based on the price difference in the data
    gain = change.mask(change < 0, 0)
    loss = change.mask(change > 0, 0)

    #Running the exponetial mean based on the loss and gain
    average_gain = gain.ewm(com = rsi_period - 1, min_periods = rsi_period).mean()
    average_loss = loss.ewm(com = rsi_period - 1, min_periods = rsi_period).mean()
    rs = abs(average_gain / average_loss)
    rsi = 100 - (100 / (1 + rs))

    #Adding the calulated RSI columns to the data
    df['RSI'] = rsi

def macd(df, close):
    
    #Declaring default MACD parameters
    fast_length = 12
    slow_length = 26
    signal_smoothing = 9

    #Calcualting the fast, slow and smoothing exponential means
    ema1 = close.ewm(span = fast_length, adjust = False).mean()
    ema2 = close.ewm(span = slow_length, adjust = False).mean()
    macd = ema1 - ema2
    ema3 = macd.ewm(span = signal_smoothing, adjust = False).mean()
    #Getting the MACD histogtam values
    macd_histogram = macd - ema3

    #Adding the calulated MACD, MACDS and MACDH columns to the data
    df['MACD'] = macd
    df['MACDS'] = ema3
    df['MACDH'] = macd_histogram
    
def macd_analysis(df):
   
   #Compared the MACD and MACDS movement and crossover to check whether to buy, sell or hold. 
   #2 - Buy
   #1 - Hold
   #0 - Sell 
   # Note: These parameters above are similar in the rsi and stochastic-rsi analysis
    df.loc[((df['MACD'] < df['MACDS'])), 'MADC_Indication'] = 2
    df.loc[((df['MACD'] > df['MACDS'])), 'MADC_Indication'] = 0 
    df['MADC_Indication'].fillna(1, inplace = True)

def rsi_analysis(df):

    #Compared the RSI position to check whether to buy, sell or hold. 
    #Overbought >= 70
    #Oversold <= 30
    df.loc[((df['RSI'] >= 70)), 'RSI_Divagence_Convergence'] = 0
    df.loc[((df['RSI'] <= 30)), 'RSI_Divagence_Convergence'] = 2
    df['RSI_Divagence_Convergence'].fillna(1, inplace = True)

def stochastic_rsi_analysis(df):

    #Compared %K & %D crossover to check whether to buy, sell or hold. Included RSI to cancel out noisy (false) stoch signals
    #Overbought >= 80
    #Oversold <= 20
    df.loc[((df['%K'] > df['%D']) & (df['%K'] >= 80) & (df['RSI'] >= 70) & (df['MACDH'] < 0)), 'SR_Indication'] = 0
    df.loc[((df['%K'] < df['%D']) & (df['%K']) <= 20) & (df['RSI'] <= 30) & (df['MACDH'] > 0), 'SR_Indication'] = 2
    df['SR_Indication'].fillna(1, inplace = True)
    
def price_action(df): #Does all the buy sell indications used by the buy & sell MLs

    df['Indication'] =  df.loc[:, 'MADC_Indication':].mean(axis = 1).round(3) #Calculating the mean from the 3 indications above (macd, rsi, stochastic_rsi analysis)

    #Uses the mean values from the indications to determine general buy, sell and hold regions
    #Think of it as values from the indications from the Saturday Charts
    #Values also used for the general buy sell ML model. Shall explain further when I present the model
    df.loc[((df['Indication'] < 1 )), 'General_Action'] = 'Sell' #< 1 - Sell
    df.loc[((df['Indication'] > 1 )), 'General_Action'] = 'Buy' #> 1 - Buy
    df.loc[((df['Indication'] == 1 )), 'General_Action'] = 'Hold' #= 1 - Hold

    #Further using values from the general indications to identify distinctive buy sell points
    #Setting conditions where if the signal changes at least after 3 similar consecutive indications in the past, give that point as either a define buy, sell or hold. 
    #Filters false and erratic signals further
    #Made up the most recent Charts. Also necessary for backtesting
    df.loc[((df['General_Action'] == 'Buy') & (df['General_Action'].shift(-3) == 'Sell')), 'Action_Buy'] = 1
    df.loc[((df['General_Action'] == 'Sell') & (df['General_Action'].shift(-3) == 'Buy')), 'Action_Sell'] = 1
    df.loc[((df['General_Action'] == 'Hold')), 'Action_Hold'] = 1
    df['Action_Buy'].fillna(0, inplace = True)
    df['Action_Sell'].fillna(0, inplace = True)
    df['Action_Hold'].fillna(0, inplace = True)

    #Creating column holding values for the final buy sell ML model.
    df.loc[((df['Action_Buy'] == 0 ) & (df['Action_Sell'] == 1 )), 'Distinct_Action'] = 'Sell'
    df.loc[((df['Action_Buy'] == 1 ) & (df['Action_Sell'] == 0 )), 'Distinct_Action'] = 'Buy'
    df.loc[((df['Action_Buy'] == 0 ) & (df['Action_Sell'] == 0 )), 'Distinct_Action'] = 'Hold'

    #Dropping all columns that are no longer necessary for analysis, backtesting and prediction
    df.drop(['Indication', 'MADC_Indication', 'RSI_Divagence_Convergence', 'SR_Indication'], inplace = True, axis =1)
