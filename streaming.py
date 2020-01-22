from stream_ig import main
import pandas as pd
import numpy as np
from analysis import pivot_point

df = main()

df = df.rename(columns = {'OFR_OPEN':'Open', 'OFR_HIGH':'High', 'OFR_LOW':'Low', 'OFR_CLOSE':'Close'})

pivot_point(df, df['Close'], df['High'], df['Low'])
