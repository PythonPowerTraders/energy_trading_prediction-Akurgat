from rest_ig import main
import pandas as pd
import numpy as np
from analysis import technical_analysis, indications

df = main()

technical_analysis(df, df['Close'], df['High'], df['Low'])
indications(df)