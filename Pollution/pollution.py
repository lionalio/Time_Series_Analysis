import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path + "/common")

from TSA import *

df = pd.read_csv('NSE-Tata-Global-Beverages-Limited.csv')
print(df)

processings = {
    'scaler': MinMaxScaler()
}

tsa = TimeSeriesAnalysis('pollution.csv', 'date')
