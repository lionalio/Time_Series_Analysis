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
tsa = TimeSeriesAnalysis('NSE-Tata-Global-Beverages-Limited.csv', 'Date')
#tsa.df['Close'].plot()
#plt.show()
tsa.set_methods_process(processings)
tsa.processing()
tsa.predict_single_instance(col='Close', window=60)