from data_preparation import *
from modules_DL import *


class TimeSeriesAnalysis(DataPreparation):
    def __init__(self, filename, time_col, delimiter=',', format='%Y-%m-%d', test_split=0.2):
        super().__init__(filename, time_col, delimiter, format, test_split)
        #self.data_train_DL = np.reshape(self.data_train, 
        #                                (self.data_train.shape[0], self.data_train.shape[1], 1))

    def convert_to_lstm_format(self, data):
        data1 = np.array(data)
        data1 = np.reshape(data1, (data1.shape[0], data1.shape[1], 1))
        
        return data1

    def get_windowed_data(self, data, window):
        out = []
        for i in range(window, len(data)):
            out.append(data[i-window:i])
        
        return out

    def predict_single_instance(self, col, window=60):
        # Training phase
        X_train = self.get_windowed_data(self.data_train[col], window)
        y_train = np.array([self.data_train[col][i] for i in range(window, len(self.data_train[col]))])
        X_train = self.convert_to_lstm_format(X_train)
        model = model_lstm(input_shape=(X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

        # Testing phase
        test_data = self.df[len(self.df)- len(self.data_test) - window:]        
        X_test = self.get_windowed_data(test_data[col], window)
        X_test = self.convert_to_lstm_format(X_test)
        preds = model.predict(X_test)
        preds = self.methods['scaler'].inverse_transform(preds)[:, 0]

        # Showing results
        train_data = pd.DataFrame()
        train_data[col] = self.methods['scaler'].inverse_transform(self.data_train[col].values.reshape(-1, 1))[:, 0]
        train_data.index = self.data_train.index
        valid_data = self.data_test
        valid_data['predictions'] = preds
        valid_data[col] = self.methods['scaler'].inverse_transform(self.data_test[col].values.reshape(-1, 1))[:, 0]
        print(valid_data)
        plt.plot(train_data[col])
        plt.plot(valid_data[[col,"predictions"]])
        plt.show()
