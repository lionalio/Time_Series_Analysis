from data_preparation import *
from modules_DL import *


class TimeSeriesAnalysis(DataPreparation):
    def __init__(self, filename, time_col, delimiter=',', format='%Y-%m-%d', test_split=0.2):
        super().__init__(filename, time_col, delimiter, format, test_split)
        #self.data_train_DL = np.reshape(self.data_train, 
        #                                (self.data_train.shape[0], self.data_train.shape[1], 1))

    def convert_to_lstm_format(self, data):
        data1 = np.array(data)
        data1 = np.reshape(data1, (data1.shape[0], data1.shape[1], data1.shape[2]))
        
        return data1

    def get_windowed_data(self, data, window):
        out = []
        for i in range(window, len(data)):
            out.append(data[i-window:i])
        
        return out

    def split_series(self, data, n_past, n_future):
        # n_past: number of past observations
        # n_future: number of future observations
        X, y = [], []
        for w in range(len(data)):
            past_end = w + n_past
            future_end = past_end + n_future
            if future_end > len(series):
                break
            past, future = data[w:past_end, :], data[past_end:future_end, :]
            X.append(past)
            y.append(future)

        return np.array(X), np.array(y)

    def build_model_multistep_DL(self, n_past, n_future, method='LSTM'):
        # prepare data
        train_X, train_y = self.split_series(self.data_train, n_past, n_future)
        timesteps, features, outputs = train_X.shape[1], train_X.shape[2], train_X.shape[1]
        if method == 'LSTM':
            verbose, epochs, batch_size=0, 70, 16
            model = model_lstm(input_shape=(timesteps, features))
            model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        elif method == 'LGB':
            train_X = train_X.reshape(train_X.shape[0], timesteps*features)
            model = XGBRegressor()
            model.fit(train_X, train_y)
            
        return model

    def evaluation(self, model, n_past, n_future, method='LSTM'):
        test_X, test_y = self.split_series(self.data_test, n_past, n_future)
        if method != 'LSTM':
            test_X = test_X.reshape(train_X.shape[0], timesteps*features)
        forecast = model.predict(test_X)
        # create a list random indexes in forecast data
        # get the actual and forecast data from those indexes
        # get the evaluation metrics on that comparison       
    
    def predict_single_instance(self, colx, coly, window=60):
        # Training phase
        X_train = self.get_windowed_data(self.data_train[colx], window)
        y_train = np.array([self.data_train[coly][i] for i in range(window, len(self.data_train[coly]))])
        X_train = self.convert_to_lstm_format(X_train)
        model = model_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

        # Testing phase
        test_data = self.df[len(self.df)- len(self.data_test) - window:]        
        X_test = self.get_windowed_data(test_data[colx], window)
        X_test = self.convert_to_lstm_format(X_test)
        preds = model.predict(X_test)
        preds = self.methods['scaler'].inverse_transform(preds)[:, 0]

        # Showing results
        train_data = pd.DataFrame()
        train_data[coly] = self.methods['scaler'].inverse_transform(self.data_train[coly].values.reshape(-1, 1))[:, 0]
        train_data.index = self.data_train.index
        valid_data = self.data_test
        valid_data['predictions'] = preds
        valid_data[coly] = self.methods['scaler'].inverse_transform(self.data_test[coly].values.reshape(-1, 1))[:, 0]
        print(valid_data)
        plt.plot(train_data[coly])
        plt.plot(valid_data[[coly,"predictions"]])
        plt.show()
