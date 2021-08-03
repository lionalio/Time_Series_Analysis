from libs import *


class DataPreparation():
    def __init__(self, filename, time_col, delimiter=',', format='%Y-%m-%d', test_split=0.2):
        self.df = pd.read_csv(filename, delimiter=delimiter)
        self.time_col = time_col
        self.df[self.time_col] = pd.to_datetime(self.df[self.time_col], format=format)
        self.df.index = self.df[self.time_col]
        self.df = self.df.sort_index(ascending=True, axis=0)
        self.df.drop(self.time_col, axis=1, inplace=True)
        self.test_split = test_split
        self.data_train, self.data_test = None, None

    def set_methods_process(self, methods):
        '''
        format: {
            'imputer': imputer/None,
            'mapping': mapping,
            'pca': True/False,
            'preprocess': scaler,
        }
        '''
        self.methods = methods

    def create_train_test(self):
        cut_point = int((1 - self.test_split) * len(self.df))
        self.data_train = self.df[:cut_point]
        self.data_test = self.df[cut_point:]

    def scaling(self):
        if 'scaler' in self.methods and self.methods['scaler'] is not None:
            for f in self.df.columns:
                self.data_train[f] = self.methods['scaler'].fit_transform(self.data_train[f].values.reshape(-1, 1))
                self.data_test[f] = self.methods['scaler'].transform(self.data_test[f].values.reshape(-1, 1))

    def processing(self):
        self.create_train_test()
        self.scaling()