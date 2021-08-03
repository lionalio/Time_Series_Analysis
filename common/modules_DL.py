from libs import *


def model_lstm(input_shape):
    model=Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')

    return model