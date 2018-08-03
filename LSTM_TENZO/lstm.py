from math import sqrt
import numpy
from numpy import concatenate
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np 

class LSTM_tenzo():
        """docstring for ClassName"""
        def __init__(self, revenue,days,feature="weather"):
            self.revenue=revenue
            self.days=days
            self.feature=feature
        # convert series to supervised learning
        
        def series_to_supervised(self,data, n_in=1, n_out=1, dropnan=True):
            n_vars = 1 if type(data) is list else data.shape[1]
            df = DataFrame(data)
            cols, names = list(), list()
            # input sequence (t-n, ... t-1)
            for i in range(n_in, 0, -1):
                cols.append(df.shift(i))
                names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
            # forecast sequence (t, t+1, ... t+n)
            for i in range(0, n_out):
                cols.append(df.shift(-i))
                if i == 0:
                    names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
                else:
                    names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
            # put it all together
            agg = concat(cols, axis=1)
            agg.columns = names
            # drop rows with NaN values
            if dropnan:
                agg.dropna(inplace=True)
            return agg


        def forecast_without_external(self):

            data = self.revenue[["paid_no_tax"]]
            #TS = ts.rolling(window=7).mean()
            # fix random seed for reproducibility
            data.dropna(inplace=True)
            np.random.seed(7)
            # load dataset
            values = data.values
            # ensure all data is float
            values = values.astype('float32')
            # normalize features
    
            #values=values.reshape(values.shape[0],1)
            scaler = MinMaxScaler(feature_range=(0, 1))      ####better than (-1,1)
            scaled = scaler.fit_transform(values)
            # frame as supervised learning
            supervised = self.series_to_supervised(scaled, 1, 1)
            # split into train and test sets
            supervised_values = supervised.values
            ###train_size = int(supervised.shape[0] * 0.67)   #####change the size of training set
            ###train = supervised_values[:train_size, :]
            ###test = supervised_values[train_size:, :]
            train = supervised_values[:-self.days, :]
            test = supervised_values[-self.days:, :]    # forecast 7 days
            train_size = len(train)
            # split into input and outputs
            train_X, train_y = train[:, :-1], train[:, -1]
            test_X, test_y = test[:, :-1], test[:, -1]
            # reshape input to be 3D [samples, timesteps, features]
            train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
            test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
            #print 'train_X.shape, train_y.shape, test_X.shape, test_y.shape:\n',(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
            # design LSTM network
            model = Sequential()
            model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2])))   #neurons = 50; 4
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')   #loss='mean_squared_error'; 'mae'
            # fit network
            history = model.fit(train_X, train_y, epochs=20, batch_size=1, verbose=2)
            #history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
            # make predictions
            #trainPredict = model.predict(train_X)
            testPredict = model.predict(test_X)
            # invert predictions
            #trainPredict = scaler.inverse_transform(trainPredict)
            #train_y = scaler.inverse_transform([train_y])
            testPredict = scaler.inverse_transform(testPredict)
            test_y = scaler.inverse_transform([test_y])
            df=pd.DataFrame({"forecast":testPredict.ravel(),"real":test_y.ravel()})

            return df


        def forecast_with_external(self):
            if self.feature=="weather":
                data = self.revenue[["paid_no_tax","temperature"]]
            else:
                data = self.revenue[["paid_no_tax","guest_ticket_count"]]
            #TS = ts.rolling(window=7).mean()
            # fix random seed for reproducibility
            data.dropna(inplace=True)
            np.random.seed(7)
            # load dataset
            values = data.values
            # ensure all data is float
            values = values.astype('float32')
            # normalize features
    
            #values=values.reshape(values.shape[0],1)
            scaler = MinMaxScaler(feature_range=(0, 1))      ####better than (-1,1)
            scaled = scaler.fit_transform(values)
            # frame as supervised learning
            supervised = self.series_to_supervised(scaled, 1, 1)
            # split into train and test sets
            supervised.drop(supervised.columns[[3]], axis=1, inplace=True)

            supervised_values = supervised.values
            ###train_size = int(supervised.shape[0] * 0.67)   #####change the size of training set
            ###train = supervised_values[:train_size, :]
            ###test = supervised_values[train_size:, :]
            train = supervised_values[:-self.days, :]
            test = supervised_values[-self.days:, :]    # forecast 7 days
            # split into input and outputs
            train_X, train_y = train[:, :-1], train[:, -1]
            test_X, test_y = test[:, :-1], test[:, -1]
            # reshape input to be 3D [samples, timesteps, features]
            train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
            test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
            #print 'train_X.shape, train_y.shape, test_X.shape, test_y.shape:\n',(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
            # design LSTM network
            model = Sequential()
            model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2])))   #neurons = 50; 4
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')   #loss='mean_squared_error'; 'mae'
            # fit network
            history = model.fit(train_X, train_y, epochs=20, batch_size=1, verbose=2)
            #history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
            # make predictions
            #trainPredict = model.predict(train_X)
            yhat = model.predict(test_X)
            test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
            # invert scaling for forecast
            inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
            inv_yhat = scaler.inverse_transform(inv_yhat)
            testPredict = inv_yhat[:,0]
            # invert predictions
            #trainPredict = scaler.inverse_transform(trainPredict)
            #train_y = scaler.inverse_transform([train_y])
            test_y = test_y.reshape((len(test_y), 1))
            inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
            inv_y = scaler.inverse_transform(inv_y)
            test_y = inv_y[:,0]
            df=pd.DataFrame({"forecast":testPredict.ravel(),"real":test_y.ravel()})

            return df

        def calculate_mape(self,df_cv,plot=False):
            df_cv['diff'] = (df_cv['real'] - df_cv['forecast'])/df_cv['real']
            df_cv['difference'] = df_cv['diff'].abs()*100
            if plot:
                show_results=pd.DataFrame({"forecast":df_cv['forecast'],"real":df_cv['real']})
                show_results.plot()
                plt.show()
            error2 = df_cv['difference'].mean()
            return error2

            # calculate root mean squared error
            #trainScore = sqrt(mean_squared_error(train_y[0], trainPredict[:,0]))
            #testScore = sqrt(mean_squared_error(test_y[0], testPredict[:,0]))
            #mape = ((abs(testPredict[:,0]-test_y[0])/test_y[0]).mean())*100
            #MAPE.append(mape)
            #print('Test error: {0:.4f}% MAPE').format(mape)
            
            #--- Plot predictions ---#
            