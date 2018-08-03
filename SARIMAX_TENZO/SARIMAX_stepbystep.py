
# coding: utf-8

# In[ ]:
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np


                
                
class SARIMAX_stepbystep():
    def __init__(self,data,order,days_number=7,external=1,feature="weather"):
        self.order=order
        self.endog=data["paid_no_tax"]
        self.days_number=days_number
        self.external=external
        if self.external == 1:
            if feature=="weather":
                self.exog=data["temperature"]
            else:
                self.exog=data["guest_ticket_count"]
    
    def aic_and_bic_all_data_without_external(self):
        
        size = int(len(self.endog.values) -(self.days_number-1))
        train, test = self.endog.values[0:size], self.endog.values[size:len(self.endog.values)]
        history = [x for x in train]
        #predictions = list()
        aic=[]
        bic=[]
        try:
            for t in range(len(test)):
                model = sm.tsa.statespace.SARIMAX(history ,order=self.order)
                model_fit = model.fit(disp=0)
                   # output = model_fit.forecast()
                   # yhat = output[0]
                   # predictions.append(yhat)
                obs = test[t]
                history.append(obs)
                aic.append(model_fit.aic)
                bic.append(model_fit.bic)
            return np.mean(aic)+np.mean(bic)
        except:
            return float("inf")
                
               
    def aic_and_bic_all_data_with_external(self):
        
        size = int(len(self.endog.values) -(self.days_number-1))
        train, test = self.endog.values[0:size], self.endog.values[size:len(self.endog.values)]
        train_external, test_external = self.exog.values[0:size], self.exog.values[size:len(self.exog.values)]
        history_external =  [x for x in train_external]
        history = [x for x in train]
        #predictions = list()
        aic=[]
        bic=[]
        try:
            for t in range(len(test)):
                model = sm.tsa.statespace.SARIMAX(history ,exog=history_external,order=self.order)
                model_fit = model.fit(disp=0)
                   #                 output = model_fit.forecast(exog= test_external[t].reshape(1,1))

                   # yhat = output[0]
                   # predictions.append(yhat)
                obs = test[t]
                history.append(obs)
                obs_external = test_external[t]
                history_external.append(obs_external)
                aic.append(model_fit.aic)
                bic.append(model_fit.bic)
            return np.mean(aic)+np.mean(bic)
        except:
            return float("inf")
        
        
        
       
        
        
        
    def predict_without_external(self,plot=False):
        size = int(len(self.endog.values) -(self.days_number-1))
        train, test = self.endog.values[0:size], self.endog.values[size:len(self.endog.values)]
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
                model = sm.tsa.statespace.SARIMAX(history ,order=self.order)
                model_fit = model.fit(disp=0)
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)
        
        self.results=model_fit
        if plot:
            show_results=pd.DataFrame({"forecast":predictions,"real":test})
            show_results.plot()
            plt.show()

        return predictions
    
    
    
    def predict_with_external(self,plot=False):
        size = int(len(self.endog.values) -(self.days_number-1))
        train, test = self.endog.values[0:size], self.endog.values[size:len(self.endog.values)]
        train_external, test_external = self.exog.values[0:size], self.exog.values[size:len(self.exog.values)]
        history_external =  [x for x in train_external]
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
                model = sm.tsa.statespace.SARIMAX(history ,exog=history_external,order=self.order)
                model_fit = model.fit(disp=0)
                output = model_fit.forecast(exog=np.array([test_external[t]]).reshape(1,1))
                yhat = output[0]
                predictions.append(yhat)
                obs_external = test_external[t]
                history_external.append(obs_external)
                obs = test[t]
                history.append(obs)
        self.results=model_fit
        if plot:
            show_results=pd.DataFrame({"forecast":predictions,"real":test})
            show_results.plot()
            plt.show()

        return predictions  
    
    def printresults(self):
        print(self.results.summary())
        
    
   
    
    
    def aic_evaluation(self):
        return self.results.aic
        

    def bic_evaluation(self):
        return self.results.bic
        
    def MSE_evaluation(self,forecast):
        size = int(len(self.endog.values) -(self.days_number-1))
        test = self.endog.values[size:len(self.endog.values)]
        error = mean_squared_error(test,forecast)
        return error
    
    def ABS_evaluation(self,forecast):
        size = int(len(self.endog.values) -(self.days_number-1))
        test = self.endog.values[size:len(self.endog.values)]
        ABS_error=test-forecast
        return np.abs(ABS_error).mean()

        
    
    def MAPE_evaluation(self,forecast,log=False):
        size = int(len(self.endog.values) -(self.days_number-1))
        test = self.endog.values[size:len(self.endog.values)]
        ABS_error=test-forecast

        per_error=(ABS_error/test)*100
        MAPE=np.abs(per_error).mean()
        return MAPE  

        

