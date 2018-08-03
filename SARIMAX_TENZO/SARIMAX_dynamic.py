
# coding: utf-8

# In[ ]:
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

class SARIMAX_dynamic():
    def __init__(self,data,order,days_number=0,external=1,feature="weather"):
        self.order=order
        self.endog=data["paid_no_tax"]
        self.days_number=days_number
        self.external=external
        if self.external == 1:
            self.feature=feature
            if feature=="weather":
                self.exog=data[["temperature","wind_speed","precipitation_intensity"]]
            else:
                self.exog=data["guest_ticket_count"]
    
    def fitModel_alldata_without_external(self):
        X=self.endog.values
        model = sm.tsa.statespace.SARIMAX(endog=X,order=self.order)
        self.results = model.fit(disp=False)
        return self.results 
   
    def fitModel_alldata_with_external(self):
        model = sm.tsa.statespace.SARIMAX(endog=self.endog.values,exog=self.exog.values,order=self.order)
        self.results = model.fit(disp=False)
        return self.results   
        
             
    def fitModel_without_external(self):
        X=self.endog.values
        size = int(len(X) -(self.days_number-1))
        train = X[0:size]
        model = sm.tsa.statespace.SARIMAX(endog=train ,order=self.order)
        self.results = model.fit(disp=False)
        return self.results 
    
    def fitModel_with_external(self):
        size = int(len(self.endog.values) -(self.days_number-1))
        train = self.endog.values[0:size]
        train_external = self.exog.values[0:size]
        model = sm.tsa.statespace.SARIMAX(endog=train,exog=train_external ,order=self.order)
        self.results = model.fit(disp=False)
        return self.results  
    
    def printresults(self):
        print(self.results.summary())
        
    
    def forecast(self,plot=False):
        size = int(len(self.endog.values) -(self.days_number-1))
        test = self.endog.values[size:len(self.endog.values)]
        if self.external ==1:
            if self.feature=="weather":
                test_external = self.exog.values[size:len(self.exog.values)]
                forecast = self.results.predict(start = size, end= len(self.endog.values)-1,exog=test_external.reshape(test_external.shape[0],3))
            else:
                test_external = self.exog.values[size:len(self.exog.values)]
                forecast = self.results.predict(start = size, end= len(self.endog.values)-1,exog=test_external.reshape(test_external.shape[0],1))
        else:
            forecast = self.results.predict(start = size, end= len(self.endog.values)-1) 
        if plot:
            show_results=pd.DataFrame({"forecast":forecast,"real":test})
            show_results.plot()
            plt.show()

        return forecast
    
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

