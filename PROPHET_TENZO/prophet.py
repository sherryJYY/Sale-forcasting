import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation


class Prophet_tenzo():
    def __init__(self, revenue,days,feature="weather"):
        self.days=days
        self.revenue=revenue
        self.feature=feature

    
    def forecast_dynamic_without_external_features(self):
        temp=self.revenue
        temp.reset_index(inplace=True)
        ts2=self.revenue[["date","paid_no_tax"]].rename(index=str, columns={"date": "ds", "paid_no_tax": "y"})
        l=len(ts2)
        s= str(l-(self.days+2))+'days'
        d=str(self.days)+'days'
        m = Prophet()
        m.fit(ts2)
        df_cv = cross_validation(m, horizon = d, initial = s,period = d) #1820

        return df_cv


    def forecast_day_per_day_without_external_features(self):
        temp=self.revenue
        temp.reset_index(inplace=True)
        ts2=self.revenue[["date","paid_no_tax"]].rename(index=str, columns={"date": "ds", "paid_no_tax": "y"})
        l=len(ts2)
        s= str(l-(self.days+2))+'days'
        m = Prophet()
        m.fit(ts2)
        df_cv = cross_validation(m, horizon = "1 days", initial = s,period = "1 days") #1820

        return df_cv

    def forecast_dynamic_with_external_features(self):
        temp=self.revenue
        temp.reset_index(inplace=True)
        if self.feature=="weather":
            ts2=self.revenue[["date","paid_no_tax","temperature","wind_speed","precipitation_intensity"]].rename(index=str, columns={"date": "ds", "paid_no_tax": "y"})
            feature=['temperature','wind_speed',"precipitation_intensity"]
        else:
            ts2=self.revenue[["date","paid_no_tax","guest_ticket_count"]].rename(index=str, columns={"date": "ds", "paid_no_tax": "y"})
            feature=['guest_ticket_count']            
        l=len(ts2)
        s= str(l-(self.days+2))+'days'
        d=str(self.days)+'days'

        m = Prophet()
        for i in feature:
            m.add_regressor(i)
        m.fit(ts2)
        df_cv = cross_validation(m, horizon = d, initial = s,period = d) #1820

        return df_cv
    

    def forecast_day_per_day_with_external_features(self):
        temp=self.revenue
        temp.reset_index(inplace=True)
        if self.feature=="weather":
            ts2=self.revenue[["date","paid_no_tax","temperature","wind_speed","precipitation_intensity"]].rename(index=str, columns={"date": "ds", "paid_no_tax": "y"})
            feature=['temperature','wind_speed',"precipitation_intensity"]
        else:
            ts2=self.revenue[["date","paid_no_tax","guest_ticket_count"]].rename(index=str, columns={"date": "ds", "paid_no_tax": "y"})
            feature=['guest_ticket_count']    
        l=len(ts2)
        s= str(l-(self.days+2))+'days'

        m = Prophet()
        for i in feature:
            m.add_regressor(i)
        m.fit(ts2)
        df_cv = cross_validation(m, horizon = "1 days", initial = s,period = "1 days") #1820

        return df_cv

    def calculate_mape(self,df_cv,plot=False):
        df_cv['diff'] = (df_cv['y'] - df_cv['yhat'])/df_cv['y']
        df_cv['difference'] = df_cv['diff'].abs()*100
        if plot:
            show_results=pd.DataFrame({"forecast":df_cv['yhat'],"real":df_cv['y']})
            show_results.plot()
            plt.show()
        error2 = df_cv['difference'].mean()
        return error2
