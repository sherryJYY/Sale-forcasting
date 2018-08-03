
# coding: utf-8

# In[ ]:

import pandas as pd 
import numpy as np
import psycopg2
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

class Get_Data:

    def __init__(self,location_id):
        self.host = "tenzo-postgres.postgres.database.azure.com"
        self.user = "prim_forecast@tenzo-postgres"
        self.dbname = "postgres"
        self.password = "3z08SBuQG0Pp4"
        self.sslmode = "require"
        self.location_id=location_id
            
    # Construct connection string

    def connect(self):
        try:
            conn_string = "host={0} user={1} dbname={2} password={3} sslmode={4}".format(self.host, self.user, self.dbname, self.password, self.sslmode)
            conn = psycopg2.connect(conn_string) 
            # create a psycopg2 cursor that can execute queries
            cursor = conn.cursor()
            return cursor
        except Exception as e:
            print("Uh oh, can't connect. Invalid dbname, user or password?")
            print(e)
    
    
    def execute(self):
        cursor=self.connect()
        cursor.execute("""select sd.date_of_business, sd.paid_no_tax , fw.temperature , fw.wind_speed , fw.precipitation_intensity , sd.guest_ticket_count
    from forecast_weather fw, collector_summarysalesbyday sd,collector_location l,collector_poslocation pl
    where fw.location_id = l.id
      and pl.app_location_id = l.id
      and sd.poslocation_id = pl.id
      and sd.poslocation_id = {0}
      and sd.date_of_business=fw.date
      order by sd.date_of_business;""".format(self.location_id))
        revenues=cursor.fetchall()
        return revenues
    def execute_without_external(self):
        cursor=self.connect()
        cursor.execute("""select sd.date_of_business, sd.paid_no_tax 
        	from collector_summarysalesbyday sd
    		where sd.poslocation_id = {0}
		    order by sd.date_of_business;""".format(self.location_id))
        revenues=cursor.fetchall()
        return revenues
        
    
    def To_Timseries(self,to_csv=False,log=False,ext=0):
        if ext==0:
            data=self.execute_without_external()
            revenue = pd.DataFrame(data,columns=['date','paid_no_tax'],dtype='float')
        else:
            data=self.execute()
            revenue = pd.DataFrame(data,columns=['date','paid_no_tax','temperature','wind_speed','precipitation_intensity','guest_ticket_count'],dtype='float')
        revenue['date'] = pd.to_datetime(revenue['date'],format = '%Y-%m-%d')
        revenue = revenue.groupby('date',as_index=True).mean()
        revenue = revenue.resample('D').pad()  #complete the missing date
        for i in range(len(revenue)):
            if revenue.iloc[i]["paid_no_tax"]<=0:
                revenue.iloc[i]["paid_no_tax"] = revenue.iloc[i-1]["paid_no_tax"]
        revenue=revenue.drop(revenue.index[len(revenue)-1])
        if log==True:
            revenue["paid_no_tax"]=revenue["paid_no_tax"].apply(np.log)
            if ext==1:
                revenue["guest_ticket_count"]=revenue["guest_ticket_count"].apply(np.log)
        if to_csv:
            revenue.to_csv(path_or_buf ="TempData{0}".format(GetData.location_id),index =False)
        return revenue
        
    def TimeseriesDescription(self,revenue):
        display(revenue.head())
        display(revenue.tail())
        display(revenue.describe())
        
    def TimeseriesVisualise(self,RollingMean,revenue):
        plt.figure(figsize=(15,8))
        revenue["paid_no_tax"].plot()
        revenue["paid_no_tax"].rolling(RollingMean).mean().plot(lw=8,label='12 Month Rolling Mean')
        revenue["paid_no_tax"].rolling(RollingMean).std().plot(label='12 Month Rolling Std')
        plt.legend()
        
    def TimeseriesETS(self,revenue,frequence=0):
        if frequence:
            decomposition = seasonal_decompose(revenue["paid_no_tax"],freq=frequence)
        else:
            decomposition = seasonal_decompose(revenue["paid_no_tax"])
        fig = plt.figure()  
        fig = decomposition.plot()  
        fig.set_size_inches(15, 8)
    
    def Is_Stationary(self,revenue,prt=False):
        # Store in a function for later use!
        result = adfuller(revenue["paid_no_tax"])
        if prt:
            print('Augmented Dickey-Fuller Test:')
            labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']

            for value,label in zip(result,labels):
                print(label+' : '+str(value) )
        if result[1] <= 0.05:
            return True
        else:
            return False
        
    def Differenciate(self,revenue,order=1):
        diff = revenue["paid_no_tax"]
        for i in range(order):
            diff = diff - diff.shift(1)
        return diff.dropna()
        
        
    def ACF_PACF(self,revenue):
        if isinstance(revenue,pd.core.frame.DataFrame):
            fig_first = plot_acf(revenue["paid_no_tax"].dropna()[:365])
            result = plot_pacf(revenue["paid_no_tax"].dropna()[:365])
        else:
            fig_first = plot_acf(revenue.dropna()[:365])
            result = plot_pacf(revenue.dropna()[:365])
            

