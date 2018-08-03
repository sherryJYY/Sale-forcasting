import warnings
warnings.filterwarnings("ignore")

from DATABASE_CONNECTOR.Get_Data import Get_Data


import matplotlib.pyplot as plt

if __name__ == '__main__':
    #just run mapreduce !
        #args = sys.argv[1:]
    model=int(input("\nChoose the model you want to use for forecasting data :\n1: SARIMAX\n2: PROPHET\n3: LSTM\n "))
    if model==1:
        from SARIMAX_TENZO.SARIMAX_dynamic import SARIMAX_dynamic
        from SARIMAX_TENZO.SARIMAX_stepbystep import SARIMAX_stepbystep
        import SARIMAX_TENZO.SARIMAX_AUTOMATIC_EVALUATION.evaluation_functions as ef

        ide=int(input("\nEnter the location ID you want to forecast : "))
        option=int(input("Choose evaluation method (enter the number) :\n1: AIC and BIC\n2: MAPE\n"))
        method=int(input("Choose forecasting method (enter the number) :\n1: Dynamic\n2: Day per Day\n"))
        ext=int(input("Do you wanna add external variables to improve accuracy ? :\n1: YES\n2: NO\n"))
        if ext==1:
            ext_choice=int(input("Do you wanna use weather variables or -you can add other variable here- (Weather is recommended) ? :\n1: Weather\n2: other variable\n"))
            if ext_choice==1:
                feature_variable="weather"
            else:
                feature_variable="guest_ticket_count"
        log=int(input("Do you wanna apply log to data to reduce variance (No is recommended)? :\n1: YES\n2: NO\n"))
        daysnumber=int(input('Enter how many days you want to predict : \n'))
        plot=int(input("Do you wanna plot results ?:\n1: YES\n2: NO\n"))
        #ide=args[0]
        
        if ext ==1:
            data=Get_Data(ide)
            print( "\nData for location{0} imported\n".format(ide))
            if log==1:
                revenue = data.To_Timseries(log=True,ext=1)
            else:
                revenue = data.To_Timseries(log=False,ext=1)

            if len(revenue.values)==0:
                print("No sales data , table is empty ,Cannot continue forecasting")
            else:   
                print( "Data transformed to time series\n")
                print("Searching for best parameters for SARIMAX.....\n")
                if method==1:
                    if option==1:
                        ordr=ef.evaluate_using_aic_bic_dynamic_with_external(revenue,feature_variable)
                    else:
                        ordr=ef.evaluate_using_MAPE_dynamic_with_external(revenue,feature_variable)
                else:
                    if option==1:
                        ordr=ef.evaluate_using_aic_bic_stepbystep_with_external(revenue,feature_variable)
                    else:
                        ordr=ef.evaluate_using_MAPE_step_by_step_with_external(revenue,feature_variable)                
        else:
            data=Get_Data(ide)
            print( "\nData for location{0} imported\n".format(ide))
            if log==1:
                revenue = data.To_Timseries(log=True,ext=0)
            else:
                revenue = data.To_Timseries(log=False,ext=0)

            if len(revenue.values)==0:
                print("No sales data , table is empty ,Cannot continue forecasting")
            else:   
                print( "Data transformed to time series\n")
                print("Searching for best parameters for SARIMAX.....\n")
                if method==1:
                    if option==1:
                        ordr=ef.evaluate_using_aic_bic_dynamic_without_external(revenue)
                    else:
                        ordr=ef.evaluate_using_MAPE_dynamic_without_external(revenue)
                else:
                    if option==1:
                        ordr=ef.evaluate_using_aic_bic_stepbystep_without_external(revenue)
                    else:
                        ordr=ef.evaluate_using_MAPE_step_by_step_without_external(revenue)    
        print("Best parameters found !! success{0}\n".format(ordr))
        print("Building predictive model using best parameters found")
        if method==1:
            if ext==1:
                modeldc=SARIMAX_dynamic(revenue,ordr,daysnumber,external=1,feature=feature_variable)
                print('\nmodel built\n')
                print('Fitting model with data\n')
                modeldc.fitModel_with_external()
                print("Starting forecasting\n")
                if plot == 1:
                    forecast=modeldc.forecast(plot=True)
                else:
                    forecast=modeldc.forecast()

            else:
                modeldc=SARIMAX_dynamic(revenue,ordr,daysnumber,external=0)
                print('\nModel built\n')
                print('Fitting model with data\n')
                modeldc.fitModel_without_external()
                print("Starting forecasting\n")
                if plot == 1:
                    forecast=modeldc.forecast(plot=True)
                else:
                    forecast=modeldc.forecast()
        else:
            if ext==1:
                modeldc=SARIMAX_stepbystep(revenue,ordr,daysnumber,external=1,feature=feature_variable)
                print('\nModel built\n')
                print('Fitting model with data\n')
                print("Starting forecasting\n")
                if plot == 1:
                    forecast=modeldc.predict_with_external(plot=True)
                else:
                    forecast=modeldc.predict_with_external()
            else:
                modeldc=SARIMAX_stepbystep(revenue,ordr,daysnumber,external=0)
                print('\nModel built\n')
                print('Fitting model with data\n')
                print("Starting forecasting\n")
                if plot == 1:
                    forecast=modeldc.predict_without_external(plot=True)
                else:
                    forecast=modeldc.predict_without_external()                
        print('All done !!\n')
        print('Calculating MAPE\n')
        if log==1:
            MAPE=modeldc.MAPE_evaluation(forecast,log=True)
        else:
            MAPE=modeldc.MAPE_evaluation(forecast,log=False)
        print('Your MAPE is : {0}'.format(MAPE))


    if model==2:
        from prophet import Prophet_tenzo

        ide=int(input("\nEnter the location ID you want to forecast : "))
        method=int(input("Choose forecasting method (enter the number) :\n1: Dynamic\n2: Day per Day\n"))
        ext=int(input("Do you wanna add external variables to improve accuracy ? :\n1: YES\n2: NO\n"))
        if ext==1:
            ext_choice=int(input("Do you wanna use weather variables or -you can add other variable here- (Weather is recommended) ? :\n1: Weather\n2: other variable\n"))
            if ext_choice==1:
                feature_variable="weather"
            else:
                feature_variable="guest_ticket_count"
        daysnumber=int(input('Enter how many days you want to predict : \n'))
        plot=int(input("Do you wanna plot results ?:\n1: YES\n2: NO\n"))
        #ide=args[0]
        
        if ext ==1:
            data=Get_Data(ide)
            print( "\nData for location{0} imported\n".format(ide))
            revenue = data.To_Timseries(to_csv=False,log=False,ext=1)

            if len(revenue.values)==0:
                print("No sales data , table is empty ,Cannot continue forecasting")
            else:   
                print( "Data transformed to time series\n")
                          
        else:
            data=Get_Data(ide)
            print( "\nData for location{0} imported\n".format(ide))
            revenue = data.To_Timseries(to_csv=False,log=False,ext=0)

            if len(revenue.values)==0:
                print("No sales data , table is empty ,Cannot continue forecasting")
            else:   
                print( "Data transformed to time series\n")
                  
        print("Building predictive model")
        if method==1:
            if ext==1:
                modeldc=Prophet_tenzo(revenue,daysnumber,feature_variable)
                print('\nModel built\n')
                print('Fitting model with data\n')
                print("Starting forecasting\n")
                forecast=modeldc.forecast_dynamic_with_external_features()

            else:
                modeldc=Prophet_tenzo(revenue,daysnumber)
                print('\nModel built\n')
                print('Fitting model with data\n')
                print("Starting forecasting\n")
                forecast=modeldc.forecast_dynamic_without_external_features()
        else:
            if ext==1:
                modeldc=Prophet_tenzo(revenue,daysnumber,feature_variable)
                print('\nModel built\n')
                print('Fitting model with data\n')
                print("Starting forecasting\n")
                forecast=modeldc.forecast_day_per_day_with_external_features()

            else:
                modeldc=Prophet_tenzo(revenue,daysnumber)
                print('\nModel built\n')
                print('Fitting model with data\n')
                print("Starting forecasting\n")
                forecast=modeldc.forecast_day_per_day_without_external_features()


        print('All done !!\n')
        print('Calculating MAPE\n')
        if plot==1:
            MAPE=modeldc.calculate_mape(forecast,plot=True)
        else:
            MAPE=modeldc.calculate_mape(forecast)
        print('Your MAPE is : {0}'.format(MAPE))



    if model==3:
        from lstm import LSTM_tenzo

        ide=int(input("\nEnter the location ID you want to forecast : "))
        ext=int(input("\nDo you wanna add external variables to improve accuracy ? :\n1: YES\n2: NO\n"))
        if ext==1:
            ext_choice=int(input("Do you wanna use weather variables or -you can add other variable here- (Weather is recommended) ? :\n1: Weather\n2: other variable\n"))
            if ext_choice==1:
                feature_variable="weather"
            else:
                feature_variable="guest_ticket_count"
        daysnumber=int(input('Enter how many days you want to predict : \n'))
        plot=int(input("Do you wanna plot results ?:\n1: YES\n2: NO\n"))
        #ide=args[0]
        
        if ext ==1:
            data=Get_Data(ide)
            print( "\nData for location{0} imported\n".format(ide))
            revenue = data.To_Timseries(to_csv=False,log=False,ext=1)

            if len(revenue.values)==0:
                print("No sales data , table is empty ,Cannot continue forecasting")
            else:   
                print( "Data transformed to time series\n")
                          
        else:
            data=Get_Data(ide)
            print( "\nData for location{0} imported\n".format(ide))
            revenue = data.To_Timseries(to_csv=False,log=False,ext=0)

            if len(revenue.values)==0:
                print("No sales data , table is empty ,Cannot continue forecasting")
            else:   
                print( "Data transformed to time series\n")
                  
        print("Building predictive model")
        if ext==1:
            modeldc=LSTM_tenzo(revenue,daysnumber,feature_variable)
            print('\nModel built\n')
            print('Fitting model with data\n')
            print("Starting forecasting\n")
            forecast=modeldc.forecast_with_external()

        else:
            modeldc=LSTM_tenzo(revenue,daysnumber)
            print('\nModel built\n')
            print('Fitting model with data\n')
            print("Starting forecasting\n")
            forecast=modeldc.forecast_without_external()



        print('All done !!\n')
        print('Calculating MAPE\n')
        if plot==1:
            MAPE=modeldc.calculate_mape(forecast,plot=True)
        else:
            MAPE=modeldc.calculate_mape(forecast)
        print('Your MAPE is : {0}'.format(MAPE))
