
# coding: utf-8

# In[ ]:
import warnings
warnings.filterwarnings("ignore")

#from scipy.optimize import brute
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from SARIMAX_dynamic import SARIMAX_dynamic
from SARIMAX_stepbystep import SARIMAX_stepbystep

#we have 8 evaluation functions :


#without external variables

"evaluate_using_aic_bic_dynamic_without_external"
"evaluate_using_MAPE_dynamic_without_external"

"evaluate_using_aic_bic_stepbystep_without_external"
"evaluate_using_MAPE_step_by_step_without_external"

"--------------------------------------------------------------------------------------------------------------------------------------------"
#with external variables

"evaluate_using_aic_bic_dynamic_with_external"
"evaluate_using_MAPE_dynamic_with_external"

"evaluate_using_MAPE_step_by_step_with_external"
"evaluate_using_aic_bic_stepbystep_with_external"


"--------------------------------------------------------------------------------------------------------------------------------------------"
"--------------------------------------------------------------------------------------------------------------------------------------------"
"--------------------------------------------------------------------------------------------------------------------------------------------"
def objfunc_dynamic_with_external(ordr,revenue,feature_variable):
    try:
        model = SARIMAX_dynamic(data=revenue ,order=ordr,feature=feature_variable)
        model.fitModel_alldata_with_external()
    except:
        return float("inf")
    return model.aic_evaluation() +model.bic_evaluation() 

def brute_with_external(revenue,feature_variable):
    minimum = float("inf")
    final_order=(1,1,1)
    verbose=1
    total=56
    for p in range(0,7):
        for i in range(0,2):
            for q in range(0,4):
                print("{0} out of {1}".format(verbose,total))
                verbose+=1
                ordr=(p,i,q)
                baic=objfunc_dynamic_with_external(ordr,revenue,feature_variable)
                if (baic < minimum):
                    final_order=ordr
                    minimum=baic
                    
    return final_order

def evaluate_using_aic_bic_dynamic_with_external(revenue,feature_variable):
    return brute_with_external(revenue,feature_variable)


"--------------------------------------------------------------------------------------------------------------------------------------------"
"--------------------------------------------------------------------------------------------------------------------------------------------"
"--------------------------------------------------------------------------------------------------------------------------------------------"
def objfunc_dynamic_with_external_MAPE(ordr,revenue,feature_variable):
    try:
        model = SARIMAX_dynamic(data=revenue ,order=ordr,days_number=7,feature=feature_variable)
        model.fitModel_with_external()
        forecast=model.forecast(plot=False)
    except:
        return float("inf")
    return model.MAPE_evaluation(forecast)


def brute_with_external_mape(revenue,feature_variable):
    minimum = float("inf")
    final_order=(1,1,1)
    verbose=1
    total=56
    for p in range(0,7):
        for i in range(0,2):
            for q in range(0,4):
                print("{0} out of {1}".format(verbose,total))
                verbose+=1
                ordr=(p,i,q)
                baic=objfunc_dynamic_with_external_MAPE(ordr,revenue,feature_variable)
                if (baic < minimum):
                    final_order=ordr
                    minimum=baic
                    
    return final_order

def evaluate_using_MAPE_dynamic_with_external(revenue,feature_variable):
    return brute_with_external_mape(revenue,feature_variable)

"--------------------------------------------------------------------------------------------------------------------------------------------"
"--------------------------------------------------------------------------------------------------------------------------------------------"
"--------------------------------------------------------------------------------------------------------------------------------------------"

def objfunc_dynamic_without_external(ordr,revenue):
    try:
        model = SARIMAX_dynamic(data=revenue ,order=ordr,external=0)
        model.fitModel_alldata_without_external()
    except ValueError as e:
        return float("inf")
    return model.aic_evaluation() +model.bic_evaluation() 
def brute_without_external(revenue):
    minimum = float("inf")
    final_order=(1,1,1)
    verbose=1
    total=56
    for p in range(0,7):
        for i in range(0,2):
            for q in range(0,4):
                print("{0} out of {1}".format(verbose,total))
                verbose+=1
                ordr=(p,i,q)
                baic=objfunc_dynamic_without_external(ordr,revenue)
                if (baic < minimum):
                    final_order=ordr
                    minimum=baic
                    
    return final_order

def evaluate_using_aic_bic_dynamic_without_external(revenue):
    return brute_without_external(revenue)

"--------------------------------------------------------------------------------------------------------------------------------------------"
"--------------------------------------------------------------------------------------------------------------------------------------------"
"--------------------------------------------------------------------------------------------------------------------------------------------"

def objfunc_dynamic_without_external_MAPE(ordr,revenue):
    try:
        model = SARIMAX_dynamic(data=revenue ,order=ordr,days_number=7,external=0)
        model.fitModel_without_external()
        forecast=model.forecast(plot=False)
    except:
        return float("inf")
    return model.MAPE_evaluation(forecast)


def brute_without_external_mape(revenue):
    minimum = float("inf")
    final_order=(1,1,1)
    verbose=1
    total=56
    for p in range(0,7):
        for i in range(0,2):
            for q in range(0,4):
                print("{0} out of {1}".format(verbose,total))
                verbose+=1
                ordr=(p,i,q)
                baic=objfunc_dynamic_without_external_MAPE(ordr,revenue)
                if (baic < minimum):
                    final_order=ordr
                    minimum=baic
                    
    return final_order

def evaluate_using_MAPE_dynamic_without_external(revenue):
    return brute_without_external_mape(revenue)
"--------------------------------------------------------------------------------------------------------------------------------------------"
"--------------------------------------------------------------------------------------------------------------------------------------------"
"--------------------------------------------------------------------------------------------------------------------------------------------"
def objfunc_stepbystep_without_external(ordr,revenue):
    model = SARIMAX_stepbystep(data=revenue ,order=ordr,days_number=7,external=0)
    return  model.aic_and_bic_all_data_without_external()

def brute_without_external_step_by_step(revenue):
    minimum = float("inf")
    final_order=(1,1,1)
    verbose=1
    total=56
    for p in range(0,7):
        for i in range(0,2):
            for q in range(0,4):
                print("{0} out of {1}".format(verbose,total))
                verbose+=1
                ordr=(p,i,q)
                baic=objfunc_stepbystep_without_external(ordr,revenue)
                if (baic < minimum):
                    final_order=ordr
                    minimum=baic
                    
    return final_order

def evaluate_using_aic_bic_stepbystep_without_external(revenue):
    return brute_without_external_step_by_step(revenue)

"--------------------------------------------------------------------------------------------------------------------------------------------"
"--------------------------------------------------------------------------------------------------------------------------------------------"
"--------------------------------------------------------------------------------------------------------------------------------------------"
def objfunc_stepbystep_without_external_MAPE(ordr,revenue):
    try:
        model = SARIMAX_stepbystep(data=revenue ,order=ordr,days_number=7,external=0)
        forecast=model.predict_without_external()
    except:
        return float("inf")
    return model.MAPE_evaluation(forecast)


def brute_without_external_mape_step_by_step(revenue):
    minimum = float("inf")
    final_order=(1,1,1)
    verbose=1
    total=56
    for p in range(0,7):
        for i in range(0,2):
            for q in range(0,4):
                print("{0} out of {1}".format(verbose,total))
                verbose+=1
                ordr=(p,i,q)
                baic=objfunc_stepbystep_without_external_MAPE(ordr,revenue)
                if (baic < minimum):
                    final_order=ordr
                    minimum=baic
                    
    return final_order

def evaluate_using_MAPE_step_by_step_without_external(revenue):
    return brute_without_external_mape_step_by_step(revenue)

"--------------------------------------------------------------------------------------------------------------------------------------------"
"--------------------------------------------------------------------------------------------------------------------------------------------"
"--------------------------------------------------------------------------------------------------------------------------------------------"
def objfunc_stepbystep_with_external(ordr,revenue,feature_variable):
    model = SARIMAX_stepbystep(data=revenue ,order=ordr,days_number=7,feature=feature_variable)
    return  model.aic_and_bic_all_data_with_external()

def brute_with_external_step_by_step(revenue,feature_variable):
    minimum = float("inf")
    final_order=(1,1,1)
    verbose=1
    total=56
    for p in range(0,7):
        for i in range(0,2):
            for q in range(0,4):
                print("{0} out of {1}".format(verbose,total))
                verbose+=1
                ordr=(p,i,q)
                baic=objfunc_stepbystep_with_external(ordr,revenue,feature_variable)
                if (baic < minimum):
                    final_order=ordr
                    minimum=baic
                    
    return final_order

def evaluate_using_aic_bic_stepbystep_with_external(revenue,feature_variable):
    return brute_with_external_step_by_step(revenue,feature_variable)


"--------------------------------------------------------------------------------------------------------------------------------------------"
"--------------------------------------------------------------------------------------------------------------------------------------------"
"--------------------------------------------------------------------------------------------------------------------------------------------"

def objfunc_stepbystep_with_external_MAPE(ordr,revenue,feature_variable):
    try:
        model = SARIMAX_stepbystep(data=revenue ,order=ordr,days_number=7,external=1,feature=feature_variable)
        forecast=model.predict_with_external()
    except:
        return float("inf")
    return model.MAPE_evaluation(forecast)


def brute_with_external_mape_step_by_step(revenue,feature_variable):
    minimum = float("inf")
    final_order=(1,1,1)
    verbose=1
    total=56
    for p in range(0,7):
        for i in range(0,2):
            for q in range(0,4):
                print("{0} out of {1}".format(verbose,total))
                verbose+=1
                ordr=(p,i,q)
                print(ordr)
                baic=objfunc_stepbystep_with_external_MAPE(ordr,revenue,feature_variable)
                if (baic < minimum):
                    final_order=ordr
                    minimum=baic
                    
    return final_order

def evaluate_using_MAPE_step_by_step_with_external(revenue,feature_variable):
    return brute_with_external_mape_step_by_step(revenue,feature_variable)

