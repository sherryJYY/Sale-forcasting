# TENZO UK SALES FORECASTING

```
What is this ?
```
This an automatic multi model platform for forecasting sales made for tenzo company by Telecom Paristech team

```
Tenzo
```

Tenzo is a start up that helps restaurants gain value from their data by connecting all data
sources and providing valuable data to owners, managers and store employees in real time
on mobile and web. More specifically, it helps restaurants increase procurement efficiency
with automated forecasting and dynamically find the right balance of delivery and in-store
resources.


[![Build Status](https://www.gotenzo.com/wp-content/themes/tenzo_new3/img/logo-big.png)](https://www.gotenzo.com/)


Supervisors: 
```
Christian MOUYSSET
ADAM TAYLOR
```


Authors: 
```
Jawher Soudani

Xiaoyue LI

Yiting Sun

Yiying Jiang

Yuezhu Fang

```

Requiremnet :



Python 3.6 or higher version


Go to [`Prophet Installation`](https://facebook.github.io/prophet/docs/installation.html) for Prophet prerequisites installations

Go to [`Tenso_Flow_Installation`](https://www.tensorflow.org/install/) for TensorFlow installations

Go to [`Keras_Installation`](https://keras.io/) for Keras installation



```
Integrated Platform develepped by our team
```

This final integrated model resumes all three models mentioned above in one automatic model
The characteristics of this model are :

• Multiple Models platform: when using this integrated model , it first asks the user
to choose one of the three models to start working with. The configuration and the
options proposed depend on this choice.<br />
• Multiple options : Our integrated model propose multiple option do perform forecasting
on any number of days. The user can ask to do forecasting using dynamic method or
step by step method. Dynamic method forecast sales on any period of time using only
past data , the second method needs to retrain the model after each forecast using the
real value of the prediction.<br />
• Multi-usage : this model can be used to describe and analyze data or to forecast future
sales<br />
• Auto-Configuration : the model propose a high level configuration method that does
not require any technical knowledge to perform forecasting<br />
• Auto-Error handler : the model handle or internal error so that is does not interrupt
forecasting when used for multiple locations.<br />

```
How to run it
```

Super easy , you just run program runner.py  (command : python porgram_runner.py) in the same folder after you download this respository
It's guided , no knowledge needed to perform forecasting.

```
Can I modify this platform ? 
```

The models in this platform are seprated , and each class is also seperated so you can add other models or delete them , or add other features , or add others options
This is a beta version that show a great results but it can be improved by adding graphical interface to it
We deleted random forest from this platform because it gave the results as LSTM

```
Why it asked for put an other variable with weather ?
```

We put this to tell you that the model can be modified to add any varibale you want
We used count_guest_number for that other varibale just to fill it but count_guest_number is not an external feature !!! be carefull


```
Some screen shots
```
This a screen shot showing how the first interface of the model and how it is super easy to configure (all the real work is done in background , you only give orders )

[![Build Status](https://lh3.googleusercontent.com/Qv_0_8_E4cC-TLuTQwYdg68pExyJeMQcxTM74DqshQe9VJqgLIGtm79a5IGb-EI3GGrp-Y7t4J3puGp62rW4vrXffevTvChR4M3w=w1532-h694-rw-no)]


This a screen shot show the plot results that you will get if you ask the model to plot results

[![Build Status](https://lh3.googleusercontent.com/nQ7cSXiMlKT75tWtH3heIXGZD87e-ciC_HzaFJ0MG_C2QLMfgiIq8AeZC_V1oeUQLohUvuz5NRNUg6SCx7yRWU75wsnhNNinQIld=w1465-h833-rw-no)]


This a screen shot show the MAPE calculated by the model ( you need to close the plot graph to get the MAPE)

[![Build Status](https://lh3.googleusercontent.com/5RBlgnZqtVsdmMQmBlMyj9DBIjmi4izwNd4Q4CERvPXTVqwv2svBS_vTWAAj9oWWarxIN1xzWUFWZ4athzrWQD8BEHgGMqbtpOgg=w1684-h717-rw-no)]


```
Final remarks 
```
The platform works for all locations<br />
If you choose location and you want to add external feature , make sure that this location has external features , otherwise you will import empty table and the model won't work<br />
If you choose not to add external features, no verification needed , it will always work.<br />
Results are optimised for all models<br />
Parameters are autotuned by the models<br />
