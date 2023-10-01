#!/usr/bin/env python
# coding: utf-8

# The goal of making this machine learning model is to be able to predict the total daily quantity of products sold.

# # Load Dataset

# In[1]:


pip install pmdarima


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.metrics import mean_squared_error
import numpy as np


# In[3]:


import warnings

# Disable all warnings
warnings.filterwarnings("ignore")

# Or, only disable specific warnings based on category
# Example: Disabling DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[4]:


df = pd.read_csv("df_kalbe.csv")


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df = df.drop(["storeid.1", "productid.1", "price.1", "customerid.1"], axis=1)


# In[8]:


df.info()


# In[15]:


df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')


# # Total Quantity Per Day

# In[16]:


df_totalquantity = df.groupby('Date')["qty"].sum().reset_index()


# In[17]:


df_totalquantity.info()


# In[18]:


df_totalquantity.set_index('Date', inplace=True)


# In[19]:


from pylab import rcParams
rcParams['figure.figsize'] = 20, 7
df_totalquantity.plot(marker='o')
plt.show()


# In[20]:


df_totalquantity.boxplot()
plt.title('Boxplot of Total Quantity')
plt.ylabel('Total Quantity')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[21]:


def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )

    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
    else:
        print("weak evidence against null hypothesis,indicating it is non-stationary ")


# In[23]:


adfuller_test(df_totalquantity['qty'])


# # Choose p, d, and q

# In[25]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(df_totalquantity['qty'], lags=30, ax=ax1)
plot_pacf(df_totalquantity['qty'], lags=30, ax=ax2)

plt.tight_layout()
plt.show()


# In[26]:


aic_scores = []
# Fit the ARIMA model
model = ARIMA(df_totalquantity['qty'], order=(4,0,4))
model_fit = model.fit()
# Add AIC score to the list
aic_scores.append({'par': '(4,0,4)', 'aic': model_fit.aic})


# In[27]:


aic_scores


# In[28]:


from itertools import product

# Define ranges for p, d, and q
p = range(0, 5)  # 0 to 7
d = range(0, 3)  # 0 to 2
q = range(0, 5)  # 0 to 7

# Use the product function from itertools
# to create combinations of p, d, and q
pdq = list(product(p, d, q))
print(pdq)


# In[30]:


# Splitting data into training and testing with ratio 8 : 2
data_train = df_totalquantity[:292]["qty"]
data_test = df_totalquantity[292:]['qty']

# Creating a list to store AIC scores
aic_scores = []

# Performing manual grid search to find optimal p, d, q
for param in pdq:
    # Fitting the ARIMA model
    model = ARIMA(data_train, order=param)
    model_fit = model.fit()
    # Adding AIC score to the list
    aic_scores.append({'par': param, 'aic': model_fit.aic})

# Finding the smallest AIC score
best_aic = min(aic_scores, key=lambda x: x['aic'])

print(best_aic)

# Creating an ARIMA model with the best p, d, and q from grid search
model = ARIMA(data_train, order=(best_aic['par']))
model_fit = model.fit()

# Making predictions for the next 73 days (testing data)
preds = model_fit.forecast(73)


# In[31]:


preds.plot()


# In[32]:


import pmdarima as pm

auto_arima = pm.auto_arima(data_train,stepwise=False, seasonal=False)
forecast = auto_arima.predict(n_periods=73)

auto_arima.summary()


# In[33]:


# ploting
forecast.plot(label='auto arima')
preds.plot(label='grid search')

data_train.plot(label='train')
data_test.plot(label='test')
plt.legend()


# # Evaluate Model

# In[34]:


# Calculate RMSE for training data
train_predictions = model_fit.predict(start=data_train.index[0], end=data_train.index[-1])
train_rmse = np.sqrt(mean_squared_error(data_train, train_predictions))

# Calculate RMSE for testing data
test_rmse = np.sqrt(mean_squared_error(data_test, preds))

print(f"RMSE for Training Data: {train_rmse:.2f}")
print(f"RMSE for Testing Data: {test_rmse:.2f}")


# In[35]:


df_totalquantity.plot(figsize=(15, 6), alpha=0.5, marker="o")
preds.plot(linewidth=2, marker="o", legend=True)


# The parameter values p = 2, d = 1, and q = 3 resulted in an RMSE value of 14.37 dan RMSE for Training Data is 17.27.
# 
# The difference between the RMSE values for training (17.27) and testing (14.37) is not too large suggests that the model is not overfitting. A smaller difference indicates that the model is generalizing reasonably well to unseen data.

# In[37]:


from pandas.tseries.offsets import DateOffset

future_dates=[df_totalquantity.index[-1]+ DateOffset(days=x)for x in range(0,31)]
future_dates_df=pd.DataFrame(index=future_dates[1:],columns=df_totalquantity.columns)

future_df = pd.concat([df_totalquantity,future_dates_df])

model=ARIMA(df_totalquantity['qty'], order=(2,1,3))
model_fit=model.fit()

future_df['forecast'] = model_fit.predict(start = 0, end = 395, dynamic = False)
future_df[['qty', 'forecast']].plot(figsize=(12, 8))


# In[38]:


future_df.tail(30).mean()


# In[40]:


import pickle

# Creating an ARIMA model with the best p, d, and q from grid search
model = ARIMA(df_totalquantity['qty'], order=best_aic['par'])
model_fit = model.fit()

# Save the model to a file using pickle
model_filename = 'arima_model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(model_fit, model_file)

print("Model saved successfully!")


# In[42]:


import pickle
# Load the ARIMA model from the file
model_filename = 'arima_model.pkl'
with open(model_filename, 'rb') as model_file:
    loaded_model_fit = pickle.load(model_file)

# Number of days for prediction
num_days = 30

# Forecast the next 30 days
forecast = loaded_model_fit.forecast(steps=num_days)

print("Forecasted quantities for the next", num_days, "days:")
print(forecast)


# In[ ]:


forecast.plot()


# In[ ]:




