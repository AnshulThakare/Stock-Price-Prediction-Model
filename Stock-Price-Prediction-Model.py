import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# the ticker symbol
ticker_symbol = 'AAPL'

# historical data
ticker = yf.Ticker(ticker_symbol)
hist = ticker.history(period='1y')

df = hist.reset_index()
df['Close_Pred'] = df['Close'].shift(-1)

df.dropna(inplace=True)

X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
y = df['Close_Pred']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')

# prepare data
ticker_symbol = 'AAPL'
ticker = yf.Ticker(ticker_symbol)
hist = ticker.history(period='1y')
df = hist.reset_index()
df['Close_Pred'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# target
X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
y = df['Close_Pred']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training model
model = LinearRegression()
model.fit(X_train, y_train)

# predictions
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')

# Training
model = LinearRegression()
param_grid = {'fit_intercept': [True, False]}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# predictions
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Prices')
plt.plot(predictions, label='Predicted Prices')
plt.legend()
plt.title('Actual vs Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()

import joblib
joblib.dump(best_model, 'stock_price_model.pkl')
