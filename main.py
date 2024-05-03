# main.py

import sys
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras import Sequential
from keras.layers import Dense, LSTM
import yfinance as yf

# Fetch historical price data
stock_symbol = "AAPL"
start_date = "2024-01-01"
end_date = "2024-05-01"
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Preprocess the data
data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
data['HL_PCT'] = (data['High'] - data['Low']) / data['Close'] * 100.0
data['PCT_change'] = (data['Close'] - data['Open']) / data['Open'] * 100.0

# Add moving averages
data['MA_3'] = data['Close'].rolling(window=3).mean()
data['MA_5'] = data['Close'].rolling(window=5).mean()
data['MA_7'] = data['Close'].rolling(window=7).mean()

# Add RSI
delta = data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# Add MACD
ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema_12 - ema_26

# Add standard deviation
data['STD_5'] = data['Close'].rolling(window=5).std()
data['STD_10'] = data['Close'].rolling(window=10).std()

data = data.dropna()

forecast_col = 'Close'
forecast_out = 30
data['label'] = data[forecast_col].shift(-forecast_out)

X = data.drop(['label', 'Open', 'High', 'Low'], axis=1)
X = X[:-forecast_out]
y = data['label'][:-forecast_out]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the data for LSTM
X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Train the base models
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(1, X_train.shape[1])))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=0)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

svm_model = SVR(kernel='rbf', C=1e3, gamma=0.1)
svm_model.fit(X_train, y_train)

# Make predictions using the base models
lstm_preds = lstm_model.predict(X_test_lstm)
rf_preds = rf_model.predict(X_test)
svm_preds = svm_model.predict(X_test)

# Combine the predictions using stacking
stacked_preds = np.column_stack((lstm_preds, rf_preds, svm_preds))

meta_model = RandomForestRegressor(n_estimators=100, random_state=42)
meta_model.fit(stacked_preds, y_test)

# Evaluate the stacked model
final_preds = meta_model.predict(stacked_preds)
mse = mean_squared_error(y_test, final_preds)
print(f"Stacked Model MSE: {mse:.4f}")

# Prepare the data for visualization
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date, periods=forecast_out + 1, freq='B')  # 'B' for business days
future_data = pd.DataFrame(index=future_dates, columns=['Close'])
future_data['Close'] = np.concatenate((data['Close'][-1:], final_preds))

# Combine actual and predicted data
visualize_data = pd.concat([data[['Close']], future_data[1:]])

# Save the data for visualization
visualize_data.to_csv('visualize_data.csv')

print("Data saved for visualization. Run visualizer.py to see the plot.")