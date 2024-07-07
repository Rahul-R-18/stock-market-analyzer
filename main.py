import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Data Collection
nvda_data = yf.download('NVDA', start='2010-01-01', end='2023-07-01')
nvda_data.to_csv('nvda_data.csv')

# Step 2: Feature Engineering
nvda_data = pd.read_csv('nvda_data.csv', index_col='Date', parse_dates=True)
nvda_data['MA_5'] = nvda_data['Close'].rolling(window=5).mean()
nvda_data['MA_20'] = nvda_data['Close'].rolling(window=20).mean()
nvda_data['MA_50'] = nvda_data['Close'].rolling(window=50).mean()
nvda_data['Daily_Return'] = nvda_data['Close'].pct_change()
nvda_data['Volatility'] = nvda_data['Close'].rolling(window=20).std()
nvda_data['Volume_Mean'] = nvda_data['Volume'].rolling(window=20).mean()

# RSI calculation
delta = nvda_data['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
nvda_data['RSI'] = 100 - (100 / (1 + rs))

# MACD calculation
ema_12 = nvda_data['Close'].ewm(span=12, adjust=False).mean()
ema_26 = nvda_data['Close'].ewm(span=26, adjust=False).mean()
nvda_data['MACD'] = ema_12 - ema_26
nvda_data['Signal_Line'] = nvda_data['MACD'].ewm(span=9, adjust=False).mean()

# Drop rows with NaN values
nvda_data = nvda_data.dropna()

# Define the target variable
nvda_data['Target'] = nvda_data['Close'].shift(-1)
nvda_data = nvda_data.dropna()

nvda_data.to_csv('nvda_features.csv')

# Step 3: Model Training
nvda_data = pd.read_csv('nvda_features.csv', index_col='Date', parse_dates=True)
features = ['MA_5', 'MA_20', 'MA_50', 'Daily_Return', 'Volatility', 'Volume_Mean', 'RSI', 'MACD', 'Signal_Line']
target = 'Target'
X = nvda_data[features]
y = nvda_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Step 4: Prediction
def predict_next_day(data, model):
    last_row = data.iloc[-1]
    features = np.array([
        last_row['MA_5'], 
        last_row['MA_20'], 
        last_row['MA_50'], 
        last_row['Daily_Return'], 
        last_row['Volatility'], 
        last_row['Volume_Mean'], 
        last_row['RSI'], 
        last_row['MACD'], 
        last_row['Signal_Line']
    ]).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

next_day_prediction = predict_next_day(nvda_data, model)
print(f'Predicted next day closing price: {next_day_prediction}')

# Visualization
plt.figure(figsize=(14, 10))

# Plot the closing price
plt.subplot(3, 1, 1)
plt.plot(nvda_data.index, nvda_data['Close'], label='Close Price', color='blue')
plt.plot(nvda_data.index, nvda_data['MA_5'], label='MA 5', color='red')
plt.plot(nvda_data.index, nvda_data['MA_20'], label='MA 20', color='green')
plt.plot(nvda_data.index, nvda_data['MA_50'], label='MA 50', color='orange')
plt.title('NVDA Stock Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Plot the RSI
plt.subplot(3, 1, 2)
plt.plot(nvda_data.index, nvda_data['RSI'], label='RSI', color='purple')
plt.axhline(70, linestyle='--', alpha=0.5, color='red')
plt.axhline(30, linestyle='--', alpha=0.5, color='green')
plt.title('Relative Strength Index (RSI)')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()

# Plot the MACD and Signal Line
plt.subplot(3, 1, 3)
plt.plot(nvda_data.index, nvda_data['MACD'], label='MACD', color='blue')
plt.plot(nvda_data.index, nvda_data['Signal_Line'], label='Signal Line', color='red')
plt.title('MACD and Signal Line')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()
