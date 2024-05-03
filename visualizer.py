# visualizer.py

import pandas as pd
import matplotlib.pyplot as plt

# Read the data
data = pd.read_csv('visualize_data.csv', index_col=0, parse_dates=True)

# Plot the actual and predicted data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Actual')
plt.plot(data.index[-30:], data['Close'][-30:], label='Predicted')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Actual vs Predicted Closing Prices')
plt.legend()
plt.show()