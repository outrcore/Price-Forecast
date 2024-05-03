import numpy as np
import neat
import pandas as pd
import yfinance as yf
import time
from sklearn.model_selection import TimeSeriesSplit

from visualizer import show_stock_visualizer

ticker = 'SPY'

def download_stock_data(ticker, start_date, end_date, max_retries=3, retry_delay=5):
    for attempt in range(max_retries):
        try:
            return yf.download(ticker, start=start_date, end=end_date)
        except ConnectionError as e:
            if attempt < max_retries - 1:
                print(f"Download failed: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise

def preprocess_data(stock_data):
    stock_data['MA3'] = stock_data['Close'].rolling(window=3).mean()
    stock_data['MA5'] = stock_data['Close'].rolling(window=5).mean()
    stock_data['MA7'] = stock_data['Close'].rolling(window=7).mean()
    
    # Calculate RSI
    delta = stock_data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = stock_data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = exp1 - exp2
    
    stats = {}
    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'MA3', 'MA5', 'MA7', 'RSI', 'MACD']:
        mean, std = stock_data[col].mean(), stock_data[col].std()
        stock_data[col] = (stock_data[col] - mean) / std
        stats[col] = (mean, std)
    stock_data.dropna(inplace=True)
    return stock_data, stats


def denormalize(value, mean, std):
    return (value * std) + mean

stock_visualizer = download_stock_data(ticker, '2024-04-01', '2024-05-01')
stock_data, stats = preprocess_data(download_stock_data(ticker, '2024-01-01', '2024-05-02'))
close_mean, close_std = stats['Close']

def eval_genomes(genomes, config, train_data, test_data, close_mean, close_std):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        total_squared_error = 0
        predictions = []

        for i in range(len(train_data) - 1):
            inputs = train_data.iloc[i][['Open', 'High', 'Low', 'Close', 'Volume', 'MA3', 'MA5', 'MA7', 'RSI', 'MACD']].values
            output = net.activate(inputs)
            predicted_price = denormalize(output[0], close_mean, close_std)
            predictions.append(predicted_price)

            expected_output = train_data.iloc[i + 1]['Close']
            total_squared_error += (output[0] - expected_output)**2

        mse = total_squared_error / (len(train_data) - 1)
        genome.fitness = -mse

        # Evaluate on the test set
        test_predictions = []
        for i in range(len(test_data) - 1):
            inputs = test_data.iloc[i][['Open', 'High', 'Low', 'Close', 'Volume', 'MA3', 'MA5', 'MA7', 'RSI', 'MACD']].values
            output = net.activate(inputs)
            predicted_price = denormalize(output[0], close_mean, close_std)
            test_predictions.append(predicted_price)

        genome.test_predictions = test_predictions

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config')


stock_visualizer = download_stock_data(ticker, '2024-04-01', '2024-05-1')
stock_data, stats = preprocess_data(download_stock_data(ticker, '2024-01-01', '2024-05-2'))
close_mean, close_std = stats['Close']

n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

best_genome = None
best_test_mse = float('inf')

for train_index, test_index in tscv.split(stock_data):
    train_data = stock_data.iloc[train_index]
    test_data = stock_data.iloc[test_index]

    pop = neat.Population(config)
    print(f"Running NEAT algorithm for split {tscv.get_n_splits() - n_splits + 1}/{tscv.get_n_splits()}...")
    winner = pop.run(lambda genomes, config: eval_genomes(genomes, config, train_data, test_data, close_mean, close_std), 50)

    # Evaluate the winner genome on the test set
    test_predictions = winner.test_predictions
    test_data['Predicted'] = pd.Series(test_predictions, index=test_data.index[:-1])
    test_squared_error = np.sum((test_data['Close'][1:] - test_data['Predicted'])**2)
    test_mse = test_squared_error / (len(test_data) - 1)

    if test_mse < best_test_mse:
        best_genome = winner
        best_test_mse = test_mse

    n_splits -= 1

print("NEAT algorithm completed.")

# Use the best genome for final prediction
predicted_prices = best_genome.test_predictions
stock_data['Predicted'] = pd.Series(predicted_prices, index=stock_data.index[-len(predicted_prices)-1:-1])
print(f"Best test MSE: {best_test_mse:.4f}")
print(f"Predicted price for tomorrow: {predicted_prices[-1]:.2f}")

show_stock_visualizer(stock_visualizer, ticker)