import numpy as np
import neat
import pandas as pd
import yfinance as yf
import time

from visualizer import show_stock_visualizer

ticker = 'AAPL'

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
    stock_data['MA7'] = stock_data['Close'].rolling(window=7).mean()
    stats = {}
    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'MA7']:
        mean, std = stock_data[col].mean(), stock_data[col].std()
        stock_data[col] = (stock_data[col] - mean) / std
        stats[col] = (mean, std)
    stock_data.dropna(inplace=True)
    return stock_data, stats

def denormalize(value, mean, std):
    return (value * std) + mean

stock_visualizer = download_stock_data(ticker, '2024-04-01', '2024-04-30')
stock_data, stats = preprocess_data(download_stock_data(ticker, '2024-04-01', '2024-04-30'))
close_mean, close_std = stats['Close']

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_squared_error = 0
        predictions = []
        for i in range(len(stock_data) - 1):
            inputs = stock_data.iloc[i][['Open', 'High', 'Low', 'Close', 'Volume', 'MA7']].values
            output = net.activate(inputs)
            predicted_price = denormalize(output[0], close_mean, close_std)  # Extract the first element if output is a list
            predictions.append(predicted_price)
            expected_output = stock_data.iloc[i + 1]['Close']
            total_squared_error += (output[0] - expected_output)**2  # Make sure to use output[0] here as well
        mse = total_squared_error / (len(stock_data) - 1)
        genome.fitness = -mse
        genome.predictions = predictions  # Store predictions in the genome for later use

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config')
pop = neat.Population(config)
print("Running NEAT algorithm...")
winner = pop.run(eval_genomes, 50)  # Winner is the best genome after training
print("NEAT algorithm completed.")

predicted_prices = winner.predictions  # Get predictions stored in the genome
stock_data['Predicted'] = pd.Series(predicted_prices, index=stock_data.index[:-1])  # Align index

print(f"Predicted price for tomorrow: {predicted_prices[0]:.2f}")

show_stock_visualizer(stock_visualizer, ticker)

