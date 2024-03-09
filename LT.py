import yfinance as yf
import numpy as np
import pandas as pd
import sys  # Add this for command line arguments
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Function to fetch historical data
def fetch_historical_data(symbol, period='15y'):
    stock = yf.Ticker(symbol)
    return stock.history(period=period)

# Function to calculate SMA, Returns, and other features
def calculate_features(dataframe):
    dataframe['SMA_50'] = dataframe['Close'].rolling(window=50).mean()
    dataframe['1D_Returns'] = dataframe['Close'].pct_change(1)  # 1 day
    dataframe['1W_Returns'] = dataframe['Close'].pct_change(5)  # 1 week
    dataframe['2W_Returns'] = dataframe['Close'].pct_change(10) # 2 weeks
    dataframe['3W_Returns'] = dataframe['Close'].pct_change(15) # 3 weeks
    dataframe['1M_Returns'] = dataframe['Close'].pct_change(21) # 1 month
    return dataframe.dropna()

# Function to create dataset for LSTM
def create_dataset(X, y, time_step=1):
    Xs, ys = [], []
    for i in range(len(X) - time_step):
        v = X.iloc[i:(i + time_step)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_step])
    return np.array(Xs), np.array(ys)

# Function to generate a single prediction using a model
def generate_prediction(model, data, scaler_feature, scaler_target, time_step):
    # Scale input data
    input_data = scaler_feature.transform(data)[-time_step:].reshape(1, time_step, data.shape[1])

    # Generate prediction
    predicted_scaled_value = model.predict(input_data)

    # Inverse transform the prediction to the original scale
    predicted_value = scaler_target.inverse_transform(predicted_scaled_value)
    return predicted_value[0, 0]

def run_long_term_prediction(symbol):

    # Fetch data and calculate features
    historical_data = fetch_historical_data(symbol)
    historical_data = calculate_features(historical_data)

    time_step = 100  # Time step for LSTM
    intervals = ['1D_Returns', '1W_Returns', '2W_Returns', '3W_Returns', '1M_Returns']
    models = {}
    scalers = {}
    target_scalers = {}

    # Standardize features and create models for each interval
    for interval in intervals:
        print(f"Processing for {interval}")
        features = ['SMA_50', interval]

        # Feature scaler
        scaler_feature = StandardScaler()
        scaled_features = scaler_feature.fit_transform(historical_data[features])
        scalers[interval] = scaler_feature

        # Target scaler
        scaler_target = StandardScaler()
        target = scaler_target.fit_transform(historical_data[interval].values.reshape(-1, 1))
        target_scalers[interval] = scaler_target

        X, y = create_dataset(pd.DataFrame(scaled_features, columns=features), pd.Series(target.flatten()), time_step)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Building the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1)  # Reduced epochs for faster processing

        models[interval] = model

    # Visualization customization
    plt.style.use('dark_background')
    plt.rcParams['axes.facecolor'] = '#0A0E2A'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = '#0A0E2A'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['grid.color'] = 'yellow'

    # Visualization in a single window
    fig, axs = plt.subplots(len(intervals), 1, figsize=(10, 20))
    fig.tight_layout(pad=5.0)

    # Prepare the input data for each model and generate predictions
    last_values = historical_data.iloc[-time_step:]

    for i, interval in enumerate(intervals):
        model = models[interval]
        scaler_feature = scalers[interval]
        scaler_target = target_scalers[interval]

        interval_features = ['SMA_50', interval]
        data = historical_data[interval_features].values[-time_step:]

        prediction = generate_prediction(model, data, scaler_feature, scaler_target, time_step)

        predicted = prediction * 100  # Convert to percentage
        color = 'green' if predicted > 0 else 'red'

        axs[i].bar([''], [predicted], color=color)
        axs[i].set_title(f'{interval} Predicted Returns')
        axs[i].set_ylabel('Predicted Returns (%)')
        axs[i].grid()

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
        run_long_term_prediction(symbol)
    else:
        print("No symbol provided.")