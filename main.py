import AVdata
import pandas as pd
import numpy as np
import sys  # Add this for command line arguments
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volatility import AverageTrueRange
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Function to calculate technical indicators
def calculate_indicators(dataframe):
    dataframe['close'] = pd.to_numeric(dataframe['close'], errors='coerce')
    dataframe['high'] = pd.to_numeric(dataframe['high'], errors='coerce')
    dataframe['low'] = pd.to_numeric(dataframe['low'], errors='coerce')

    dataframe['SMA'] = SMAIndicator(dataframe['close'], window=14).sma_indicator()
    dataframe['EMA'] = EMAIndicator(dataframe['close'], window=14).ema_indicator()
    dataframe['RSI'] = RSIIndicator(dataframe['close']).rsi()
    bollinger = BollingerBands(dataframe['close'])
    dataframe['Bollinger_High'] = bollinger.bollinger_hband()
    dataframe['Bollinger_Low'] = bollinger.bollinger_lband()
    dataframe['Supertrend'] = calculate_supertrend(dataframe)

    return dataframe.dropna()

# Function to calculate Supertrend
def calculate_supertrend(dataframe):
    atr = AverageTrueRange(dataframe['high'], dataframe['low'], dataframe['close'], window=14).average_true_range()
    multiplier = 3.0
    hl2 = (dataframe['high'] + dataframe['low']) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    return upperband

# Function to categorize stock movement
def categorize_movement(dataframe, threshold=0.3, look_forward=12):
    dataframe = dataframe.copy()
    dataframe['Future_Price'] = dataframe['close'].shift(-look_forward)
    dataframe['Price_Change'] = ((dataframe['Future_Price'] - dataframe['close']) / dataframe['close']) * 100
    
    conditions = [
        dataframe['Price_Change'] > threshold,
        dataframe['Price_Change'] < -threshold,
        abs(dataframe['Price_Change']) <= threshold
    ]
    choices = [2, 0, 1]  # Buy, Sell, Hold
    dataframe.loc[:, 'Movement'] = np.select(conditions, choices, default=np.nan)
    
    return dataframe.dropna()

# Function to fetch and prepare data
def fetch_and_prepare_data(symbol, api_key):
    try:
        intraday_data = AVdata.fetch_data_alpha_vantage(symbol, api_key)
        if intraday_data is None:
            raise ValueError("API request limit reached or no data returned")
        
        prepared_data = calculate_indicators(intraday_data)
        prepared_data = categorize_movement(prepared_data)
        return prepared_data
    except ValueError as e:
        print(e)
        return None

def run_intraday_prediction(symbol):

    # Fetch intraday data from Alpha Vantage
    api_key = '57LRNX0IZLI86I2U'  # Replace with your actual API key
    prepared_data = fetch_and_prepare_data(symbol, api_key)

    # Create a figure with custom background color
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0A0E2A')  # Deep bluish background color

    # Define num_predictions_to_display here to ensure it's defined even if there's an error while fetching data
    num_predictions_to_display = 6

    if prepared_data is not None and len(prepared_data) > 0:
        # Feature Engineering
        features = ['SMA', 'EMA', 'RSI', 'Bollinger_High', 'Bollinger_Low', 'Supertrend']
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(prepared_data[features])

        # Prepare the dataset for XGBoost
        X = scaled_features[:-12]  # Exclude the last 12 intervals (last hour)
        y = prepared_data['Movement'][12:].values

        # Check if data is sufficient for model training
        if len(X) > 0 and len(y) > 0:
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize XGBoost classifier
            model = xgb.XGBClassifier(objective='multi:softmax', num_class=3)

            # Train the model
            model.fit(X_train, y_train)

            # Making predictions
            predicted = model.predict(X_test)

            # Visualization (Limiting to the last few hours of the trading day)
            for i in range(num_predictions_to_display):
                pred = predicted[-(i+1)]
                if pred == 2:  # Buy
                    ax.bar(num_predictions_to_display-i, 1, color='green', edgecolor='cyan', linewidth=2)
                elif pred == 0:  # Sell
                    ax.bar(num_predictions_to_display-i, -1, color='red', edgecolor='cyan', linewidth=2)
                elif pred == 1:  # Hold
                    ax.scatter(num_predictions_to_display-i, 0, color='blue', marker='o', s=100, edgecolor='cyan')
        else:
            print("Insufficient data for model training.")
    else:
        print("Failed to fetch or prepare data.")
        ax.text(0.5, 0.5, "API Limit Reached\nPlease Try Again Tomorrow", color='white', fontsize=16, ha='center',va='center',transform=ax.transAxes)

    # Customize plot
    ax.set_title('Hourly Stock Movement Prediction for Next Hours', color='white', fontsize=16)
    ax.set_xlabel('Hours from Now', color='white')
    ax.set_ylabel('Predictions (2: Buy, -1: Sell, 0: Hold)', color='white')
    ax.set_yticks([-1, 0, 1])
    ax.set_xticks(range(1, num_predictions_to_display + 1))
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='both', colors='white')
    ax.grid(axis='y', linestyle='--', color='yellow', alpha=0.6)
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.set_facecolor('#0A0E2A')

    plt.show()
if __name__ == "__main__":
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
        run_intraday_prediction(symbol)
    else:
        print("No symbol provided.")