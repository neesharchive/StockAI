import requests
import pandas as pd

def fetch_data_alpha_vantage(symbol, api_key):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={api_key}&outputsize=full'
    response = requests.get(url)
    data = response.json()

    if "Time Series (5min)" in data:
        intraday_data = pd.DataFrame(data["Time Series (5min)"]).transpose()
        intraday_data.columns = [col.split(' ')[1] for col in intraday_data.columns]
        return intraday_data
    else:
        print("Error fetching data or data is empty")
        return None
