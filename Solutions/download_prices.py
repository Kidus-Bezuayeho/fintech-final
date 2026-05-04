import pandas as pd
import yfinance as yf
import os


# I load the herding events to get the tickers and date range needed
JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'question2_daily.json')
daily = pd.read_json(JSON)
daily['Date'] = pd.to_datetime(daily['Date'])
herd = daily[daily['rh_herd'] == 1]

tickers = herd['Ticker'].unique().tolist() + ['SPY']
start   = herd['Date'].min()
end     = herd['Date'].max()

print(f"Downloading {len(tickers)} tickers from {start.date()} to {end.date()}...")

prices = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=True)['Close']

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'prices.json')
prices.to_json(out, date_format='iso')

print(f"Saved to {out}")
