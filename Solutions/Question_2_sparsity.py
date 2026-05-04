import pandas as pd
import os

# I reuse the same data prep from Question 1 to get one closing price per ticker per day
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Robintrack_subsample.csv')
df = pd.read_csv(DATA, parse_dates=['DateTime', 'Date'])
df = df.sort_values(['Ticker', 'DateTime'])
mins_since_open = (df['DateTime'].dt.hour - 9) * 60 + df['DateTime'].dt.minute - 30
df_win = df[(mins_since_open >= 270) & (mins_since_open <= 390)].copy()
daily_all = (df_win.groupby(['Ticker', 'Date'], as_index=False).agg(users_close=('Users', 'last')))
daily_all = daily_all.sort_values(['Ticker', 'Date'])

# I calculate how many days passed between each observation for each ticker
date_diff = daily_all.groupby('Ticker')['Date'].diff()

# I count how many stocks each day have a previous observation within 3 days
stocks_with_prev = daily_all[date_diff <= pd.Timedelta('3 days')].groupby('Date')['Ticker'].nunique()

# I calculate the average gap to show how sparse the data is across tickers
avg_gap = date_diff.dt.days.mean()

print()
print("=" * 50)
print(f"  Stocks per day with prev observation <= 3 days")
print("=" * 50)
print(f"  Average: {stocks_with_prev.mean():.2f}")
print(f"  Median:  {stocks_with_prev.median():.0f}")
print("=" * 50)
print()
print("=" * 50)
print(f"  Avg gap between observations per ticker")
print("=" * 50)
print(f"  {avg_gap:.2f} days")
print("=" * 50)
print()
