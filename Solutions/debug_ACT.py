import pandas as pd
import yfinance as yf
import os

# I check what ACT looks like in the raw Robintrack data
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Robintrack_subsample.csv')
df = pd.read_csv(DATA, parse_dates=['DateTime', 'Date'])
act_rows = df[df['Ticker'] == 'ACT'].head(5)

print()
print("=" * 60)
print("  ACT in Robintrack_subsample.csv")
print("=" * 60)
print(act_rows.to_string(index=False))
print("=" * 60)
print()

# I attempt to download ACT from yfinance to see what it returns
print("=" * 60)
print("  yfinance download attempt for ACT")
print("=" * 60)
data = yf.download('ACT', start='2018-01-01', end='2021-01-01', auto_adjust=True, progress=False)
if data.empty:
    print("  Result: no data returned (ticker likely delisted)")
else:
    print(data)
print("=" * 60)
print()
