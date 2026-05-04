import pandas as pd
import os


DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Robintrack_subsample.csv')
df = pd.read_csv(DATA, parse_dates=['DateTime', 'Date'])
df = df.sort_values(['Ticker', 'DateTime'])

# users_close = last Users observation in 2-4 PM ET window
# 2 PM ET = 270 min after 9:30 AM open; 4 PM ET = 390 min after open

mins_since_open = (df['DateTime'].dt.hour - 9) * 60 + df['DateTime'].dt.minute - 30
df_win = df[(mins_since_open >= 270) & (mins_since_open <= 390)].copy()

daily_all = (df_win.groupby(['Ticker', 'Date'], as_index=False)
                   .agg(users_close=('Users', 'last')))

# Step 2: userchg and userratio (day-over-day, prior available obs per ticker)
daily_all = daily_all.sort_values(['Ticker', 'Date'])
prev = daily_all.groupby('Ticker')['users_close'].shift(1)
date_diff = daily_all.groupby('Ticker')['Date'].diff()
daily_all['userchg']   = daily_all['users_close'] - prev
daily_all['userratio'] = (daily_all['users_close'] / prev).replace([float('inf'), float('-inf')], float('nan'))
# only keep rows where the prior obs is at most 3 calendar days back (handles gaps)
daily_all.loc[date_diff > pd.Timedelta('3 days'), ['userchg', 'userratio']] = float('nan')
daily = daily_all.dropna(subset=['userchg', 'userratio'])

# Panel A 
vars_a = ['users_close', 'userchg', 'userratio']
panelA = pd.DataFrame({
    'N':    [daily[v].count()        for v in vars_a],
    'Mean': [daily[v].mean()         for v in vars_a],
    'SD':   [daily[v].std()          for v in vars_a],
    'Min':  [daily[v].min()          for v in vars_a],
    'p25':  [daily[v].quantile(.25)  for v in vars_a],
    'p50':  [daily[v].quantile(.50)  for v in vars_a],
    'p75':  [daily[v].quantile(.75)  for v in vars_a],
    'Max':  [daily[v].max()          for v in vars_a],
}, index=vars_a).round(4)
print("Panel A:\n", panelA)

# Panel B
date_agg = daily_all.groupby('Date').agg(
    n_stocks        =('Ticker',      'nunique'),
    users_close_sum =('users_close', 'sum'),
).reset_index()

print("\nPanel B:")
print(f"  Mean # unique tickers per day:              {date_agg['n_stocks'].mean():.4f}")
print(f"  Mean total user positions (summed/day):     {date_agg['users_close_sum'].mean():.4f}")
print(f"  Median total user positions (summed/day):   {date_agg['users_close_sum'].median():.4f}")

