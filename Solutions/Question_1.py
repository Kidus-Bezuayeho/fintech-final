import pandas as pd
import os


DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Robintrack_subsample.csv')
df = pd.read_csv(DATA, parse_dates=['DateTime', 'Date'])
df = df.sort_values(['Ticker', 'DateTime'])

# I filter to the 2-4 PM ET window to get the closing price for each day
# 2 PM ET = 270 min after 9:30 AM open; 4 PM ET = 390 min after open
mins_since_open = (df['DateTime'].dt.hour - 9) * 60 + df['DateTime'].dt.minute - 30
df_win = df[(mins_since_open >= 270) & (mins_since_open <= 390)].copy()

# I take the last observation in that window as the closing user count for each ticker-day
daily_all = (df_win.groupby(['Ticker', 'Date'], as_index=False)
                   .agg(users_close=('Users', 'last')))

# I calculate userchg and userratio relative to the prior available observation
daily_all = daily_all.sort_values(['Ticker', 'Date'])
prev = daily_all.groupby('Ticker')['users_close'].shift(1)
date_diff = daily_all.groupby('Ticker')['Date'].diff()
daily_all['userchg']   = daily_all['users_close'] - prev
daily_all['userratio'] = (daily_all['users_close'] / prev).replace([float('inf'), float('-inf')], float('nan'))
# I drop observations where the prior day is more than 3 calendar days back, since the ratio wouldn't be meaningful
daily_all.loc[date_diff > pd.Timedelta('3 days'), ['userchg', 'userratio']] = float('nan')
daily = daily_all.dropna(subset=['userchg', 'userratio'])

# Panel A — summary stats for the three main variables
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

# Panel B — I use daily_all here (not daily) so the ticker and position counts aren't affected by the gap filter
date_agg = daily_all.groupby('Date').agg(
    n_stocks        =('Ticker',      'nunique'),
    users_close_sum =('users_close', 'sum'),
).reset_index()

print("\nPanel B:")
print(f"  Mean # unique tickers per day:              {date_agg['n_stocks'].mean():.4f}")
print(f"  Mean total user positions (summed/day):     {date_agg['users_close_sum'].mean():.4f}")
print(f"  Median total user positions (summed/day):   {date_agg['users_close_sum'].median():.4f}")

# I save the daily panel to a JSON file so Question 2 can load it without redoing the data prep
# I add users_close_prev to a copy so Panel A and B above are not affected
out = daily.copy()
out['users_close_prev'] = daily_all.groupby('Ticker')['users_close'].shift(1).reindex(daily.index)
out.to_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'question1_daily.json'),
            orient='records', date_format='iso')

# I also save the date-level aggregation so Question 2d can plot it without re-aggregating
date_agg.to_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'question1_date_agg.json'),
                 orient='records', date_format='iso')

