import pandas as pd
import matplotlib.pyplot as plt
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# I rebuild the daily panel from raw CSV with no 3-day gap restriction
DATA = os.path.join(BASE, '..', 'Robintrack_subsample.csv')
df = pd.read_csv(DATA, parse_dates=['DateTime', 'Date'])
df = df.sort_values(['Ticker', 'DateTime'])

mins_since_open = (df['DateTime'].dt.hour - 9) * 60 + df['DateTime'].dt.minute - 30
df_win = df[(mins_since_open >= 270) & (mins_since_open <= 390)].copy()
daily_all = df_win.groupby(['Ticker', 'Date'], as_index=False).agg(users_close=('Users', 'last'))
daily_all = daily_all.sort_values(['Ticker', 'Date'])
daily_all['users_close_prev'] = daily_all.groupby('Ticker')['users_close'].shift(1)
daily_all['userchg'] = daily_all['users_close'] - daily_all['users_close_prev']
daily_all['userratio'] = daily_all['users_close'] / daily_all['users_close_prev']
daily = daily_all.dropna(subset=['userchg']).copy()
daily['Date'] = pd.to_datetime(daily['Date'])

# I recompute rh_herd on this no-gap panel
qualifying = daily[(daily['userratio'] > 1) & (daily['users_close_prev'] >= 100)]
day_threshold = qualifying.groupby('Date')['userratio'].quantile(0.995).rename('threshold')
daily = daily.merge(day_threshold, on='Date', how='left')
daily['rh_herd'] = (
    (daily['userratio'] > 1) &
    (daily['users_close_prev'] >= 100) &
    (daily['userratio'] >= daily['threshold'])
).astype(int)

print(f"Total herding events (no-gap panel): {daily['rh_herd'].sum()}")

herd = daily[daily['rh_herd'] == 1].copy()

# I load prices and compute daily returns
prices = pd.read_json(os.path.join(BASE, 'data', 'prices.json'))
prices.index = pd.to_datetime(prices.index)
returns = prices.pct_change()
trading_days = prices.index

rel_days     = list(range(-10, 21))
userchg_rows = []
bhar_rows    = []

for _, row in herd.iterrows():
    ticker     = row['Ticker']
    event_date = row['Date']

    if event_date not in trading_days:
        continue
    pos = trading_days.get_loc(event_date)

    for rd in rel_days:
        target_pos = pos + rd
        if target_pos < 0 or target_pos >= len(trading_days):
            continue
        target_date = trading_days[target_pos]

        match = daily[(daily['Ticker'] == ticker) & (daily['Date'] == target_date)]
        if not match.empty:
            userchg_rows.append({'rel_day': rd, 'userchg': match.iloc[0]['userchg']})

        if ticker in returns.columns and target_date in returns.index:
            ar = returns.loc[target_date, ticker] - returns.loc[target_date, 'SPY']
            bhar_rows.append({
                'rel_day': rd,
                'event':   f"{ticker}_{event_date.date()}",
                'ar':      ar,
            })

uc_df   = pd.DataFrame(userchg_rows)
uc_mean = uc_df.groupby('rel_day')['userchg'].mean()

bhar_df     = pd.DataFrame(bhar_rows)
event_bhars = []

for event_id, grp in bhar_df.groupby('event'):
    grp = grp.sort_values('rel_day').set_index('rel_day')
    for rd in rel_days:
        if rd not in grp.index:
            continue
        if rd >= 0:
            window = grp.loc[0:rd, 'ar']
        else:
            window = grp.loc[rd:0, 'ar']
        bhar = (1 + window).prod() - 1
        event_bhars.append({'rel_day': rd, 'bhar': bhar})

bhar_mean = pd.DataFrame(event_bhars).groupby('rel_day')['bhar'].mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Question 4b — No 3-Day Gap Restriction (Test)', fontsize=12)

ax1.bar(rel_days, uc_mean.reindex(rel_days), color='steelblue', width=0.8)
ax1.axvline(0, color='red', linestyle='--', linewidth=1.2)
ax1.set_xlabel('Day Relative to Herding Event')
ax1.set_ylabel('Mean userchg')
ax1.set_title('Mean User Change Around Herding Event')

ax2.plot(rel_days, bhar_mean.reindex(rel_days) * 100, color='green', marker='o', markersize=3)
ax2.axvline(0, color='red', linestyle='--', linewidth=1.2)
ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_xlabel('Day Relative to Herding Event')
ax2.set_ylabel('BHAR (%)')
ax2.set_title('Buy-and-Hold Abnormal Return Around Herding Event')

plt.tight_layout()
out = os.path.join(BASE, '..', 'write-up', 'tables', '4b_without_restrictions.png')
plt.savefig(out, dpi=150)
plt.show()
