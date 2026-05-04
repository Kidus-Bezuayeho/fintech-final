import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# I load the daily panel with rh_herd flags computed by Question_2.py
daily = pd.read_json(os.path.join(BASE, 'data', 'question2_daily.json'))
daily['Date'] = pd.to_datetime(daily['Date'])

# ── Question 4(b) ────────────────────────────────────────────────────────────

# I isolate the 366 herding events
herd = daily[daily['rh_herd'] == 1].copy()

# I load prices and compute daily returns; prices.index is the trading calendar
prices = pd.read_json(os.path.join(BASE, 'data', 'prices.json'))
prices.index = pd.to_datetime(prices.index)
returns = prices.pct_change()
trading_days = prices.index

# I collect userchg and abnormal returns for each event at relative days -10 to +20
rel_days     = list(range(-10, 21))
userchg_rows = []
bhar_rows    = []

for _, row in herd.iterrows():
    ticker     = row['Ticker']
    event_date = row['Date']

    # I skip events where the event date is not in the trading calendar (delisted)
    
    if event_date not in trading_days:
        continue
    pos = trading_days.get_loc(event_date)

    for rd in rel_days:
        target_pos = pos + rd
        if target_pos < 0 or target_pos >= len(trading_days):
            continue
        target_date = trading_days[target_pos]

        # I look up userchg for this ticker on this date in the Robintrack panel
        match = daily[(daily['Ticker'] == ticker) & (daily['Date'] == target_date)]
        if not match.empty:
            userchg_rows.append({'rel_day': rd, 'userchg': match.iloc[0]['userchg']})

        # I compute the abnormal return as stock return minus SPY return on this day
        if ticker in returns.columns and target_date in returns.index:
            ar = returns.loc[target_date, ticker] - returns.loc[target_date, 'SPY']
            bhar_rows.append({
                'rel_day': rd,
                'event':   f"{ticker}_{event_date.date()}",
                'ar':      ar,
            })

# I compute mean userchg at each relative day across all events

uc_df   = pd.DataFrame(userchg_rows)
uc_mean = uc_df.groupby('rel_day')['userchg'].mean()

# I compute BHAR for each event at each relative day, then average across events
# BHAR at day t = product of (1 + AR) from day 0 to day t, minus 1

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

# I plot both panels side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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
out = os.path.join(BASE, '..', 'write-up', 'tables', 'question4b.png')
plt.savefig(out, dpi=150)
plt.show()

# ── Question 4(c) ────────────────────────────────────────────────────────────

# I rebuild a no-gap panel from the raw CSV for comparison
DATA = os.path.join(BASE, '..', 'Robintrack_subsample.csv')
df = pd.read_csv(DATA, parse_dates=['DateTime', 'Date'])
df = df.sort_values(['Ticker', 'DateTime'])
mins_since_open = (df['DateTime'].dt.hour - 9) * 60 + df['DateTime'].dt.minute - 30
df_win = df[(mins_since_open >= 270) & (mins_since_open <= 390)].copy()
daily_all = df_win.groupby(['Ticker', 'Date'], as_index=False).agg(users_close=('Users', 'last'))
daily_all = daily_all.sort_values(['Ticker', 'Date'])
daily_all['users_close_prev'] = daily_all.groupby('Ticker')['users_close'].shift(1)
daily_all['userratio'] = daily_all['users_close'] / daily_all['users_close_prev']
daily_nogap = daily_all.dropna(subset=['userratio']).copy()

# daily (3-day gap panel) is already loaded above from question2_daily.json
cutoffs = np.arange(1.1, 8.6, 0.5)
counts_nogap = []
counts_gap   = []

for c in cutoffs:
    counts_nogap.append(daily_nogap[(daily_nogap['userratio'] >= c) & (daily_nogap['users_close_prev'] >= 100)].shape[0])
    counts_gap.append(daily[(daily['userratio'] >= c) & (daily['users_close_prev'] >= 100)].shape[0])

import matplotlib.ticker as ticker

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(cutoffs, counts_nogap, width=0.4, color='green', alpha=0.8)
ax1.set_yscale('log')
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax1.set_xlabel('User Change Ratio Cutoff')
ax1.set_ylabel('Number of Events (log scale)')
ax1.set_title('No 3-Day Gap Restriction')
ax1.set_xticks(cutoffs)
ax1.set_xticklabels([f'{c:.1f}' for c in cutoffs], rotation=45)

ax2.bar(cutoffs, counts_gap, width=0.4, color='steelblue', alpha=0.8)
ax2.set_yscale('log')
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax2.set_xlabel('User Change Ratio Cutoff')
ax2.set_ylabel('Number of Events (log scale)')
ax2.set_title('With 3-Day Gap Restriction')
ax2.set_xticks(cutoffs)
ax2.set_xticklabels([f'{c:.1f}' for c in cutoffs], rotation=45)

fig.suptitle('Panel A: Herding Intensity and Daily User Change Ratio Cutoff', fontsize=13)
plt.tight_layout()
out = os.path.join(BASE, '..', 'write-up', 'tables', 'question4c.png')
plt.savefig(out, dpi=150)
plt.show()
