import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# I load the daily panel saved by Question 1
JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'question1_daily.json')
daily = pd.read_json(JSON)
daily['Date'] = pd.to_datetime(daily['Date'])

# (a) rh_herd indicator — a stock is a herding event if it meets all three criteria:
# 1. userratio(t) > 1 — users increased from the prior day
# 2. users_close(t-1) >= 100 — at least 100 users the prior day
# 3. userratio(t) in top 0.5% of qualifying stocks on that day

# I first find all stocks that meet criteria 1 and 2
qualifying = daily[(daily['userratio'] > 1) & (daily['users_close_prev'] >= 100)]

# I then compute the 99.5th percentile threshold of userratio for each day among qualifying stocks
day_threshold = (qualifying.groupby('Date')['userratio']
                            .quantile(0.995)
                            .rename('threshold'))

# I merge the threshold back so I can apply criterion 3
daily = daily.merge(day_threshold, on='Date', how='left')

# I flag stocks that meet all three criteria
daily['rh_herd'] = (
    (daily['userratio'] > 1) &
    (daily['users_close_prev'] >= 100) &
    (daily['userratio'] >= daily['threshold'])
).astype(int)

print()
print("=" * 50)
print("  (a) rh_herd summary")
print("=" * 50)
print(f"  Total herding events (rh_herd=1): {daily['rh_herd'].sum()}")
print("=" * 50)
print()

# (b) Total herding events and average per day
total_events = daily['rh_herd'].sum()
unique_days = daily['Date'].nunique()
avg_per_day = total_events / unique_days

print()
print("=" * 50)
print("  (b) Herding events vs. paper benchmark")
print("=" * 50)
print(f"  Total herding events:    {total_events}")
print(f"  Unique days in sample:   {unique_days}")
print(f"  Avg herding events/day:  {avg_per_day:.4f}")
print(f"  Paper benchmark:         ~9 events/day")
print(f"  Difference:              {9 - avg_per_day:.4f} fewer per day")
print("=" * 50)
print()

# I save the full daily panel with rh_herd for reference
daily.to_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'question2_daily.json'),
              orient='records', date_format='iso')

# (c) Match each herding event to its return and the SPY return on that day
prices = pd.read_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'prices.json'))
prices.index = pd.to_datetime(prices.index)
returns = prices.pct_change()

herd = daily[daily['rh_herd'] == 1].copy()
herd = herd.set_index('Date')

herd['returns'] = [
    returns.loc[d, t] if d in returns.index and t in returns.columns else float('nan')
    for d, t in zip(herd.index, herd['Ticker'])
]
herd['spy_ret'] = returns['SPY'].reindex(herd.index).values

# I compute abnormal return as the stock's daily return minus SPY's return on the same day
herd['abnormal_returns'] = herd['returns'] - herd['spy_ret']
herd = herd.reset_index()

def stats(s):
    return pd.Series({
        'Mean': s.mean(),
        'SD':   s.std(),
        'Min':  s.min(),
        'p25':  s.quantile(0.25),
        'p50':  s.quantile(0.50),
        'p75':  s.quantile(0.75),
        'Max':  s.max(),
        'N':    s.count(),
    })

vars_c = {
    'users_close':      herd['users_close'],
    'userchg':          herd['userchg'],
    'userratio':        herd['userratio'],
    'returns':          herd['returns'],
    'abnormal_returns': herd['abnormal_returns'],
}

panel_c = pd.DataFrame({k: stats(v) for k, v in vars_c.items()}).T
panel_c = panel_c[['N', 'Mean', 'SD', 'Min', 'p25', 'p50', 'p75', 'Max']].round(4)

print()
print("=" * 90)
print("  (c) Summary statistics — herding event subsample (rh_herd = 1)")
print("=" * 90)
print(panel_c.to_string())
print("=" * 90)
print()

# I identify tickers with no return data — these are delisted stocks yfinance cannot retrieve
missing = herd[herd['returns'].isna()][['Date', 'Ticker']].reset_index(drop=True)
missing['Date'] = missing['Date'].dt.strftime('%Y-%m-%d')

print()
print("=" * 50)
print(f"  Tickers with no return data (delisted)")
print("=" * 50)
print(f"  Total missing: {len(missing)} of {len(herd)} herding events")
print(f"  Unique tickers: {missing['Ticker'].nunique()}")
print("-" * 50)
print(missing.to_string(index=False))
print("=" * 50)
print()

# (d) Plot total Robinhood user-stock positions over time
date_agg = pd.read_json(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'question1_date_agg.json')
)
date_agg['Date'] = pd.to_datetime(date_agg['Date'])
date_agg = date_agg.sort_values('Date')

# I mark March 13, 2020 — the date the US declared a national COVID-19 emergency
covid_date = pd.Timestamp('2020-03-13')

fig, ax = plt.subplots(figsize=(10, 4))

# I plot the daily total across all stocks, scaled to millions for readability
ax.plot(date_agg['Date'], date_agg['users_close_sum'] / 1e6, color='steelblue', linewidth=1)

# I add a vertical dashed line at the COVID emergency declaration
ax.axvline(covid_date, color='red', linestyle='--', linewidth=1.2, label='COVID-19 Emergency\n(March 13, 2020)')

ax.set_xlabel('Date')
ax.set_ylabel('Total User-Stock Positions (millions)')
ax.set_title('Figure 1, Panel A — Total Robinhood User-Stock Positions Over Time')

# I match the paper's sample window
ax.set_xlim(pd.Timestamp('2018-07-01'), pd.Timestamp('2020-07-01'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)

ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'write-up', 'tables', 'Total_User_Position_Overtime.png')
plt.savefig(out, dpi=150)
plt.show()
