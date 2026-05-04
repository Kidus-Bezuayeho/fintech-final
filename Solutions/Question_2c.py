import pandas as pd
import os

# I load the daily panel with rh_herd already computed by Question_2_a_b.py
JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'question2_daily.json')
daily = pd.read_json(JSON)
daily['Date'] = pd.to_datetime(daily['Date'])

herd = daily[daily['rh_herd'] == 1].copy()

# I load prices downloaded once by download_prices.py
prices = pd.read_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'prices.json'))
prices.index = pd.to_datetime(prices.index)
returns = prices.pct_change()

# I match each herding event to its return and the SPY return on that day
herd = herd.set_index('Date')
herd['returns'] = [
    returns.loc[d, t] if d in returns.index and t in returns.columns else float('nan')
    for d, t in zip(herd.index, herd['Ticker'])
]
herd['spy_ret'] = returns['SPY'].reindex(herd.index).values

# I compute abnormal return as the stock's daily return minus SPY's return on the same day
# This removes the market-wide movement, leaving only the stock-specific return
herd['abnormal_returns'] = herd['returns'] - herd['spy_ret']
herd = herd.reset_index()

# (c) Summary statistics for the herding event subsample
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
    'users_close':     herd['users_close'],
    'userchg':         herd['userchg'],
    'userratio':       herd['userratio'],
    'returns':         herd['returns'],
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
