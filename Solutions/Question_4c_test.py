import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# ── No-gap panel (closest prior observation) ─────────────────────────────────
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

# ── 3-day gap panel ───────────────────────────────────────────────────────────
daily_gap = pd.read_json(os.path.join(BASE, 'data', 'question2_daily.json'))
daily_gap['Date'] = pd.to_datetime(daily_gap['Date'])

# ── Count events at each cutoff for both panels ───────────────────────────────
cutoffs = np.arange(1.1, 8.6, 0.5)

counts_nogap = []
counts_gap   = []

for c in cutoffs:
    counts_nogap.append(daily_nogap[(daily_nogap['userratio'] >= c) & (daily_nogap['users_close_prev'] >= 100)].shape[0])
    counts_gap.append(daily_gap[(daily_gap['userratio'] >= c) & (daily_gap['users_close_prev'] >= 100)].shape[0])

# ── Side-by-side plot ─────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(cutoffs, counts_nogap, width=0.4, color='green', alpha=0.8)
ax1.set_yscale('log')
ax1.set_xlabel('User Change Ratio Cutoff')
ax1.set_ylabel('Number of Events (log scale)')
ax1.set_title('No 3-Day Gap Restriction')
ax1.set_xticks(cutoffs)
ax1.set_xticklabels([f'{c:.1f}' for c in cutoffs], rotation=45)

ax2.bar(cutoffs, counts_gap, width=0.4, color='steelblue', alpha=0.8)
ax2.set_yscale('log')
ax2.set_xlabel('User Change Ratio Cutoff')
ax2.set_ylabel('Number of Events (log scale)')
ax2.set_title('With 3-Day Gap Restriction')
ax2.set_xticks(cutoffs)
ax2.set_xticklabels([f'{c:.1f}' for c in cutoffs], rotation=45)

fig.suptitle('Panel A: Herding Intensity and Daily User Change Ratio Cutoff', fontsize=13)
plt.tight_layout()
plt.show()
