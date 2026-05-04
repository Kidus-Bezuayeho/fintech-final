import pandas as pd
import matplotlib.pyplot as plt
import os


DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Robintrack_subsample.csv')
df = pd.read_csv(DATA, parse_dates=['DateTime', 'Date'])
df = df.sort_values(['Ticker', 'DateTime'])

# I keep only the last observation in the 2-4 PM ET window, same as Question 1
mins_since_open = (df['DateTime'].dt.hour - 9) * 60 + df['DateTime'].dt.minute - 30
df_win = df[(mins_since_open >= 270) & (mins_since_open <= 390)].copy()
daily_all = df_win.groupby(['Ticker', 'Date'], as_index=False).agg(users_close=('Users', 'last'))
daily_all = daily_all.sort_values(['Ticker', 'Date'])

# I compute userchg as the change from the closest prior observation, no gap limit
daily_all['userchg'] = daily_all.groupby('Ticker')['users_close'].diff()
daily = daily_all.dropna(subset=['userchg']).copy()

# I get all unique dates in the sample, sorted chronologically
all_dates = sorted(daily['Date'].unique())

# I loop over each date to compute the top-10 buying concentration
# and collect each rank's % share of total positive buying for the bar chart
results = []
rank_rows = []

for date in all_dates:
    day = daily[daily['Date'] == date]

    positive = day[day['userchg'] > 0]

    top10 = positive.nlargest(10, 'userchg').reset_index(drop=True)

    total_pos = positive['userchg'].sum()
    top10_sum = top10['userchg'].sum()

    # I record each rank's % share of total positive buying for part (c)
    if total_pos > 0:
        for rank_idx in range(len(top10)):
            rank_num = rank_idx + 1
            pct = top10.loc[rank_idx, 'userchg'] / total_pos * 100
            rank_rows.append({'rank': rank_num, 'pct': pct})

    if total_pos > 0:
        concentration = top10_sum / total_pos
    else:
        concentration = float('nan')

    # I find stocks with negative user change and pick the bottom 10 (most negative)
    negative = day[day['userchg'] < 0]
    bottom10 = negative.nsmallest(10, 'userchg')

    # I use absolute values so the concentration is a positive fraction
    total_neg = negative['userchg'].abs().sum()
    bottom10_sum = bottom10['userchg'].abs().sum()

    if total_neg > 0:
        bot_concentration = bottom10_sum / total_neg
    else:
        bot_concentration = float('nan')

    results.append({
        'Date':                date,
        'top10_concentration':  concentration,
        'n_positive':           len(positive),
        'top10_sum':            top10_sum,
        'total_pos':            total_pos,
        'bot10_concentration':  bot_concentration,
        'n_negative':           len(negative),
        'bottom10_sum':         bottom10_sum,
        'total_neg':            total_neg,
    })

conc = pd.DataFrame(results)

# (a) Summary statistics for top-10 buying concentration
def stats(s):
    s = s.dropna()
    return pd.Series({
        'N':    s.count(),
        'Mean': s.mean(),
        'SD':   s.std(),
        'Min':  s.min(),
        'p25':  s.quantile(0.25),
        'p50':  s.quantile(0.50),
        'p75':  s.quantile(0.75),
        'Max':  s.max(),
    })

print()
print("=" * 60)
print("  (a) Top-10 buying concentration (Top10_buy)")
print("=" * 60)
print(stats(conc['top10_concentration']).round(4).to_string())
print("=" * 60)
print()

print()
print("=" * 60)
print("  (b) Bottom-10 selling concentration (Top10_sell)")
print("=" * 60)
print(stats(conc['bot10_concentration']).round(4).to_string())
print("=" * 60)
print()

# (c) Bar chart — mean daily % of total net buying by rank, with 95% CIs
rank_df = pd.DataFrame(rank_rows)
grouped = rank_df.groupby('rank')['pct']
means = grouped.mean()
sems  = grouped.std() / grouped.count() ** 0.5
ci95  = 1.96 * sems

ranks = list(range(1, 11))
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(ranks, means[ranks], color='green', alpha=0.8, width=0.6)
ax.errorbar(ranks, means[ranks], yerr=ci95[ranks], fmt='none', color='black', capsize=4)
ax.set_xlabel('Rank of Net Buying')
ax.set_ylabel('Percent of Net Buying')
ax.set_title('Panel A: Concentration of Buying')
ax.set_xticks(ranks)
plt.tight_layout()

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'write-up', 'tables', 'concentration_buying.png')
plt.savefig(out, dpi=150)
plt.show()
