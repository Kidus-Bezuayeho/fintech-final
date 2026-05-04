import pandas as pd
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
print(f"  Total observations:               {len(daily)}")
print(f"  Fraction flagged:                 {daily['rh_herd'].mean():.6f}")
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

# I save the full daily panel with rh_herd for Question 2c to use
daily.to_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'question2_daily.json'),
              orient='records', date_format='iso')
