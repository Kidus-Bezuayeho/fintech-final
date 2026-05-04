import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# I load the date-level aggregation saved by Question_1.py
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

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figure1_panelA.png')
plt.savefig(out, dpi=150)
print(f"\nFigure saved to {out}\n")
plt.show()
