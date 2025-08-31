"""
Script: sarimax_forecast_groups.py
Purpose:
    - Forecast housing transactions using SARIMAX by grouping data into:
        * "Budapest és Pest"
        * "Többi város"
    - Aggregates yearly transaction counts by group.
    - Fits SARIMAX models and produces 5-year forecasts.
    - Visualizes historical data, forecast, and confidence intervals.

Author: LLMozes
"""

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# --- 1. Adatok előkészítése ---
dataset = pd.DataFrame(dataset)   # biztosítjuk, hogy DataFrame legyen
dataset['Év'] = pd.to_datetime(dataset['Év'], errors='coerce')

# --- 2. Csoportosítás definíciója ---
def assign_group(row):
    if row['Régió'] in ['Budapest', 'Pest']:
        return 'Budapest és Pest'
    elif row['Településtípus'] == 'városok':
        return 'Többi város'
    else:
        return 'Egyéb'

dataset['Csoport'] = dataset.apply(assign_group, axis=1)

# Csak két fő csoportot tartunk meg
dataset = dataset[dataset['Csoport'].isin(['Budapest és Pest', 'Többi város'])]

# --- 3. Aggregálás: tranzakciók száma évente és csoportonként ---
agg_data = dataset.groupby(['Év', 'Csoport'])['db'].sum().reset_index()

# --- 4. SARIMAX előrejelzés minden csoportra ---
all_plots = []
for group, group_data in agg_data.groupby('Csoport'):
    time_series = group_data.set_index('Év')['db']

    # SARIMAX modell (ARIMA(3,1,3) szezonális komponens nélkül)
    model = SARIMAX(time_series, order=(3, 1, 3), seasonal_order=(0, 0, 0, 0))
    results = model.fit(disp=False)

    # Előrejelzés (5 évre)
    forecast = results.get_forecast(steps=5)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    all_plots.append({
        'group': group,
        'time_series': time_series,
        'forecast_mean': forecast_mean,
        'forecast_ci': forecast_ci
    })

# --- 5. Vizualizáció ---
plt.figure(figsize=(12, 6))

colors = {
    'Budapest és Pest': 'green',
    'Többi város': 'red'
}

for plot_data in all_plots:
    group = plot_data['group']
    time_series = plot_data['time_series']
    forecast_mean = plot_data['forecast_mean']
    forecast_ci = plot_data['forecast_ci']

    # Történeti adatok
    plt.plot(time_series, label=f'Történeti adatok - {group}', color=colors[group])

    # Előrejelzés
    plt.plot(forecast_mean, marker='o', label=f'Előrejelzés - {group}', color=colors[group])

    # Előrejelzési intervallum (confidence interval)
    plt.fill_between(
        forecast_ci.index,
        forecast_ci.iloc[:, 0],
        forecast_ci.iloc[:, 1],
        alpha=0.2, color=colors[group],
        label=f'Előrejelzési intervallum - {group}'
    )

    # Értékek kiírása a grafikonra
    for year, value in forecast_mean.items():
        plt.text(
            year, value, f'{value/1000:.1f}k',
            fontsize=10, ha='center', va='bottom',
            color=colors[group]
        )

plt.ylabel('Tranzakciók száma')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
