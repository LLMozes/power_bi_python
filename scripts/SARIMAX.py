"""
Script: sarimax_forecast_housing_loans.py
Purpose:
    - Forecast housing loans (count & total value) using SARIMAX model.
    - Aggregates data for loans without government support.
    - Produces forecasts for both:
        * Number of loans (db)
        * Total loan value in billion HUF (BillióFt)
    - Visualization combines bar chart (db) and line chart (BillióFt) with forecast.

Author: LLMozes
"""

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# --- 1. Adatok előkészítése ---
dataset = dataset.drop_duplicates()

# Dátum konverzió
try:
    dataset['Év'] = pd.to_datetime(dataset['Év'], errors='coerce')
except Exception as e:
    print(f"Hiba a dátumok konvertálásakor: {e}")

# Csak támogatás nélküli hitelek
dataset = dataset[dataset['Hitel finanszírozása'] == 'támogatás nélküli']

# Billió Ft oszlop (milliárd HUF → billió HUF)
dataset['BillióFt'] = dataset['Ft'] / 1e9

# Aggregálás év szintre
agg_data = dataset.groupby('Év').agg({
    'db': 'sum',
    'BillióFt': 'sum'
}).reset_index()

# --- 2. Idősor modellezés: db ---
time_series_db = agg_data.set_index('Év')['db']
model_db = SARIMAX(time_series_db, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12))
results_db = model_db.fit(disp=False)
forecast_db = results_db.get_forecast(steps=5)
forecast_db_mean = forecast_db.predicted_mean

# --- 3. Idősor modellezés: BillióFt ---
time_series_billion = agg_data.set_index('Év')['BillióFt']
model_billion = SARIMAX(time_series_billion, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
results_billion = model_billion.fit(disp=False)
forecast_billion = results_billion.get_forecast(steps=5)
forecast_billion_mean = forecast_billion.predicted_mean
forecast_billion_ci = forecast_billion.conf_int()

# --- 4. Vizualizáció ---
fig, ax1 = plt.subplots(figsize=(14, 8))
ax2 = ax1.twinx()

# (a) db – történeti és előrejelzés (bar chart)
ax1.bar(
    agg_data['Év'], agg_data['db'],
    label='db - Történeti', alpha=0.8, width=100
)
ax1.bar(
    forecast_db_mean.index, forecast_db_mean,
    label='db - Előrejelzés', alpha=0.7,
    width=100, color='darkblue'
)

# Annotációk a db előrejelzéshez
for x, y in zip(forecast_db_mean.index, forecast_db_mean):
    ax1.annotate(
        f"{y/1000:.1f}k",
        (x, y / 2),
        textcoords="offset points", xytext=(3, 0),
        ha='center', color='black'
    )

# (b) BillióFt – történeti és előrejelzés (line chart)
ax2.plot(
    agg_data['Év'], agg_data['BillióFt'],
    marker='o', color='red', label='BillióFt - Történeti'
)
ax2.plot(
    forecast_billion_mean.index, forecast_billion_mean,
    linestyle='--', color='green', label='BillióFt - Előrejelzés'
)
ax2.fill_between(
    forecast_billion_mean.index,
    forecast_billion_ci.iloc[:, 0],
    forecast_billion_ci.iloc[:, 1],
    color='darkgreen', alpha=0.2
)

# Annotációk a BillióFt előrejelzéshez
for x, y in zip(forecast_billion_mean.index, forecast_billion_mean):
    xytext_value = (0, 10) if x.year == 2025 else (0, 25)
    ax2.annotate(
        f"{y:.2f}",
        (x, y),
        textcoords="offset points", xytext=xytext_value,
        ha='center', color='green'
    )

# Utolsó történeti érték annotálása
last_historical_date = agg_data['Év'].iloc[-1]
last_historical_value = agg_data['BillióFt'].iloc[-1]
ax2.annotate(
    f"{last_historical_value:.2f}",
    (last_historical_date, last_historical_value),
    textcoords="offset points", xytext=(0, 10),
    ha='center', color='red'
)

# --- 5. Tengelyek és címkék ---
max_billion = agg_data['BillióFt'].max()
ax2.set_ylim(0, max_billion * 1.2)

ax1.set_xlabel('Év')
ax1.set_ylabel('db (államilag lakáscélú hitelek)', color='blue')
ax2.set_ylabel('BillióFt', color='green')
ax1.set_ylim(0, ax1.get_ylim()[1] * 1.2)

# Grid + tickek
ax1.set_yticks(range(0, int(ax1.get_ylim()[1]), 50000))
ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

# Legend
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()
