"""
Script: prophet_forecast_regional.py
Purpose:
    - Use Facebook Prophet to forecast average housing-related values by region.
    - Historical data is aggregated by year & region.
    - Logistic growth model is applied with yearly seasonality.
    - Produces forecasts for the next 4 years.
    - Results are visualized with scatter plots for historical vs forecasted values.

Author: LLMozes
"""

from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- 1. Év konverzió és tisztítás ---
dataset['Év'] = pd.to_datetime(dataset['Év'], errors='coerce')
dataset = dataset.dropna(subset=['Év'])
dataset['Év'] = dataset['Év'].dt.year

# --- 2. Átlagolt értékek régiónként és évenként ---
agg_data = dataset.groupby(['Év', 'Régió'])['Ft'].mean().reset_index()

# --- 3. Prophet előrejelzés minden régióra ---
forecasts = []
unique_regions = agg_data['Régió'].unique()

for region in unique_regions:
    region_data = agg_data[agg_data['Régió'] == region]
    prophet_data = region_data.rename(columns={'Év': 'ds', 'Ft': 'y'})
    prophet_data['ds'] = pd.to_datetime(prophet_data['ds'], format='%Y')

    # kapacitás (logistic growth miatt kell)
    prophet_data['cap'] = prophet_data['y'].max()

    # Modell definiálása
    model = Prophet(
        yearly_seasonality=True,
        growth='logistic',
        changepoint_prior_scale=0.02,
        seasonality_prior_scale=1.0
    )

    # Modell illesztés
    model.fit(prophet_data)

    # Jövőbeli időpontok (4 év előre)
    future = model.make_future_dataframe(periods=4, freq='Y')
    future['cap'] = prophet_data['cap'].iloc[0]

    # Előrejelzés
    forecast = model.predict(future)

    # Csak az új (jövőbeli) éveket hagyjuk meg
    existing_years = prophet_data['ds'].dt.year.unique()
    forecast = forecast[~forecast['ds'].dt.year.isin(existing_years)]

    forecasts.append({
        'region': region,
        'historical': prophet_data,
        'forecast': forecast
    })

# --- 4. Ábrázolás ---
plt.figure(figsize=(14, 10))

# Scatter plot történeti + előrejelzés
for f in forecasts:
    plt.scatter(
        f['historical']['ds'].dt.year, f['historical']['y'],
        label=f"{f['region']} - Történeti", alpha=0.7
    )
    plt.scatter(
        f['forecast']['ds'].dt.year, f['forecast']['yhat'],
        label=f"{f['region']} - Előrejelzés", alpha=0.7, marker='x'
    )

# --- 5. Előrejelzési számok szöveges kiírása ---
x_text_pos = 2028
y_text_pos = dataset['Ft'].max() * 0.9
line_height = dataset['Ft'].max() * 0.2

for f in forecasts:
    forecast_text = f"{f['region']}:\n" + "\n".join(
        [f"{int(row['ds'].year)}: {row['yhat']/1000:.1f}k Ft"
         for _, row in f['forecast'].iterrows()]
    )
    plt.text(
        x_text_pos, y_text_pos, forecast_text,
        fontsize=10, verticalalignment='top', horizontalalignment='left'
    )
    y_text_pos -= line_height

# --- 6. Tengelyek, címkék, formázás ---
ax = plt.gca()
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x/1000:.1f}k")
)

plt.xlabel('Év', fontsize=14)
plt.ylabel('Átlagolt Ft', fontsize=14)
plt.grid(True)
plt.ylim(0, dataset['Ft'].max() * 1.6)

plt.legend(
    loc='upper left', bbox_to_anchor=(1, 1),
    ncol=1, fontsize=10
)
plt.tight_layout()
plt.show()
