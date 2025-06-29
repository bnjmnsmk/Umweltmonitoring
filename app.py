import os
import requests
import logging
from datetime import datetime, timezone, timedelta

import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
from statsmodels.tsa.statespace.sarimax import SARIMAX
import psycopg2

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output

# --- Logging ---
logging.basicConfig(level=logging.INFO)

# --- Environment / Config ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "dash_db")
DB_USER = os.getenv("DB_USER", "dashuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "dashpassword")
SENSEBOX_ID = os.getenv("SENSEBOX_ID")
API_URL_FORMAT_BOX = os.getenv("API_URL_FORMAT_BOX")
API_URL_FORMAT_SENSOR = os.getenv("API_URL_FORMAT_SENSOR")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# --- Utility Functions ---
def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

def get_sensor_names_ids():
    url = API_URL_FORMAT_BOX.format(sensebox_id=SENSEBOX_ID, response_format="json")
    response = requests.get(url)
    assert response.status_code == 200, f"Failed fetching sensors: {response.status_code}"
    sensors = response.json().get("sensors", [])
    return {
        sensor["title"].replace(" ", "").replace(".", "_").replace("-", "_"): sensor["_id"]
        for sensor in sensors
    }

def get_data(sensor_id: str, from_date: str, to_date: str):
    url = API_URL_FORMAT_SENSOR.format(sensebox_id=SENSEBOX_ID, sensor_id=sensor_id, fromDate=from_date, toDate=to_date)
    response = requests.get(url)
    assert response.status_code == 200, f"Failed fetching data: {response.status_code}"
    return [{'createdAt': d['createdAt'], 'value': d['value']} for d in response.json()]

def resample_data_one_hour(data):
    df = pd.DataFrame(data)
    df['createdAt'] = pd.to_datetime(df['createdAt'])
    df['value'] = df['value'].astype(float)
    df.set_index('createdAt', inplace=True)
    resampled = df.resample('1h').mean().dropna()
    return [{'createdAt': ts.replace(tzinfo=None).isoformat() + 'Z', 'value': f"{val:.2f}"} for ts, val in resampled.itertuples()]

def fetch_from_db(query):
    df = pd.read_sql(query, engine)
    df['createdat'] = pd.to_datetime(df['createdat'])
    df.set_index('createdat', inplace=True)
    df = df.asfreq('h')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df

def create_forecast(df, forecast_hours=48, history_days=5):
    recent = df[df.index >= (df.index[-1] - timedelta(days=history_days))]
    model = SARIMAX(df['value'], order=(2,1,2), seasonal_order=(1,1,1,24),
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    forecast = results.forecast(steps=forecast_hours)
    forecast.index = pd.date_range(start=df.index[-1] + timedelta(hours=1), periods=forecast_hours, freq='h')

    hist_df = recent.reset_index().rename(columns={'createdat': 'timestamp'})
    hist_df['type'] = 'Historical'
    forecast_df = forecast.to_frame(name='value').reset_index().rename(columns={'index': 'timestamp'})
    forecast_df['type'] = 'Forecast'
    return pd.concat([hist_df, forecast_df], ignore_index=True)

# --- Table Mapping ---
query_map = {
    "Temperatur": "SELECT * FROM public.temperatur ORDER BY createdat;",
    "Luftfeuchtigkeit": "SELECT * FROM public.rel_luftfeuchte ORDER BY createdat;",
    "Beleuchtungsstärke": "SELECT * FROM public.beleuchtungsstärke ORDER BY createdat;",
    "UV-Intensität": "SELECT * FROM public.uv_intensität ORDER BY createdat;",
    "PM1": "SELECT * FROM public.pm1 ORDER BY createdat;",
    "PM2.5": "SELECT * FROM public.pm2_5 ORDER BY createdat;",
    "PM4": "SELECT * FROM public.pm4 ORDER BY createdat;",
    "PM10": "SELECT * FROM public.pm10 ORDER BY createdat;",
}

# --- Initialize Dash App ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Environmental Dashboard"

# --- Load Forecast Data ---
initial_df = fetch_from_db(query_map["Temperatur"])
combined_forecast_df = create_forecast(initial_df)

# --- Layout ---
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("🌿 Environmental Dashboard", className="text-center text-success mb-4"), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("📊 Pure Historical Data")),
                dbc.CardBody([
                    dbc.Label("Select Data Source"),
                    dcc.Dropdown(
                        id='table-dropdown',
                        options=[{'label': k, 'value': k} for k in query_map],
                        value='Temperatur',
                        className="mb-3"
                    ),
                    dbc.Label("Select Interval"),
                    dcc.Dropdown(
                        id='interval-dropdown',
                        options=[
                            {'label': 'Last 1 Day', 'value': 1},
                            {'label': 'Last 2 Days', 'value': 2},
                            {'label': 'Last 1 Week', 'value': 7}
                        ],
                        value=7,
                        clearable=False,
                        className="mb-4"
                    ),
                    dcc.Graph(id='historical-graph')
                ])
            ])
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("🔮 Forecast for Next 48 Hours")),
                dbc.CardBody([
                    dcc.Graph(
                        id='forecast-graph',
                        figure=px.line(
                            combined_forecast_df,
                            x='timestamp', y='value', color='type',
                            labels={'value': 'Temperatur', 'timestamp': 'Time', 'type': 'Data'}
                        ).update_layout(template='plotly_white', hovermode='x unified')
                    )
                ])
            ])
        ], md=6)
    ])
], fluid=True, className="p-4 bg-light")

# --- Callbacks ---
@app.callback(
    Output('historical-graph', 'figure'),
    [Input('table-dropdown', 'value'), Input('interval-dropdown', 'value')]
)
def update_historical_graph(selected_table, days):
    df = fetch_from_db(query_map[selected_table])
    df_range = df[df.index >= (df.index.max() - pd.Timedelta(days=days))].reset_index()
    fig = px.line(
        df_range, x='createdat', y='value',
        title=f'{selected_table} (Last {days} Day{"s" if days > 1 else ""})',
        labels={'value': selected_table, 'createdat': 'Time'}
    )
    fig.update_layout(template='plotly_white', hovermode='x unified')
    return fig

# --- Run ---
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
