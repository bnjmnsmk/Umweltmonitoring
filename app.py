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

def create_table(table_name):
    table_name = table_name.lower()
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(f"""DROP TABLE IF EXISTS {table_name}""")
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name}(
                createdat TIMESTAMPTZ PRIMARY KEY,
                value FLOAT)
                    """)
        
        cursor.execute(f"""SELECT create_hypertable('{table_name}','createdat', if_not_exists => TRUE)""")

        conn.commit()
        print(f"Hypertable {table_name} created")

    except Exception as e:
        print(f"Creating tabel failed: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

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

def bulk_insert_into_table(table_name,data:list[dict]):
    conn = get_connection()
    cursor = conn.cursor()

    query = (f"""INSERT INTO {table_name} (createdat, value) VALUES (%s, %s)""")

    values = [(row['createdAt'], float(row['value'])) for row in data]

    try:
        cursor.executemany(query, values)
        conn.commit()
        print(f"{len(values)} rows inserted into {table_name}.")
    except Exception as e:
        print(f"Insert failed: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

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

def update_latest_data():
    logging.info("Fetching and inserting latest data...")
    now = datetime.now(timezone.utc).replace(microsecond=0)
    one_hour_ago = now - timedelta(hours=1)

    # Format timestamps to 'YYYY-MM-DDTHH:MM:SSZ'
    from_str = one_hour_ago.isoformat().replace("+00:00", "Z")
    to_str = now.isoformat().replace("+00:00", "Z")

    sensor_map = get_sensor_names_ids()

    with get_connection() as conn:
        with conn.cursor() as cur:
            for sensor_name, sensor_id in sensor_map.items():
                try:
                    raw_data = get_data(sensor_id, from_str, to_str)
                    resampled = resample_data_one_hour(raw_data)

                    for entry in resampled:
                        cur.execute(
                            f"""
                            INSERT INTO public.{sensor_name.lower()} (createdat, value)
                            VALUES (%s, %s)
                            ON CONFLICT (createdat) DO NOTHING;
                            """,
                            (entry['createdAt'], entry['value'])
                        )
                    logging.info(f"Updated {sensor_name}")
                except Exception as e:
                    logging.warning(f"Failed to update {sensor_name}: {e}")
        conn.commit()


sensor_dict = get_sensor_names_ids()
print(f"--- Get Sensors --- \n")


#Get Time
print(f"--- Get Time ---\n")
now = datetime.now(timezone.utc) 
two_weeeks_ago = now - timedelta(weeks=2) 
iso_now = now.isoformat().replace('+00:00','Z')
iso_two_weeks_ago = two_weeeks_ago.isoformat().replace('+00:00','Z')




#--- Data initalization ---
print("--- Data initalization ---")
for name, id in sensor_dict.items():
    #Create Tables
    create_table(name)
    #Get Data for Tabel
    data = get_data(sensor_id=id,
                from_date=iso_two_weeks_ago,
                to_date=iso_now)
    #Insert Data into table
    data = resample_data_one_hour(data)
    bulk_insert_into_table(table_name=name,
                            data=data)


print("\n--- Get Temperature Data for prediction ---\n")
create_table(table_name='Temperatur')

id = sensor_dict['Temperatur']
data = []
for iteration in range(10):
    print(f"Iteration: {iteration}")
    print(f"From: {iso_two_weeks_ago} To: {iso_now}")
    
    new_data = get_data(sensor_id=id, from_date=iso_two_weeks_ago, to_date=iso_now)
    data.extend(new_data)

    # Extract and parse latest time as datetime object
    last_time_str = new_data[-1]['createdAt'].replace('+00:00', 'Z')
    last_time = datetime.strptime(last_time_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    
    print(f'{last_time.isoformat()}')

    # Update range for next loop
    iso_now = last_time.isoformat().replace('+00:00', 'Z')
    iso_two_weeks_ago = (last_time - timedelta(weeks=2, seconds=10)).isoformat().replace('+00:00', 'Z')
    

data = resample_data_one_hour(data)
print("\n --- Resamle Data ---\n")
print("--- Insert Data into Temperatur ---")
bulk_insert_into_table('Temperatur',data=data)

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
    dbc.Col([
        dbc.Button("🔄 Refresh Data Now", id="refresh-button", color="primary", className="mb-3"),
        dcc.Interval(id="auto-refresh", interval=3600 * 1000, n_intervals=0)  # 1 hour in milliseconds
        ])
    ]),
    dbc.Row([
        dbc.Col(html.H1("🌿 Environmental Dashboard", className="text-center text-success mb-4"), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("📊 Historical Data")),
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
                ],style={"backgroundColor": "rgba(255, 255, 255, 0.9)", "borderRadius": "15px"})
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
            ],style={"backgroundColor": "rgba(255, 255, 255, 0.9)", "borderRadius": "15px"})
        ], md=6)
    ])
    ], fluid=True, className="p-4", style={
    "background": "linear-gradient(to right, #d4fc79, #96e6a1)",
    "minHeight": "100vh"
    })

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

@app.callback(
    Output('table-dropdown', 'options'),  # Dummy output to trigger the refresh
    [Input('refresh-button', 'n_clicks'), Input('auto-refresh', 'n_intervals')],
    prevent_initial_call=True
)
def trigger_data_refresh(n_clicks, n_intervals):
    update_latest_data()
    return [{'label': k, 'value': k} for k in query_map]

# --- Run ---
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
