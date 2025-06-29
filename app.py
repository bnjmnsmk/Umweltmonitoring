import requests
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import json
import os
import psycopg2
from datetime import datetime
from datetime import datetime, timezone, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sqlalchemy import create_engine
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output
import plotly.express as px
import pandas as pd


# Database connection setup
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "dash_db")
DB_USER = os.getenv("DB_USER", "dashuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "dashpassword")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# Sensbox setup
SENSEBOX_ID = os.getenv("SENSEBOX_ID")
API_URL_FORMAT_BOX = os.getenv("API_URL_FORMAT_BOX")
API_URL_FORMAT_SENSOR = os.getenv("API_URL_FORMAT_SENSOR")

# A connection function for the Backend
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
    url = API_URL_FORMAT_BOX.format(sensebox_id=SENSEBOX_ID,response_format="json")
    status_code = requests.get(url).status_code
    assert status_code , f"Failed fetching data from api {status_code}"
    
    sensors = requests.get(url).json().get("sensors")
    sensor_name_id = {}
    for sensor in sensors:
        name = sensor.get('title').replace(" ","").replace(".","_").replace("-","_")
        _id = sensor.get('_id')
        sensor_name_id.update({name : _id})
    return sensor_name_id


def get_data(sensor_id:str,fromDate,toDate):

    #get URL with sensor_id and dates
    url = API_URL_FORMAT_SENSOR.format(sensebox_id=SENSEBOX_ID, sensor_id=sensor_id, fromDate=fromDate, toDate=toDate)
    
    #Make sure we get correct response
    status_code = requests.get(url).status_code
    assert status_code == 200, f"Failed fetching data from api {status_code}"
    
    #Get the json file of the sensor
    data = requests.get(url).json()
    data = [{'createdAt': item['createdAt'], 'value': item['value']} for item in data]
    return data


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
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df['createdAt'] = pd.to_datetime(df['createdAt'])
    df['value'] = df['value'].astype(float)

    # Set datetime index
    df.set_index('createdAt', inplace=True)

    # Resample to 1-hour intervals using mean
    resampled_df = df.resample('1h').mean().dropna()

    # Convert back to list[dict]
    resampled_data = [
    {'createdAt': ts.replace(tzinfo=None).isoformat() + 'Z', 'value': f"{val:.2f}"}
    for ts, val in resampled_df.itertuples()
    ]
    return resampled_data


if __name__ == "__main__":
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
                    fromDate=iso_two_weeks_ago,
                    toDate=iso_now)
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
        
        new_data = get_data(sensor_id=id, fromDate=iso_two_weeks_ago, toDate=iso_now)
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


    conn = get_connection()
    print(f"\n --- Get Connection ---\n")


    # Use a Bootstrap theme
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
    app.title = "Temperature Dashboard"
    print("--- Initialize Dash app Name and Title --- \n")


    # ---- Table Query Mapping ----
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
    print("--- Table Query Mapping --- \n")


    # ---- Load initial data for forecast (e.g., Temperatur) ----

    initial_query = query_map["Temperatur"]
    df = pd.read_sql(initial_query, engine)
    df['createdat'] = pd.to_datetime(df['createdat'])
    df.set_index('createdat', inplace=True)
    df = df.asfreq('h')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    print("--- Load initial data for forecast (e.g., Temperatur) --- \n")


    # Forecast last 5 days + next 2 days
    history_days = 5
    forecast_hours = 48

    start_date = df.index[-1] - pd.Timedelta(days=history_days)
    historical_recent = df.loc[start_date:]

    model = SARIMAX(df['value'],
                    order=(2, 1, 2),
                    seasonal_order=(1, 1, 1, 24),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=forecast_hours)

    future_index = pd.date_range(start=df.index[-1] + pd.Timedelta(hours=1),
                                periods=forecast_hours, freq='h')
    forecast.index = future_index
    print("--- Forecast last 5 days + next 2 days --- \n")

    # Combine for forecast graph
    hist_df = historical_recent.reset_index().rename(columns={'createdat': 'timestamp'})
    hist_df['type'] = 'Historical'

    forecast_df = forecast.to_frame(name='value').reset_index()
    forecast_df.rename(columns={'index': 'timestamp'}, inplace=True)
    forecast_df['type'] = 'Forecast'

    combined_forecast_df = pd.concat([hist_df, forecast_df], ignore_index=True)
    print("--- Combine for forecast graph --- \n")

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
                            options=[{'label': name, 'value': name} for name in query_map],
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


    print("--- Layout made --- \n")

    # --- Callback ---
    @app.callback(
        Output('historical-graph', 'figure'),
        [Input('table-dropdown', 'value'),
        Input('interval-dropdown', 'value')]
    )
    def update_historical_graph(selected_table, days):
        query = query_map[selected_table]
        df = pd.read_sql(query, engine)  

        df['createdat'] = pd.to_datetime(df['createdat'])
        df.set_index('createdat', inplace=True)
        df = df.asfreq('h')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        max_date = df.index.max()
        min_date = max_date - pd.Timedelta(days=days)
        hist_df = df.loc[min_date:max_date].reset_index()

        fig = px.line(hist_df, x='createdat', y='value',
                    title=f'{selected_table} (Last {days} Day{"s" if days > 1 else ""})',
                    labels={'value': selected_table, 'createdat': 'Time'})
        fig.update_layout(template='plotly_white', hovermode='x unified')
        return fig
    print("--- Callback made --- \n")

    app.run(debug=True, host="0.0.0.0", port=8050)