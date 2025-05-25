import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import requests
import pandas as pd
import plotly.express as px

# senseBox ID
SENSEBOX_ID = "6793f4e4c326f20007c34dd2"

# Verfügbare Sensoren und ihre Titel
SENSOR_TITLES = {
    'temperature': 'Temperatur',
    'humidity': 'rel. Luftfeuchte',
    'illuminance': 'Beleuchtungsstärke',
    'uv': 'UV-Intensität', # Können wir auch weglassen, Sensor gibt nur 0 aus
    'pm1': 'PM1',
    'pm25': 'PM2.5',
    'pm4': 'PM4',
    'pm10': 'PM10'
}

# Funktion: Sensor-IDs abrufen
def get_sensor_ids(box_id):
    url = f"https://api.opensensemap.org/boxes/{box_id}"
    response = requests.get(url)
    data = response.json()
    sensor_ids = {}
    for sensor in data.get('sensors', []):
        title = sensor.get('title', '').lower()
        for key, expected in SENSOR_TITLES.items():
            if title == expected.lower():
                sensor_ids[key] = sensor.get('_id')
    return sensor_ids

# Funktion: Messdaten abrufen
def get_sensor_data(sensor_id):
    url = f"https://api.opensensemap.org/boxes/{SENSEBOX_ID}/data/{sensor_id}?format=json"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data)
    df['createdAt'] = pd.to_datetime(df['createdAt'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df

# Sensor-IDs initial laden
sensor_ids = get_sensor_ids(SENSEBOX_ID)

# Dash-App initialisieren
app = dash.Dash(__name__)
app.title = "senseBox Dashboard"

# Layout: Grafik + Dropdown rechts
app.layout = html.Div(
    style={'display': 'flex', 'flexDirection': 'row', 'height': '100vh'},
    children=[
        dcc.Graph(id='main-graph', style={'flex': 1}),
        html.Div(
            style={'width': '220px', 'padding': '10px'},
            children=[
                html.Label("Sensor auswählen:"),
                dcc.Dropdown(
                    id='sensor-dropdown',
                    options=[{'label': title, 'value': key} for key, title in SENSOR_TITLES.items()],
                    value='temperature',
                    clearable=False,
                    style={'font-size': '16px'}
                )
            ]
        ),
        dcc.Interval(
            id='interval-component',
            interval=5*60*1000,  # alle 5 Minuten
            n_intervals=0
        )
    ]
)

# Callback: Grafik basierend auf Auswahl + Intervall aktualisieren
@app.callback(
    Output('main-graph', 'figure'),
    Input('sensor-dropdown', 'value'),
    Input('interval-component', 'n_intervals')
)
def update_graph(selected_sensor, n):
    sensor_id = sensor_ids.get(selected_sensor)
    if sensor_id:
        df = get_sensor_data(sensor_id)
        # Für PM-Werte und Beleuchtungsstärke Scatter statt Liniendiagramm
        if selected_sensor in ['pm1', 'pm25', 'pm4', 'pm10', 'illuminance']:
            fig = px.scatter(df, x='createdAt', y='value', title=SENSOR_TITLES[selected_sensor])
        else:
            fig = px.line(df, x='createdAt', y='value', title=SENSOR_TITLES[selected_sensor])

        fig.update_layout(
            xaxis_title='Zeit',
            yaxis_title=SENSOR_TITLES[selected_sensor],
            margin={'l': 100, 'r': 20, 't': 40, 'b': 40},
            title={'x':0.5, 'xanchor': 'center'}
        )
        fig.update_yaxes(
            title_font_size=18,
            tickfont_size=16,
            automargin=True
        )
        fig.update_xaxes(
            tickfont_size=14
        )
    else:
        # Fehlermeldung als Scatter, falls Sensor nicht verfügbar
        fig = px.scatter(title=f"{SENSOR_TITLES.get(selected_sensor, 'Sensor')} nicht verfügbar")
        fig.update_layout(margin={'l': 100, 'r': 20, 't': 40, 'b': 40})
    return fig

# Server starten
if __name__ == '__main__':
    app.run(debug=True)