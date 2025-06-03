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
    'uv': 'UV-Intensität',  # Optional
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

# Layout: zwei Grafiken (aktuelle Daten + leere Prognose) + Dropdown rechts
app.layout = html.Div(
    style={'display': 'flex', 'flexDirection': 'row', 'height': '100vh'},
    children=[
        # Linke Seite: zwei Grafiken übereinander
        html.Div(
            style={'flex': 1, 'display': 'flex', 'flexDirection': 'column'},
            children=[
                # Grafik für aktuelle Sensordaten
                dcc.Graph(id='main-graph', style={'flex': 1}),
                # Leere Grafik für zukünftige (ML-)Daten
                dcc.Graph(
                    id='forecast-graph',
                    figure={
                        'data': [],
                        'layout': {
                            'title': 'Zukunftsprognose',
                            'xaxis': {'title': 'Zeit'},
                            'yaxis': {'title': 'Wert'},
                            'margin': {'l': 100, 'r': 20, 't': 40, 'b': 40}
                        }
                    },
                    style={'flex': 1}
                )
            ]
        ),
        # Rechte Seite: Dropdown-Menu zur Sensorauswahl
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
        # Interval-Komponente bleibt unverändert
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

        # Ermitteln des zuletzt gemessenen Werts
        latest_value = df['value'].iloc[-1] if not df.empty else None
        latest_time = df['createdAt'].iloc[-1] if not df.empty else None

        # Grafik erstellen (Linie oder Scatter)
        if selected_sensor in ['pm1', 'pm25', 'pm4', 'pm10', 'illuminance']:
            fig = px.scatter(df, x='createdAt', y='value', title=SENSOR_TITLES[selected_sensor])
        else:
            fig = px.line(df, x='createdAt', y='value', title=SENSOR_TITLES[selected_sensor])

        # Layout-Anpassungen
        fig.update_layout(
            xaxis_title='Zeit',
            yaxis_title=SENSOR_TITLES[selected_sensor],
            margin={'l': 100, 'r': 20, 't': 40, 'b': 40},
            title={'x': 0.5, 'xanchor': 'center'}
        )
        # Tick-Größen und Range-Selector Buttons
        fig.update_yaxes(
            title_font_size=18,
            tickfont_size=16,
            automargin=True
        )
        fig.update_xaxes(
            tickfont_size=14,
            rangeselector=dict(
                buttons=[
                    dict(step='all', label='All'),
                    dict(count=1, label='D', step='day', stepmode='backward'),
                    dict(count=7, label='W', step='day', stepmode='backward'),
                    dict(count=1, label='M', step='month', stepmode='backward')
                ]
            ),
            rangeslider=dict(visible=False)
        )

        # Annotation: zuletzt gemessener Wert oben rechts
        if latest_value is not None:
            text = f"Aktuell: {latest_value:.2f}"
            fig.add_annotation(
                xref='paper', yref='paper',
                x=1, y=1,
                xanchor='right', yanchor='top',
                text=text,
                showarrow=False,
                font=dict(size=14)
            )
    else:
        # Fehlermeldung, falls Sensor nicht verfügbar
        fig = px.scatter(title=f"{SENSOR_TITLES.get(selected_sensor, 'Sensor')} nicht verfügbar")
        fig.update_layout(margin={'l': 100, 'r': 20, 't': 40, 'b': 40})

    return fig

# Server starten
if __name__ == '__main__':
    app.run(debug=True)
