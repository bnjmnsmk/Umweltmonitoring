import requests
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import json
from datetime import datetime

senseBoxId = "6793f4e4c326f20007c34dd2"
data = requests.get(url=(f"https://api.opensensemap.org/boxes/{senseBoxId}?format:json"))
content = json.loads(data.content)
sensors = pd.json_normalize(content['sensors'])



datenow = datetime.now().isoformat() + "Z"
dateweekb4 = "2025-01-01T00:00:00Z"
data = requests.get(url=f"https://api.opensensemap.org/boxes/{senseBoxId}/data/6793f4e4c326f20007c34dd3?from-date={dateweekb4}&to-date={datenow}&download=false&format=json")
json.loads(data.content)
df2 = pd.DataFrame(json.loads(data.content))[["createdAt","value"]]
df2["value"] = df2["value"].astype(float)
fig = px.line(df2,x=df2["createdAt"],y=df2["value"])


app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = (dash_table.DataTable(data=sensors.to_dict('records')),
            html.H1(children="""Temperature in Brazzzil"""),
            dcc.Graph(figure=fig)
            )


if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0", port=8050)
    

