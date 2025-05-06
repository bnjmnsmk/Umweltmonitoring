import requests
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import json
import os
import psycopg2
from datetime import datetime


# Database connection setup
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "dash_db")
DB_USER = os.getenv("DB_USER", "dashuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "dashpassword")

# Sensbox setup
SENSEBOX_ID = os.getenv("SENSEBOX_ID")
API_URL_FORMAT_BOX = os.getenv("API_URL_FORMAT_BOX")

# A connection function for the Backend
def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )    
def create_table(table_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name}(
            id INT PRIMARY KEY,
            name VARCHAR(50))
                   """)
    try:
        conn.commit()
        print(f"Table {table_name} created")
    except Exception as e:
        print(f"Creating tabel failed: {e}")

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

def insert_into_table(table_name,column_values:dict):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
                    INSERT INTO {table_name}
                    ({column_values.keys()})
                    VALUES ({column_values.values()})
                    """)

table_names = get_sensor_names_ids().keys
for table in table_names:
    create_table(table)




app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([
    html.H1("Sensor Dashboard"),
    html.P("Sensor tables initialized.")
])


# Initialize the Dash app
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8050)
    

