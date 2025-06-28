import requests
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import json
import os
import psycopg2
from datetime import datetime
from datetime import datetime, timezone, timedelta

# Database connection setup
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "dash_db")
DB_USER = os.getenv("DB_USER", "dashuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "dashpassword")

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

# def insert_into_table(table_name,column_values:dict):
    
#     conn = get_connection()
#     cursor = conn.cursor()

#     query = (f"""INSERT INTO {table_name} (createdAt, value) VALUES (%s, %s)""")

#     values = [(row['createdAt'], float(row['value'])) for row in data]

#     try:
#         cursor.executemany(query, values)
#         conn.commit()
#         print(f"{len(values)} rows inserted into {table_name}.")
#     except Exception as e:
#         print(f"Insert failed: {e}")
#     finally:
#         cursor.close()
#         conn.close()





app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([
    html.H1("Sensor Dashboard"),
    html.P("Sensor tables initialized.")
])


# Initialize the Dash app
if __name__ == '__main__':
    sensor_dict = get_sensor_names_ids()
    #Get Time
    now = datetime.now(timezone.utc) 
    two_weeeks_ago = now - timedelta(weeks=2) 
    iso_now = now.isoformat().replace('+00:00','Z')
    iso_two_weeks_ago = two_weeeks_ago.isoformat().replace('+00:00','Z')

        #Data initalization
    for name, id in sensor_dict.items():
        #Create Tables
        create_table(name)

        #Get Data for Tabel
        data = get_data(sensor_id=id,
                    fromDate=iso_two_weeks_ago,
                    toDate=iso_now)
        #Insert Data into table
        bulk_insert_into_table(table_name=name,
                                data=data)

    create_table(table_name='Temperatur')

    id = sensor_dict['Temperatur']
    data = []
    for iteration in range(5):
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
        print(f"*UPDATED* From: {iso_two_weeks_ago} To: {iso_now}")
        
    bulk_insert_into_table('Temperatur',data=data)


    app.run(debug=True, host="0.0.0.0", port=8050)
    

