from flask import Flask, request, render_template
import folium
import openrouteservice
import pandas as pd
import pickle
import numpy as np
from math import radians, cos, sin, asin, sqrt

app = Flask(__name__)

# Load the saved model
with open("taxi_model.pkl", "rb") as file:
  model = pickle.load(file) # or use pickle.load if you used pickle

def plot_map(start_coords, end_coords, api_key):
    client = openrouteservice.Client(key=api_key)
    route = client.directions(
        coordinates=[start_coords[::-1], end_coords[::-1]],
        profile='driving-car',
        format='geojson'
    )
    m = folium.Map(location=[(start_coords[0] + end_coords[0]) / 2, (start_coords[1] + end_coords[1]) / 2], zoom_start=12)
    folium.Marker(start_coords, popup='Start').add_to(m)
    folium.Marker(end_coords, popup='End').add_to(m)
    folium.GeoJson(route, name='route').add_to(m)
    return m
def distance(lat1, lat2, lon1, lon2):
     
    # The math module contains a function named
    # radians which converts from degrees to radians.
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
      
    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
 
    c = 2 * asin(sqrt(a)) 
    
    # Radius of earth in kilometers. Use 3956 for miles
    r = 6371
      
    # calculate the result
    return(c * r)
vectorized_distance = np.vectorize(distance)
def preprocess_input(data):
    df_sampled = pd.DataFrame([data])
    df_sampled['PTP_distance'] = vectorized_distance(lon1 = df_sampled['pickup_longitude'] , lat1 = df_sampled['pickup_latitude'] , lon2= df_sampled['dropoff_longitude'],lat2 = df_sampled['dropoff_latitude'])
    df_sampled['pickup_datetime'] = pd.to_datetime(df_sampled['pickup_datetime'], utc=True)
    df_sampled['pickup_year'] = df_sampled['pickup_datetime'].dt.year
    df_sampled['pickup_month'] = df_sampled['pickup_datetime'].dt.month
    df_sampled['pickup_day'] = df_sampled['pickup_datetime'].dt.day
    df_sampled['pickup_hour'] = df_sampled['pickup_datetime'].dt.hour
    df_sampled['pickup_minute'] = df_sampled['pickup_datetime'].dt.minute
    df_sampled['pickup_second'] = df_sampled['pickup_datetime'].dt.second
    df_sampled['pickup_dayofweek'] = df_sampled['pickup_datetime'].dt.dayofweek
    df_sampled['pickup_dayofyear'] = df_sampled['pickup_datetime'].dt.dayofyear
    df_sampled['pickup_weekofyear'] = df_sampled['pickup_datetime'].dt.isocalendar().week
    df_sampled['pickup_quarter'] = df_sampled['pickup_datetime'].dt.quarter
    X = df_sampled.drop(columns=['pickup_datetime'])
    return X

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/route', methods=['POST'])
def route():
    pickup_datetime = request.form['pickup_datetime']
    pickup_longitude = float(request.form['pickup_longitude'])
    pickup_latitude = float(request.form['pickup_latitude'])
    dropoff_longitude = float(request.form['dropoff_longitude'])
    dropoff_latitude = float(request.form['dropoff_latitude'])
    passenger_count = int(request.form['passenger_count'])

        # Prepare the input data
    input_data = {
        'pickup_datetime': pickup_datetime,
        'pickup_longitude': pickup_longitude,
        'pickup_latitude': pickup_latitude,
           'dropoff_longitude': dropoff_longitude,
        'dropoff_latitude': dropoff_latitude,
        'passenger_count': passenger_count
        }

        # Preprocess the input data
    X = preprocess_input(input_data)

        # Make prediction
    prediction = model.predict(X)

        # Extract the coordinates
    pickup_coords = (pickup_latitude, pickup_longitude)
    dropoff_coords = (dropoff_latitude, dropoff_longitude)

    api_key = '5b3ce3597851110001cf6248779f7a2474034ca39bae302033640277'  # Replace with your actual OpenRouteService API key
    map_ = plot_map(pickup_coords, dropoff_coords, api_key)
    map_html = map_._repr_html_()

    return render_template('map.html', map_html=map_html, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
