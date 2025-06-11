# project.py


import pandas as pd
import numpy as np
from pathlib import Path

###
from collections import deque
from shapely.geometry import Point
###

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
pd.options.plotting.backend = 'plotly'

import geopandas as gpd

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def create_detailed_schedule(schedule, stops, trips, bus_lines):
    merged = schedule.merge(trips[['trip_id', 'route_id','service_id','direction_name']], on='trip_id')
    
    merged = merged.merge(stops, on='stop_id')
    
    merged = merged[merged['route_id'].isin(bus_lines)]
    
    merged = merged.sort_values(by=['trip_id', 'stop_sequence'])
    
    stop_counts = merged.groupby('trip_id')['stop_id'].count().reset_index()
    stop_counts.rename(columns={'stop_id': 'num_stops'}, inplace=True)
    merged = merged.merge(stop_counts, on='trip_id')

    bus_line_order = {line: i for i, line in enumerate(bus_lines)}
    


    merged['route_order'] = merged['route_id'].map(bus_line_order)
    merged = merged.sort_values(by=['route_order', 'route_id', 'num_stops', 'trip_id', 'stop_sequence'])

    detailed_schedule = merged.set_index('trip_id')

    columns_to_keep = [
    'stop_id', 'stop_sequence', 'shape_dist_traveled',
    'stop_name', 'stop_lat', 'stop_lon',
    'route_id', 'service_id', 'direction_name'
    ]
    detailed_schedule = detailed_schedule[columns_to_keep]
    
    return detailed_schedule


def visualize_bus_network(bus_df):
    # Load the shapefile for San Diego city boundary
    san_diego_boundary_path = 'data/data_city/data_city.shp'
    san_diego_city_bounds = gpd.read_file(san_diego_boundary_path)
    
    # Ensure the coordinate reference system is correct
    san_diego_city_bounds = san_diego_city_bounds.to_crs("EPSG:4326")
    
    san_diego_city_bounds['lon'] = san_diego_city_bounds.geometry.apply(lambda x: x.centroid.x)
    san_diego_city_bounds['lat'] = san_diego_city_bounds.geometry.apply(lambda x: x.centroid.y)
    
    fig = go.Figure()
    
    # Add city boundary
    fig.add_trace(go.Choroplethmapbox(
        geojson=san_diego_city_bounds.__geo_interface__,
        locations=san_diego_city_bounds.index,
        z=[1] * len(san_diego_city_bounds),
        colorscale="Greys",
        showscale=False,
        marker_opacity=0.5,
        marker_line_width=1,
    ))

    # Update layout
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center={"lat": san_diego_city_bounds['lat'].mean(), "lon": san_diego_city_bounds['lon'].mean()},
            zoom=10,
        ),
        margin={"r":0,"t":0,"l":0,"b":0}
    )


    fig = go.Figure()

    fig.add_trace(go.Choroplethmapbox(
        geojson=san_diego_city_bounds.__geo_interface__,
        locations=san_diego_city_bounds.index,
        z=[1] * len(san_diego_city_bounds),
        colorscale="Greys",
        showscale=False,
        marker_opacity=0.5,
        marker_line_width=1,
    ))

    bus_lines = bus_df['route_id'].unique()
    colors = px.colors.qualitative.Plotly[:len(bus_lines)]
    color_dict = {route: color for route, color in zip(bus_lines, colors)}

    for route in bus_lines:
        df = bus_df[bus_df['route_id'] == route]
        fig.add_trace(go.Scattermapbox(
            lat=df['stop_lat'],
            lon=df['stop_lon'],
            mode='markers+lines',
            marker=dict(size=6, color=color_dict[route]),
            line=dict(color=color_dict[route]),
            text=df['stop_name'],
            name=f'Bus Line {route}',
            hoverinfo='text'
        ))

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center={"lat": san_diego_city_bounds['lat'].mean(), "lon": san_diego_city_bounds['lon'].mean()},
            zoom=10,
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend=True
    )

    return fig

# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def find_neighbors(station_name, detailed_schedule):
    relevent_df = detailed_schedule.reset_index()
    index_needed = relevent_df[relevent_df['stop_name'] == station_name].index.to_list()

    safe_indices = []

    for i in index_needed:
        current_row = relevent_df.loc[i]

        if current_row['stop_sequence'] == 1:
            if i + 1 < len(relevent_df):
                safe_indices.append(i + 1)
        else:
            if i + 1 < len(relevent_df) and relevent_df.loc[i + 1]['stop_sequence'] == 1:
                if i - 1 >= 0:
                    safe_indices.append(i - 1)
            else:
                if i - 1 >= 0:
                    safe_indices.append(i - 1)
                if i + 1 < len(relevent_df):
                    safe_indices.append(i + 1)

    cleaned = relevent_df.iloc[safe_indices].sort_index()
    
    return np.array(cleaned['stop_name'])


def bfs(start_station, end_station, detailed_schedule):
    if start_station not in detailed_schedule['stop_name'].values:
        return f"Start station {start_station} not found."
    if end_station not in detailed_schedule['stop_name'].values:
        return f"End station '{end_station}' not found."

    queue = [[start_station]]
    visited = set()

    while queue:
        path = queue.pop(0)
        current_stop = path[-1]

        if current_stop == end_station:
            output_rows = []
            for i, stop in enumerate(path, start=1):
                stop_info = detailed_schedule[detailed_schedule['stop_name'] == stop].iloc[0]
                output_rows.append({
                    'stop_name': stop,
                    'stop_lat': stop_info['stop_lat'],
                    'stop_lon': stop_info['stop_lon'],
                    'stop_num': i
                })
            return pd.DataFrame(output_rows)

        if current_stop not in visited:
            visited.add(current_stop)
            neighbors = find_neighbors(current_stop, detailed_schedule)
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(path + [neighbor])
    
    return "No path found"



# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def simulate_bus_arrivals(tau, seed=12):
    
    np.random.seed(seed) # Random seed -- do not change
    

    start_min = 360
    end_min = 1440
    total_range = end_min - start_min

    num_buses = int(total_range // tau)

    arrival_minutes = np.sort(np.random.uniform(start_min, end_min, num_buses))

    def minutes_to_time_str(minutes):
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        secs = int((minutes - int(minutes)) * 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"

    arrival_times_str = [minutes_to_time_str(mins) for mins in arrival_minutes]

    intervals = np.diff(np.insert(arrival_minutes, 0, start_min))

    df = pd.DataFrame({
        "Arrival Time": arrival_times_str,
        "Interval": np.round(intervals, 2)
    })

    return df



# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def simulate_wait_times(arrival_times, n_passengers):
    def time_str_to_minutes(time_str):
        h, m, s = map(int, time_str.split(":"))
        return h * 60 + m + s / 60
    
    bus_minutes = arrival_times["Arrival Time"].apply(time_str_to_minutes).values
    max_bus_time = bus_minutes.max()
    
    passenger_arrival_minutes = np.sort(np.random.uniform(360, max_bus_time, n_passengers))
    
    passenger_data = []
    
    for p_time in passenger_arrival_minutes:
        for idx, b_time in enumerate(bus_minutes):
            if b_time >= p_time:
                wait_time = b_time - p_time
                passenger_data.append({
                    "Passenger Arrival Time": f"{int(p_time // 60):02d}:{int(p_time % 60):02d}:{int((p_time % 1) * 60):02d}",
                    "Bus Arrival Time": arrival_times.iloc[idx]["Arrival Time"],
                    "Bus Index": idx,
                    "Wait Time": round(wait_time, 2)
                })
                break
        else:
            passenger_data.append({
                "Passenger Arrival Time": f"{int(p_time // 60):02d}:{int(p_time % 60):02d}:{int((p_time % 1) * 60):02d}",
                "Bus Arrival Time": "No Bus",
                "Bus Index": -1,
                "Wait Time": None
            })
    
    return pd.DataFrame(passenger_data)


def visualize_wait_times(wait_times_df, timestamp):
    def time_str_to_minutes(time_str):
        h, m, s = map(int, time_str.split(":"))
        return h * 60 + m + s / 60

    start_min = time_str_to_minutes(timestamp.strftime("%H:%M:%S"))
    end_min = start_min + 60

    wait_times_df["Passenger_Minutes"] = wait_times_df["Passenger Arrival Time"].apply(time_str_to_minutes)
    filtered = wait_times_df[(wait_times_df["Passenger_Minutes"] >= start_min) &
                             (wait_times_df["Passenger_Minutes"] < end_min)].copy()

    bus_arrival_minutes = filtered["Bus Arrival Time"].apply(lambda x: time_str_to_minutes(x) if x != "No Bus" else None)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=bus_arrival_minutes,
        y=[0]*len(bus_arrival_minutes),
        mode="markers",
        marker=dict(color="blue", size=8),
        name="Buses"
    ))

    fig.add_trace(go.Scatter(
        x=filtered["Passenger_Minutes"],
        y=filtered["Wait Time"],
        mode="markers",
        marker=dict(color="red", size=8),
        name="Passengers"
    ))

    fig.update_layout(
        title="Passenger Wait Times in a 60-Minute Block",
        xaxis_title="Time (minutes) within the block",
        yaxis_title="Wait Time (minutes)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.5,
            xanchor="center",
            x=0.1
        )
    )

    return fig

