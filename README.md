# San Diego Bus Schedule and Route Visualization

This project focuses on processing and visualizing public transportation data, specifically San Diego bus routes. It integrates various datasets including bus schedules, stops, trips, and route information to create informative, interactive visualizations and insights into the structure and frequency of bus service in the city.

## Project Overview

The main goals of this project were to:
- Clean and merge data from multiple transportation-related sources.
- Generate a detailed bus schedule with geographic coordinates.
- Visualize routes and stops using `plotly` and `geopandas`.
- Enable interactive geographic insights into transit patterns in San Diego.

## Libraries Used

- `pandas` and `numpy` for data manipulation
- `geopandas` and `shapely` for geographic processing
- `plotly` for interactive visualization
- `pathlib` and `warnings` for environment handling

## Features

- `create_detailed_schedule`: Merges stop, trip, and schedule data to create a unified, ordered transit dataset.
- Interactive plots showing bus routes, directions, and stop distributions.
- Filtered visualization by route ID and direction to explore service-specific insights.

## Data Sources

The project expects GTFS (General Transit Feed Specification) format data:
- `schedule.csv`
- `stops.csv`
- `trips.csv`

These should be preloaded or sourced from San Diego's regional transit authority or similar providers.

