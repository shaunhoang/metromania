import pandas as pd
import numpy as np
from flask_caching import Cache
import plotly.graph_objects as go
import dash_leaflet as dl
from dash import dcc, html, Dash, no_update, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
from simplekml import Kml
import json
import ast

# import cleaned datasets
stations = pd.read_csv(
    "datasets/stations_cleaned.csv",
    converters={
        'linestring_latlon': lambda x: ast.literal_eval(x),
        'linestring_lonlat': lambda x: ast.literal_eval(x)
    },
    encoding='utf-8',
    keep_default_na=False
)
tracks = pd.read_csv(
    "datasets/tracks_cleaned.csv",
    converters={
        'linestring_latlon': lambda x: ast.literal_eval(x),
        'linestring_lonlat': lambda x: ast.literal_eval(x)
    },
    encoding='utf-8',
    keep_default_na=False,
)
cities = pd.read_csv(
    "datasets/cities_cleaned.csv", index_col=0,
    encoding='utf-8',
    keep_default_na=False,
)

stations['opening'] = pd.to_numeric(stations['opening'], errors='coerce')
stations['fromyear'] = pd.to_numeric(stations['fromyear'], errors='coerce')
stations['toyear'] = pd.to_numeric(stations['toyear'], errors='coerce')
tracks['opening'] = pd.to_numeric(tracks['opening'], errors='coerce')
tracks['fromyear'] = pd.to_numeric(tracks['fromyear'], errors='coerce')
tracks['toyear'] = pd.to_numeric(tracks['toyear'], errors='coerce')

# ---------------------------------
# Define quality thresholds using quantiles and prepare dropdown options
quality_threshold = 0.1  # 10% missing openings only
quantity_threshold = 100  # larger systems only

filtered_cities = cities[
    (cities['avg_wonkiness'] <= quality_threshold) &
    (cities['total_stations'] >= quantity_threshold)
]
filtered_cities = filtered_cities.sort_values(by=['country', 'city'])

# ---------------------------------
# --- Create Dropdown Options with Country Groups (filtered/sorted) ---
selection_items = []
current_country = None

for index, row in filtered_cities.iterrows():
    city_name = row['city']
    city_id = row['city_id']
    country_name = row['country']

    # Add a header for each country
    if country_name != current_country:
        if pd.notna(country_name):
            selection_items.append({
                'label': f"--- {country_name} ---",
                'value': "disabled",
                'disabled': True
            })
            current_country = country_name

    # Add the selectable city option
    selection_items.append({
        'label': f"  {city_name}",
        'value': city_id
    })

# ---------------------------------
# Helper function for placeholder graphs


def create_placeholder_figure(text_message):
    fig = go.Figure()
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": text_message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 16, "color": "#888888"}
            }
        ],
        plot_bgcolor="#222222",
        paper_bgcolor="#222222",
    )
    return fig


# ---------------------------------
# 2. Dash app creation and callbacks
FA_CSS = "https://use.fontawesome.com/releases/v5.15.4/css/all.css"
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, FA_CSS],
    title='Metromania',
    suppress_callback_exceptions=True
)

server = app.server
cache = Cache(app.server, config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 3600
})

# # ... (callbacks) ...

# Caching filtered data


def get_filtered_data(city, year):
    if not city or not year:
        return pd.DataFrame(), pd.DataFrame()

    my_stations = stations[(stations.city_id == city) &
                           ((stations.opening <= year) | (stations.opening.isna())) &
                           (stations.closure > year)
                           ]

    my_tracks = tracks[(tracks.city_id == city) &
                       ((tracks.opening <= year) | (tracks.opening.isna())) &
                       (tracks.closure > year)]

    return my_stations, my_tracks


@cache.memoize()
def get_summary_data(city):

    my_stations = stations[(stations.city_id == city)]
    my_tracks = tracks[(tracks.city_id == city)]

    joint_df = pd.concat([my_tracks.opening, my_stations.opening])

    valid_years = joint_df[(joint_df > 0) & (joint_df < 2041)].unique()
    if len(valid_years) < 2:
        return None

    min_year = int(min(valid_years))
    max_year = int(max(valid_years))

    if min_year >= max_year:
        return None

    data = []
    for y in range(min_year, max_year + 1):
        d = {'year': y,
             'track_length': my_tracks[(my_tracks.opening <= y) & (my_tracks.closure > y)].length.sum()/1000,
             'stations_num': len(my_stations[(my_stations.opening <= y) & (my_stations.closure > y)])}
        data.append(d)

    if not data:
        return None

    return pd.DataFrame(data)


# Calculate city center coordinates for viewport update
city_centers = {}
grouped_stations = stations.dropna(
    subset=['latitude', 'longitude']).groupby('city_id')
for city, group in grouped_stations:
    center = [group['latitude'].mean(), group['longitude'].mean()]
    city_centers[city] = center

# Define a default
DEFAULT_CITY = 69  # London
DEFAULT_CENTER = city_centers.get(DEFAULT_CITY, [51.51, -0.13])
DEFAULT_ZOOM = 11

# Map it function


@app.callback(
    Output('stations-layer', 'children'),
    Output('lines-layer', 'children'),
    Output('map', 'viewport'),
    Input('dropdown', 'value'),
    Input('slider', 'value'),
)
def map_it(city, year):
    if not city or not year:
        return [], [], {'center': DEFAULT_CENTER, 'zoom': DEFAULT_ZOOM}

    center = city_centers.get(city, DEFAULT_CENTER)
    zoom = DEFAULT_ZOOM

    my_stations, my_tracks = get_filtered_data(city, year)

    markers = []
    for _, station in my_stations.iterrows():

        tooltip_html_parts = []
        tooltip_html_parts.append(f"<b>Station:</b> {station.station_name}")
        if pd.notna(station.opening) and station.opening > 0:
            tooltip_html_parts.append(f"<b>Opened:</b> {station.opening:g}")

        # Join
        tooltip_inner_html = "<br>".join(tooltip_html_parts)
        tooltip_content = f"<div style='text-align: left;'>{tooltip_inner_html}</div>" if tooltip_inner_html else ""

        marker_children = []
        if tooltip_content:
            marker_children.append(dl.Tooltip(content=tooltip_content))

        markers.append(dl.CircleMarker(
            center=[station.latitude, station.longitude],
            radius=2.5,
            fillColor='white',
            fillOpacity=0.9,
            color='black',
            stroke=True,
            weight=0.5,
            pane='markerPane',
            children=marker_children
        ))

    lines = []
    if not my_tracks.empty:
        grouped_tracks = my_tracks.groupby('section_id')
        for section_id, group in grouped_tracks:
            # Get plottable info
            group.sort_values(by='fromyear', ascending=False, inplace=True)
            track_repr = group.iloc[0]   # use first row
            positions = track_repr['linestring_latlon']
            line_color = track_repr['line_color']

            tooltip_combination_blocks = []
            # MIN OPENING
            min_opening = group['opening'][group['opening'] > 0].min()
            opening_info = ""
            if pd.notna(min_opening):
                opening_info = f"<b>Opened:</b> {min_opening:g}<br>"
            tooltip_combination_blocks.append(opening_info.strip())

            # UNIQUE LINE / SYSTEM / MODE / SERVICE YEARS COMBINATIONS
            unique_combinations = group[[
                'line_name', 'system_name', 'transport_mode_name', 'fromyear', 'toyear']].drop_duplicates()

            for _, combo_row in unique_combinations.iterrows():
                block_parts = []
                line = combo_row['line_name']
                system = combo_row['system_name']
                mode = combo_row['transport_mode_name']
                fromyear = combo_row['fromyear']
                toyear = combo_row['toyear']

                # LINE / SYSTEM / MODE block
                if pd.notna(line) and line not in ['N.A.', '']:
                    block_parts.append(f"<b>Line:</b> {line}")
                if pd.notna(system) and system not in ['N.A.', '']:
                    block_parts.append(f"<b>System:</b> {system}")
                if pd.notna(mode):
                    block_parts.append(f"<b>Mode:</b> {mode}")

                # SERVICE YEARS
                service_parts = []
                if pd.notna(fromyear):
                    service_parts.append(f"from {int(fromyear):g}")
                if pd.notna(toyear):
                    service_parts.append(f"until {int(toyear):g}")
                if service_parts:
                    block_parts.append(
                        f"<b>Service:</b> {' '.join(service_parts)}")

                # Put together combo block
                if block_parts:
                    tooltip_combination_blocks.append("<br>".join(block_parts))

            # -PUT ALL TOGETHER INTO ONE CONTENT BLOCK
            tooltip_body = f"<hr style='margin: 5px 0;'>".join(
                tooltip_combination_blocks)  # separate by horizontal line
            # wrap in div
            tooltip_content = f"<div style='text-align: left;'>{tooltip_body}</div>" if tooltip_body else ""
            polyline_children = [dl.Tooltip(content=tooltip_content)] if tooltip_content else [
            ]  # create tooltip child if content exists

            # Create Polyline
            lines.append(dl.Polyline(
                positions=positions,
                color=line_color,
                weight=2,
                children=polyline_children
            ))

    return markers, lines, {'center': center, 'zoom': zoom}

# Plot it function


@app.callback(Output('plot', 'figure'), [Input('dropdown', 'value'), Input('slider', 'value')])
def plot_it(city, year):

    if not city or not year:
        return create_placeholder_figure("Select a city and year to see the plot.")

    # --- Get boundaries ---
    city_all_stations = stations[stations.city_id == city]
    city_name = city_all_stations['city'].iloc[0]

    if city_all_stations.empty:
        return create_placeholder_figure(f"No data found for this city.")
    x_min = city_all_stations['longitude'].min()
    x_max = city_all_stations['longitude'].max()
    y_min = city_all_stations['latitude'].min()
    y_max = city_all_stations['latitude'].max()

    # --- Get track data ---
    _, my_tracks_current = get_filtered_data(city, year)
    _, my_tracks_previous = get_filtered_data(city, year - 1)

    # Calculate average latitude for scaling using cosine adjustment
    if not city_all_stations.empty and pd.notna(city_all_stations['latitude'].mean()):
        avg_latitude_deg = city_all_stations['latitude'].mean()
        avg_latitude_rad = np.radians(avg_latitude_deg)
        # Avoid division by zero near poles (cosine(90 degrees) = 0)
        if abs(avg_latitude_deg) >= 89.9:
            scale_ratio = 1 / \
                np.cos(np.radians(np.sign(avg_latitude_deg) * 89.9))
        else:
            scale_ratio = 1 / np.cos(avg_latitude_rad)
    else:
        scale_ratio = 1

    fig = go.Figure()

    if not my_tracks_current.empty:
        # --- Identify new section IDs ---
        current_section_ids = set(my_tracks_current['section_id'])
        previous_section_ids = set(my_tracks_previous['section_id'])
        new_section_ids = current_section_ids - previous_section_ids

        # --- Add each track segment as a separate trace ---
        for index, track_row in my_tracks_current.iterrows():
            section_id = track_row['section_id']
            linesegment = track_row['linestring_lonlat']

            # Check for valid list data
            if isinstance(linesegment, list) and len(linesegment) > 1:
                x_coords = [p[0] for p in linesegment if isinstance(
                    p, (list, tuple)) and len(p) == 2]
                y_coords = [p[1] for p in linesegment if isinstance(
                    p, (list, tuple)) and len(p) == 2]
                line_color = 'yellow' if section_id in new_section_ids else 'white'
                line_width = 3 if section_id in new_section_ids else 1

                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(color=line_color, width=line_width),
                    hoverinfo='none'
                ))
            else:
                print(
                    f"WARN: Skipping invalid linestring for section {section_id}")

    # --- Apply axis settings ---
    fig.update_yaxes(title_text="", showgrid=False,
                     showline=False, mirror=True, showticklabels=False, ticks='', automargin=True, zeroline=False,
                     scaleratio=scale_ratio,
                     scaleanchor="x",
                     range=[y_min, y_max]
                     )

    fig.update_xaxes(title_text="", showgrid=False,
                     showline=False, mirror=True,
                     showticklabels=False, ticks='', automargin=True, zeroline=False,
                     range=[x_min, x_max])

    # --- Add annotations ---
    fig.add_annotation(
        text=f"<b>{city_name.upper()} {year:g}</b>",
        xref="paper", yref="paper", x=0.98, y=0.98,
        showarrow=False, font=dict(size=16, color="white"),
        xanchor='right', yanchor='top'
    )
    fig.add_annotation(
        text="<span style='color:yellow;'><b>new segments</b></span> added this year",
        xref="paper", yref="paper", x=0.98, y=0.94,
        showarrow=False, font=dict(size=10, color="white"),
        xanchor='right', yanchor='top'
    )

    # --- Update layout ---
    fig.update_layout(showlegend=False,
                      template="plotly_dark",
                      autosize=True,
                      plot_bgcolor="#222222",
                      paper_bgcolor="#222222",
                      margin=dict(l=5, r=5, t=10, b=5, pad=5))

    # --- Placeholder if no data ---
    if not fig.data:
        placeholder_fig = create_placeholder_figure(
            f"No tracks found for this city in {year}.")
        placeholder_fig.update_layout(
            xaxis=dict(range=[x_min, x_max], visible=False),
            yaxis=dict(range=[y_min, y_max], visible=False,
                       scaleanchor="x", scaleratio=1)
        )
        placeholder_fig.add_annotation(text=f"<b>{year:g}</b>", xref="paper", yref="paper", x=0.98,
                                       y=0.98, showarrow=False, font=dict(size=24, color="white"), xanchor='right', yanchor='top')
        placeholder_fig.add_annotation(text="<span style='color:yellow;'><b>New sections</b></span> in operation that year", xref="paper",
                                       yref="paper", x=0.02, y=0.02, showarrow=False, font=dict(size=14, color="white"), xanchor='left', yanchor='bottom')
        return placeholder_fig

    return fig

# Count it function


@app.callback(Output('count', 'children'), [Input('dropdown', 'value'), Input('slider', 'value')])
def count_it(city, year):
    if not city or not year:
        return html.P("Select city and year for stats.", className="text-center text-muted")

    my_stations, my_tracks = get_filtered_data(city, year)

    track_length_km = my_tracks.length.sum()/1000
    num_stations = len(my_stations)

    return [
        dbc.Card(
            [
                dbc.CardBody([
                    html.P(f"{num_stations} stations",
                           className="card-title mb-0 fs-6"),
                ], className="p-1")
            ],
            color="primary",
            outline=True,
            className="text-center mb-2"
        ),
        dbc.Card(
            [
                dbc.CardBody([
                    html.P(f"{track_length_km:,.0f} km of tracks",
                           className="card-title mb-0 fs-6"),
                ], className="p-1")
            ],
            color="primary",
            outline=True,
            className="text-center"
        )
    ]

# Summarize it function


@app.callback(Output('summarize', 'figure'), [Input('dropdown', 'value'), Input('slider', 'value')])
def summarize_it(city, year):

    if not city:
        return create_placeholder_figure("Select a city to see its growth history.")

    dataset = get_summary_data(city)
    if dataset is None:
        return create_placeholder_figure(f"No historical data for this city.")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=dataset.year, y=dataset.stations_num,
                   name="Stations", mode='lines'),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=dataset.year, y=dataset.track_length,
                   name="Track (km)", mode='lines'),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Number of stations",
                     secondary_y=False, showgrid=False, zeroline=False)
    fig.update_yaxes(title_text="Track length (km)",
                     secondary_y=True, showgrid=False, zeroline=False)

    if year:
        fig.add_vline(x=year, line_width=2, line_dash="dash", line_color="red")

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#222222",
        paper_bgcolor="#222222",
        margin=dict(l=20, r=20, t=40, b=20, pad=4),
        title=dict(
            text='Growth Over Time',
            yanchor='top',
            y=0.95,
            xanchor='center',
            x=0.5
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.02,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        xaxis=dict(title_font=dict(size=12), tickfont=dict(size=10)),
        yaxis=dict(title_font=dict(size=12), tickfont=dict(size=10)),
        yaxis2=dict(title_font=dict(size=12), tickfont=dict(size=10)),
        autosize=True
    )

    return fig

# Export it function


@app.callback(
    Output("download-geojson", "data"),
    [State('dropdown', 'value'),
     State('slider', 'value')],
    Input("export_geojson_button", "n_clicks"),
    prevent_initial_call=True,
)
def export_geojson(city, year, n_clicks):
    if not city or not year:
        return no_update

    my_stations, my_tracks = get_filtered_data(city, year)
    city_name = my_stations['city'].iloc[0]

    features = []

    # Add stations as Point features
    for i in range(len(my_stations)):
        station = my_stations.iloc[i]
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [station.longitude, station.latitude]
            },
            "properties": {
                "name": station.station_name,
                "opening": f"{station.opening:g}",
                "line": station.line_name
            }
        }
        features.append(feature)

    # Add tracks as LineString features
    for i in range(len(my_tracks)):
        track = my_tracks.iloc[i]
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": track.linestring_lonlat
            },
            "properties": {
                "name": track.line_name,
                "opening": f"{track.opening:g}",
                "color": track.line_color
            }
        }
        features.append(feature)

    # Create the final FeatureCollection
    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }

    # Create data dictionary for download
    download_data = dict(
        content=json.dumps(geojson_data),
        filename=f"metromania_{city_name.replace(' ', '_')}_{year:g}.geojson"
    )

    return download_data

# Export it function


@app.callback(
    Output("download-kml", "data"),
    [State('dropdown', 'value'),
     State('slider', 'value')],
    Input("export_kml_button", "n_clicks"),
    prevent_initial_call=True,
)
def export_kml(city, year, n_clicks):
    if not city or not year:
        return no_update

    my_stations, my_tracks = get_filtered_data(city, year)
    city_name = my_stations['city'].iloc[0]
    kml = Kml()

    # --- Stations ---
    list_st = []
    for i in range(len(my_stations)):
        d = [my_stations.station_name.iloc[i], f'{my_stations.opening.iloc[i]:g}', my_stations.line_name.iloc[i],
             my_stations.latitude.iloc[i], my_stations.longitude.iloc[i]]
        list_st.append(d)

    for row in list_st:
        kml.newpoint(name=row[0], description=row[2],
                     coords=[(row[4], row[3])])

    # --- Tracks ---
    list_tr = []
    for i in range(len(my_tracks)):
        d = [my_tracks.line_name.iloc[i], my_tracks.linestring_lonlat.iloc[i],
             f'{my_tracks.opening.iloc[i]:g}']
        list_tr.append(d)

    for row in list_tr:
        kml.newlinestring(name=row[0], description=row[2], coords=row[1])

    kml_data = dict(
        content=kml.kml(),
        filename=f"metromania_{city_name.replace(' ', '_')}_{year:g}.kml"
    )

    return kml_data

# Update values


@app.callback(
    Output('slider', 'value'),
    Input('year-backward-button', 'n_clicks'),
    Input('year-forward-button', 'n_clicks'),
    State('slider', 'value'),
    prevent_initial_call=True
)
def update_slider_value(n_back, n_fwd, current_year):
    if not ctx.triggered_id:
        return no_update

    min_year = 1860
    max_year = 2040

    if current_year is None:
        current_year = 2025

    if ctx.triggered_id == 'year-backward-button':
        new_year = current_year - 1
        return max(new_year, min_year)
    elif ctx.triggered_id == 'year-forward-button':
        new_year = current_year + 1
        return min(new_year, max_year)

    return no_update


@app.callback(
    Output('map-card-header', 'children'),
    Input('dropdown', 'value'),
    Input('slider', 'value')
)
def update_map_card_title(city_id, year):
    if not city_id or not year:
        return "Interactive Map"
    if city_id == "disabled":
        return "Interactive Map"

    city_name_series = cities.loc[cities['city_id'] == city_id, 'city']
    if not city_name_series.empty:
        city_name = city_name_series.iloc[0]
        return f"{city_name} in {year:g}"
    else:
        return f"Map for ID {city_id} in {year:g}"


# ---------------------------------
# 3. Create Dash Layout
# Slider marks
slider_marks = {}
for i in range(1860, 2041, 20):
    slider_marks[i] = {'label': f'{i:g}'}

navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand([
            html.I(className="fas fa-subway fa-lg me-2"),
            html.Span("Metromania", className="fw-bold fs-5 me-1"),
            html.Span(" | Transit History Explorer",
                      className="text-light opacity-75 fs-5")
        ], className="d-flex align-items-center")
    ]),
    color="primary", dark=True, className="py-2 mb-4 shadow-sm"
)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .Select-control, .Select-menu-outer, .Select, .Select-value-label, .Select-placeholder {
                background-color: #222 !important;
                color: white !important;
            }
            .Select-option {
                background-color: #222 !important;
                color: white !important;
            }
            .Select-option.is-focused, .Select-option.is-selected {
                background-color: #444 !important;
                color: white !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# --- Controls, Stats, and Export Cards ---

controls = dbc.Card(
    dbc.CardBody([
        # Row 1: City
        dbc.Row([
            dbc.Col(html.Label(
                "CITY", className="d-flex align-items-center mb-0 justify-content-center fw-bold"), width=1),
            dbc.Col(
                dcc.Dropdown(
                    id='dropdown',
                    options=selection_items,
                    placeholder="Select a city...",
                    className="dbc",
                    value=DEFAULT_CITY
                ),
                width=True
            )
        ], className="mb-4 align-items-center g-3"),

        # Row 2: Year
        dbc.Row([
            dbc.Col(html.Label(
                "YEAR", className="d-flex align-items-center mb-0 justify-content-center fw-bold"), width=1),
            dbc.Col(dbc.Button("<", id="year-backward-button",
                    n_clicks=0, color="secondary", size="sm"), width="auto"),
            dbc.Col(
                dcc.Slider(
                    id='slider',
                    min=1860,
                    max=2040,
                    step=1,
                    included=True,
                    marks=slider_marks,
                    tooltip={"placement": "bottom", "always_visible": True},
                    value=2012
                ),
                width=True,
            ),
            dbc.Col(dbc.Button(">", id="year-forward-button",
                    n_clicks=0, color="secondary", size="sm"), width="auto")
        ], className="mb-2 align-items-center g-3"),
    ]),
    className="rounded-3 h-100"
)

stats_card = dbc.Card(
    dbc.CardBody([
        html.P("SNAPSHOT", className="text-center fw-bold mb-2"),
        dbc.Row(id='count', children=[
            dbc.Col(html.P("Select city and year for stats.",
                           className="text-center text-muted"), sm=12)
        ])
    ]),
    className="rounded-3 h-100"
)

export_card = dbc.Card(
    dbc.CardBody([
        html.P("EXPORT", className="text-center fw-bold mb-2"),
        dbc.Button("KML", id="export_kml_button",
                   color="success", className="w-100 mb-2", size="sm"),
        dbc.Button("GeoJSON", id="export_geojson_button",
                   color="info", className="w-100", size="sm"),
        dcc.Download(id="download-kml"),
        dcc.Download(id="download-geojson")
    ]),
    className="d-rounded-3 h-100"
)

app.layout = html.Div([
    navbar,
    dbc.Container([
        # Title Section
        dbc.Row([
            dbc.Col([
                html.H4(
                    "Explore and export how transit systems around the world have evolved",
                    className="text-center mb-2 ms-3"
                ),
                dbc.Col([
                    html.P("Data quality may vary greatly! Small systems or those with >10% missing segment dates are not included.",
                           className="text-muted small text-center"),
                ], className="align-self-center"),
            ], width=12)
        ]),

        # Controls and Side Cards (Stats + Export)
        dbc.Row([
            dbc.Col(
                controls, 
                width=12,lg=8
            ),
            dbc.Col(
                dbc.Row([
                    dbc.Col(stats_card, width=7),
                    dbc.Col(export_card, width=5)
                ],className="g-2"),
                width=12, lg=4, className="g-2"
            )
        ], align="stretch", className="mb-2 g-2"),

        # Map Row
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(id="map-card-header", children="Interactive Map",
                                   className="fw-bold justify-content-center text-center"),
                    dbc.CardBody([
                        dl.Map(
                            id='map',
                            center=DEFAULT_CENTER,
                            zoom=DEFAULT_ZOOM,
                            style={'width': '100%', 'height': '70vh',
                                   'borderRadius': '12px'},
                            children=[
                                dl.TileLayer(
                                    id='base-layer',
                                    url='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
                                ),
                                dl.LayerGroup(id='lines-layer', children=[]),
                                dl.LayerGroup(
                                    id='stations-layer', children=[]),
                            ]
                        )
                    ])
                ], className="shadow-lg border-0 rounded-3"), md=12, className="mb-2"
            )
        ]),

        # Plot + Summary Graph
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(id='plot', style={
                                  'height': '60vh'}, figure=create_placeholder_figure("Loading map..."))
                    ])
                ], className="shadow-lg border-0 rounded-3 mb-3"), md=5,
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(
                            id='summarize',
                            style={'height': '60vh'},
                            figure=create_placeholder_figure(
                                "Loading chart...")
                        )
                    ])
                ], className="shadow-lg border-0 rounded-3"), md=7
            )
        ], className="mb-2"),

        html.Hr(className="text-muted"),

        # Footer
        html.Footer([
            dbc.Row([

                dbc.Col([
                    "Created by: Shaun Hoang",
                    dbc.Row([
                        html.Span([
                            html.A(html.I(className="far fa-envelope fa-lg mx-2"),
                                   href="mailto:shaun.hoang@gmail.com", target="_blank"),
                            html.A(html.I(className="fab fa-github fa-lg mx-2"),
                                   href="https://github.com/shaunhoang/metromania", target="_blank"),
                            html.A(html.I(className="fab fa-linkedin fa-lg mx-2"),
                                   href="https://www.linkedin.com/in/shaunhoang", target="_blank")
                        ], className="ms-2")
                    ], className="mt-1 justify-content-center")
                ], className="text-center"),

                dbc.Col([
                    "Data source: ",
                    html.A("CityLines.co", href='https://www.citylines.co/',
                           target="_blank", className="link-info"),
                    dbc.Row([
                        " (updated 07/2025)"
                    ], className="mt-1 justify-content-center")
                ], className="text-center")

            ], className="mb-2")

        ]),

    ], fluid=True)
])


# Finally
if __name__ == "__main__":
    app.run(debug=True)
