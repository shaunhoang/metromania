import pandas as pd
from flask_caching import Cache
import plotly.express as px
import dash_leaflet as dl
from dash import dcc, html, Dash, no_update, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from simplekml import Kml
import json

######## Data intake and cleaning
cities = pd.read_csv("datasets/cities.csv")
stations = pd.read_csv("datasets/stations.csv")
tracks = pd.read_csv("datasets/tracks.csv")
lines = pd.read_csv("datasets/lines.csv")
track_lines = pd.read_csv("datasets/track_lines.csv")
station_lines = pd.read_csv("datasets/station_lines.csv")
systems = pd.read_csv("datasets/systems.csv")


stations = stations.rename(columns={'id':'station_id','name':'station_name'})
tracks = tracks.rename(columns={'id':'section_id'})
cities_simpl = pd.DataFrame({'city_id':cities.id,'country':cities.country,'city':cities.name})
station_lines_simpl = pd.DataFrame({'station_id':station_lines.station_id,'line_id':station_lines.line_id})
lines_simpl = pd.DataFrame({'line_id':lines.id,'line_name':lines.name,'line_color':lines.color,'system_id':lines.system_id})
systems_simpl = pd.DataFrame({'system_id':systems.id,'system_name':systems.name})
track_lines_simpl = pd.DataFrame({'section_id':track_lines.section_id,'line_id':track_lines.line_id})


##--- Merge multiple datasets into STATIONS
stations = pd.merge (stations, cities_simpl, how='left',on='city_id')
stations = pd.merge (stations, station_lines_simpl, how='left',on='station_id')
stations = pd.merge (stations, lines_simpl, how='left',on='line_id')
stations = pd.merge (stations, systems_simpl, how='left',on='system_id')

# Split 'geometry' into 'longitudes' and 'latitudes'
stations['longitude'] = stations['geometry'].apply(lambda x: x.split('POINT(')[1].split(' ')[0])
stations['longitude'] = stations['longitude'].apply(lambda x: float(x))
stations['latitude'] = stations['geometry'].apply(lambda x: x.split('POINT(')[1].split(' ')[1].split(')')[0])
stations['latitude'] = stations['latitude'].apply(lambda x: float(x))

# Reorder columns in STATIONS and clean up
stations = stations[['station_id','station_name','geometry','longitude',
                     'latitude','opening','closure','city_id','city',
                     'country','line_id','line_name','line_color','system_id','system_name']]

##--- Merge multiple datasets into TRACKS
tracks = pd.merge(tracks, track_lines_simpl,how='left',on='section_id')
tracks = pd.merge(tracks, lines_simpl, how='left', on='line_id')
tracks = pd.merge(tracks, cities_simpl, how='left', on='city_id')
tracks = pd.merge(tracks, systems_simpl, how='left', on='system_id')

# Define function to split coord from linestring object - two versions for different applications
def split_coord_lonlat(x):
    stripped_x = x.rstrip(')) ').lstrip(' MULTILINESTRING ((').strip() # strip non-numerical values from object 
    coord_list = []
    for point in stripped_x.split(','):
        coord = point.split(' ')                                # split into lon-lat 
        coord = [float(x.strip()) for x in coord]               # turn to float
        coord_list.append(coord)
    return coord_list

def split_coord_latlon(x):
    stripped_x = x.rstrip(')) ').lstrip(' MULTILINESTRING ((').strip() # strip non-numerical values from object 
    coord_list = []
    for point in stripped_x.split(','):
        coord = point.split(' ')                              # split into lon-lat 
        coord[0],coord[1] = coord[1],coord[0]                   # swap to lat-lon
        coord = [float(x.strip()) for x in coord]               # turn to float
        coord_list.append(coord)
    return coord_list

# Split 'geometry' for each row into 'linestring' a list of coordinates to draw track lines
tracks['linestring_latlon'] = tracks.geometry.apply(split_coord_latlon)
tracks['linestring_lonlat'] = tracks.geometry.apply(split_coord_lonlat)

# Reorder columns
tracks = tracks[['section_id','geometry','linestring_latlon','linestring_lonlat','opening','closure',
                 'length','line_id','line_name','line_color',
                 'system_id','system_name','city_id','city','country']]

# --- Handle name missing data ---

## Fill in NA 
stations['station_name'] = stations['station_name'].fillna('n/a')
stations['line_name'] = stations['line_name'].fillna('n/a')
stations['line_color'] = stations['line_color'].fillna('#000000')
tracks['line_name'] = tracks['line_name'].fillna('n/a')
tracks['line_color'] = tracks['line_color'].fillna('#000000')
stations['line_id'] = stations['line_id'].fillna(0)
tracks['line_id'] = tracks['line_id'].fillna(0)

# --- Handle opening and closure date missing data ---

# Handle Closure Dates, Fill NaN with 999999, "still open".
stations['closure'] = stations['closure'].fillna(999999)
tracks['closure'] = tracks['closure'].fillna(999999)

# Handle Opening Dates, impute missing data based on minimum known opening year for each city
stations['opening'] = pd.to_numeric(stations['opening'], errors='coerce')
tracks['opening'] = pd.to_numeric(tracks['opening'], errors='coerce')
stations.loc[stations['opening'] > 2100, 'opening'] = pd.NA
tracks.loc[tracks['opening'] > 2100, 'opening'] = pd.NA
stations.loc[stations['opening'] == 0, 'opening'] = pd.NA
tracks.loc[tracks['opening'] == 0, 'opening'] = pd.NA
stations['opening'] = stations['opening'].astype('Int64')
tracks['opening'] = tracks['opening'].astype('Int64')

# Flag rows
stations['was_missing_opening'] = stations['opening'].isna() 
tracks['was_missing_opening'] = tracks['opening'].isna()   

# Impute
stations['min_city_opening'] = stations.groupby('city')['opening'].transform('min')
tracks['min_city_opening'] = tracks.groupby('city')['opening'].transform('min')
stations['opening'] = stations['opening'].fillna(stations['min_city_opening'])
tracks['opening'] = tracks['opening'].fillna(tracks['min_city_opening'])
# Handle cities where all dates were missing (min_city_opening is NaN)
stations['opening'] = stations['opening'].fillna(1900).astype(int) 
tracks['opening'] = tracks['opening'].fillna(1900).astype(int) 
stations = stations.drop(columns=['min_city_opening']) # Drop the temporary column
tracks = tracks.drop(columns=['min_city_opening']) # Drop the temporary column

# ... (wonkiness calculation) ...
# Calculate proportion of missing/imputed opening dates per city
pivot_st = pd.pivot_table(stations, values='station_id', index='city', columns='was_missing_opening', aggfunc='count', fill_value=0)
pivot_tr = pd.pivot_table(tracks, values='section_id', index='city', columns='was_missing_opening', aggfunc='count', fill_value=0)

pivot_st = pivot_st.rename(columns={True: 'missing', False: 'valid'})
pivot_tr = pivot_tr.rename(columns={True: 'missing', False: 'valid'})
if 'missing' not in pivot_st.columns: pivot_st['missing'] = 0
if 'valid' not in pivot_st.columns: pivot_st['valid'] = 0
if 'missing' not in pivot_tr.columns: pivot_tr['missing'] = 0
if 'valid' not in pivot_tr.columns: pivot_tr['valid'] = 0

# Calculate the "wonkiness" score then merge for filtering
pivot_st['wonkiness_st'] = pivot_st['missing'] / (pivot_st['valid'] + pivot_st['missing'])
pivot_tr['wonkiness_tr'] = pivot_tr['missing'] / (pivot_tr['valid'] + pivot_tr['missing'])
wonk_table = pd.merge(pivot_st[['valid', 'missing', 'wonkiness_st']],
                      pivot_tr[['valid', 'missing', 'wonkiness_tr']],
                      on='city', suffixes=('_st', '_tr'))
wonk_table['total_stations'] = wonk_table['valid_st'] + wonk_table['missing_st']
wonk_table['total_tracks'] = wonk_table['valid_tr'] + wonk_table['missing_tr']
wonk_table['avg_wonkiness'] = (wonk_table['wonkiness_st'] + wonk_table['wonkiness_tr']) / 2 

# Define quality thresholds using quantiles
low_wonk_threshold = wonk_table['avg_wonkiness'].quantile(0.6)
selection_items = []
for index, row in wonk_table.iterrows():
    if row['avg_wonkiness'] <= low_wonk_threshold:
        label = index
        selection_items.append({'label': label, 'value': index})

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
######## 2. Dash app creation and callbacks
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
@cache.memoize()
def get_filtered_data(city, year):
    if not city or not year:
        return pd.DataFrame(), pd.DataFrame()

    my_stations = stations[(stations.city == city) & 
                           (stations.opening <= year) & 
                           (stations.closure > year)]
    
    my_tracks = tracks[(tracks.city == city) & 
                       (tracks.opening <= year) & 
                       (tracks.closure > year)]
                       
    return my_stations, my_tracks
@cache.memoize()
def get_summary_data(city):

    my_stations = stations[(stations.city == city)]
    my_tracks = tracks[(tracks.city == city)]

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
             'track_length' : my_tracks[(my_tracks.opening <= y) & (my_tracks.closure > y)].length.sum()/1000,
             'stations_num' : len(my_stations[(my_stations.opening <= y) & (my_stations.closure > y)])}
        data.append(d)
        
    if not data:
        return None

    return pd.DataFrame(data)

@app.callback(
    Output('stations-layer', 'children'),
    Output('lines-layer', 'children'),
    Output('map', 'center'),
    Output('map', 'zoom'),
    Input('dropdown', 'value'),
    Input('slider', 'value')
)
def map_it(city, year):

    if not city or not year:
        return [], [], [0, 0], 2
    my_stations, my_tracks = get_filtered_data(city, year)
    
    markers = []
    lines = []    

    for _, station in my_stations.iterrows():
        markers.append(dl.CircleMarker(
            center=[station.latitude, station.longitude],
            radius=3,
            fillColor='white',
            fillOpacity=0.8,
            color=station.line_color,
            stroke=True,
            weight=0.5,
            pane='markerPane',
        ))

    for _, track in my_tracks.iterrows():
        lines.append(dl.Polyline(
            positions=track.linestring_latlon,
            color=track.line_color,
            weight=2,
            children=[
                dl.Tooltip(f"{track.line_name} (Opened {track.opening:g})")
            ]
        ))

    if not my_stations.empty:
        center = [my_stations['latitude'].mean(), my_stations['longitude'].mean()]
        zoom = 11
    else:
        center, zoom = [0, 0], 2

    return markers, lines, center, zoom

# Plot it function
@app.callback(Output('plot','figure'),[Input('dropdown','value'),Input('slider','value')])
def plot_it(city,year):

    if not city or not year:
        return create_placeholder_figure("Select a city and year to see the plot.")

    city_all_stations = stations[stations.city == city]
    if city_all_stations.empty:
      return create_placeholder_figure(f"No data found for {city}.")
    x_min = city_all_stations['longitude'].min()
    x_max = city_all_stations['longitude'].max()
    y_min = city_all_stations['latitude'].min()
    y_max = city_all_stations['latitude'].max()

    # --- Get track data for current and previous year ---
    _, my_tracks_current = get_filtered_data(city, year)
    _, my_tracks_previous = get_filtered_data(city, year - 1)

    if my_tracks_current.empty:
        fig = create_placeholder_figure(f"No tracks found for {city} in {year}.")
    else:
        # --- Identify new section IDs ---
        current_section_ids = set(my_tracks_current['section_id'])
        previous_section_ids = set(my_tracks_previous['section_id'])
        new_section_ids = current_section_ids - previous_section_ids

        # --- Prepare plotting DataFrame ---
        plot_df = my_tracks_current[['section_id', 'linestring_lonlat']].explode('linestring_lonlat')
        plot_df[['x', 'y']] = pd.DataFrame(plot_df['linestring_lonlat'].tolist(), index=plot_df.index)
        plot_df = plot_df.rename(columns={'section_id': 'group'})
        plot_df['color'] = plot_df['group'].apply(lambda section_id: 'yellow' if section_id in new_section_ids else 'white') # Mark new sections

        # --- Create plot using color mapping ---
        fig = px.line(plot_df,
                      x="x",
                      y="y",
                      line_group="group",
                      color='color', 
                      color_discrete_map={ 
                          'white': 'white',
                          'yellow': 'yellow'
                      },
                      template="plotly_dark")

        fig.update_traces(line=dict(width=2))

    fig.update_yaxes(title_text="",showgrid=False,
                     showline=False,mirror=True,showticklabels=False,ticks='',automargin=True, zeroline=False,
                     scaleratio=1,
                     scaleanchor="x",
                     range=[y_min, y_max]
                     )

    fig.update_xaxes(title_text="",showgrid=False,
                     showline=False,mirror=True,
                     showticklabels=False,ticks='',automargin=True, zeroline=False,
                     range=[x_min, x_max])

    fig.add_annotation(
        text=f"<b>{year:g}</b>",
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        showarrow=False,
        font=dict(size=24, color="white"),
        xanchor='right',
        yanchor='top'
    )
    # add annotation for new segments
    fig.add_annotation(
        text="<span style='color:yellow;'><b>New sections</b></span> opened that year",
        xref="paper", yref="paper",
        x=0.02, y=0.02,
        showarrow=False,
        font=dict(size=14, color="white"),
        xanchor='left',
        yanchor='top'
    )

    fig.update_layout(showlegend=False, 
                      autosize=True,    
                      plot_bgcolor="#222222",
                      paper_bgcolor="#222222",
                      margin=dict(l=10, r=10, t=10, b=10, pad=4)) 

    return fig
  
# Count it function
@app.callback(Output('count','children'),[Input('dropdown','value'),Input('slider','value')])
def count_it(city,year):
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
                           className="card-title mb-0"),
                ], className="p-2")
            ],
            color="primary",
            outline=True,
            className="text-center mb-2" 
        ),
        dbc.Card(
            [
                dbc.CardBody([
                    html.P(f"{track_length_km:,.0f} km of tracks",
                           className="card-title mb-0"),
                ], className="p-2")
            ],
            color="primary",
            outline=True,
            className="text-center"
        )
    ]

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

    min_year = 1850
    max_year = 2030
    
    if current_year is None:
        current_year = 2025 

    if ctx.triggered_id == 'year-backward-button':
        new_year = current_year - 1
        return max(new_year, min_year) # Don't go below min
    elif ctx.triggered_id == 'year-forward-button':
        new_year = current_year + 1
        return min(new_year, max_year) # Don't go above max

    return no_update

# Summarize it function
@app.callback(Output('summarize','figure'),[Input('dropdown','value'),Input('slider','value')])
def summarize_it(city,year):
    
    if not city:
        return create_placeholder_figure("Select a city to see its growth history.")
    
    dataset = get_summary_data(city)
    if dataset is None:
         return create_placeholder_figure(f"No historical data for {city}.")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=dataset.year, y=dataset.stations_num, name="Stations", mode='lines'),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=dataset.year, y=dataset.track_length, name="Track (km)", mode='lines'),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Number of stations", secondary_y=False,showgrid=False,zeroline=False)
    fig.update_yaxes(title_text="Track length (km)", secondary_y=True,showgrid=False,zeroline=False)    
    
    if year:
        fig.add_vline(x=year, line_width=2, line_dash="dash", line_color="red")
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#222222",
        paper_bgcolor="#222222",
        title={
            'text': f"{city} Urban Rail System Growth Over Time",
             'y':0.9,
             'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        legend={
            'yanchor':"top",
            'y':0.99,
            'xanchor':"left",
            'x':0.01
    })
    
    return fig


# Export it function - Into 1 GeoJSON file
@app.callback(
    Output("download-geojson", "data"),
    [State('dropdown','value'),
     State('slider','value')],
    Input("export_geojson_button", "n_clicks"),
    prevent_initial_call=True,
)
def export_geojson(city, year, n_clicks):
    if not city or not year:
        return no_update

    my_stations, my_tracks = get_filtered_data(city, year)

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
                "coordinates": track.linestring_lonlat # Already in [lon, lat] format
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
        filename=f"metromania_{city.replace(' ', '_')}_{year:g}.geojson"
    )

    return download_data

# Export it function
@app.callback(
    Output("download-kml", "data"),
    [State('dropdown','value'),
     State('slider','value')],
    Input("export_kml_button", "n_clicks"),
    prevent_initial_call=True,
)
def export_kml(city, year, n_clicks): 
    
    if not city or not year:
        return no_update

    my_stations, my_tracks = get_filtered_data(city, year)

    kml = Kml(name=f"{city} Transit {year:g}")
    
    # --- Stations ---
    list_st=[]
    for i in range(len(my_stations)):
        d = [my_stations.station_name.iloc[i],f'{my_stations.opening.iloc[i]:g}', my_stations.line_name.iloc[i],
             my_stations.latitude.iloc[i],my_stations.longitude.iloc[i]]
        list_st.append(d)

    for row in list_st:
        kml.newpoint(name=row[0], description=row[2],
                        coords=[(row[4], row[3])]) 

    # --- Tracks ---
    list_tr=[]
    for i in range(len(my_tracks)):
        d = [my_tracks.line_name.iloc[i], my_tracks.linestring_lonlat.iloc[i],f'{my_tracks.opening.iloc[i]:g}']        
        list_tr.append(d)   

    for row in list_tr:
        kml.newlinestring(name=row[0],description=row[2],coords=row[1])

    kml_data = dict(
        content=kml.kml(),
        filename=f"metromania_{city.replace(' ', '_')}_{year:g}.kml"
    )

    return kml_data

# ---------------------------------
######## 3. Create Dash Layout

# Dropdown list
slider_marks = {}
for i in range(1850,2031,10):
    slider_marks[i] =  {'label': f'{i:g}'}

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(html.I(className="fas fa-subway fa-lg me-2"), width="auto"),
                    dbc.Col(
                        dbc.NavbarBrand([
                            html.Span("Metromania", className="fw-bold fs-4"),
                            html.Span(" – Urban Rail History Explorer", className="text-muted ms-1 fs-4")
                        ]),
                        width="auto"
                    ),
                ],
                align="center",
                className="g-0",
            ),

            dbc.Row(
                [
                    dbc.Col(
                        html.A(html.I(className="fab fa-github fa-lg me-2"), 
                               href="https://github.com/shaunhoang/metromania", target="_blank"),
                        width="auto",
                        className="ms-3"
                    ),
                    dbc.Col(
                        html.A(html.I(className="fab fa-linkedin fa-lg me-2"), 
                               href="https://www.linkedin.com/in/shaunhoang", target="_blank"),
                        width="auto",
                        className="ms-3"
                    ),
                ],
                align="center",
                className="g-0"
            ),
        ],
        className="d-flex justify-content-between"
    ),
    color="primary",
    dark=True,
    className="mb-4 py-3"  
)


controls = dbc.Card(
    dbc.CardBody([
        # Row 1: City
        dbc.Row([
            dbc.Col(html.Label("Select City:", className="d-flex align-items-center mb-0 justify-content-center"), width=2),
            dbc.Col(
                dcc.Dropdown(
                    id='dropdown',
                    options=selection_items,
                    placeholder="Select a city...",
                    className="dbc",
                    value="London"
                ),
                width=True
            )
        ], className="mb-3 align-items-center g-3"),

        # Row 2: Year
        dbc.Row([
            dbc.Col(html.Label("Select Year:", className="d-flex align-items-center mb-0 justify-content-center"), width=2),
            dbc.Col(
                dcc.Slider(
                    id='slider',
                    min=1850,
                    max=2030,
                    step=1,
                    included=True,
                    marks=slider_marks,
                    tooltip={"placement": "bottom", "always_visible": True},
                    value=2000
                ),
                width=True,
            ),
        ], className="mb-2 align-items-center g-3"),

        # Row 3: Buttons below slider
        dbc.Row([
            dbc.Col(html.Label("", className="d-flex align-items-center mb-0"), width=2),
            dbc.Col(dbc.Button("Prev. Year", id="year-backward-button", n_clicks=0, color="primary"), width="auto"),
            dbc.Col(html.P("Change incrementally to see new additions to the system each year",className="d-flex align-items-center justify-content-center mb-0"), width=True),
            dbc.Col(dbc.Button("Next Year", id="year-forward-button", n_clicks=0, color="primary"), width="auto")
        ], className="align-items-center")
    ]),
    className="h-100"
)

stats_card = dbc.Card(
    [
        dbc.CardBody([
            html.P("Snapshot Stats", className="text-center text-muted mb-3"),
            dbc.Row(id='count', children=[
                dbc.Col(html.P("Select city and year for stats.", className="text-center text-muted"), sm=12)
            ])
        ])
    ], className="h-100"
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


app.layout = html.Div([
    navbar,
    dbc.Container([
        # Title Section
        dbc.Row([
            dbc.Col([
                html.H4("Visualise the evolution of urban rail systems around the world",
                        className="text-center"),
                html.P(["Note: Data quality may vary between cities due to historical data availability.",
                       html.Br(),
                       "Cities with high proportions of missing opening/closure date information have been excluded."
                       ],
                       className="text-center text-muted mb-4"),
            ], width=12, lg=6, className="mx-auto") 
        ], className="py-1"),

        # Controls and Stats Row
        dbc.Row([
            dbc.Col(controls, width=12, lg=8, className="mb-1"),
            dbc.Col(stats_card, width=12, lg=2, className="mb-1"),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.P("Export Options", className="text-center text-muted mb-3"),
                        dbc.Button("Export as KML", id="export_kml_button", color="success", className="w-100 mb-2"),
                        dbc.Button("Export as GeoJSON", id="export_geojson_button", color="info", className="w-100"),
                        dcc.Download(id="download-kml"),
                        dcc.Download(id="download-geojson")
                    ])
                ], className="rounded-3 h-100")
            ], width=12, lg=2, className="mb-1")
        ], align="stretch",className="align-items-stretch mb-4"),

        # Visualization Row
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(id='plot', style={'height': '70vh'},figure=create_placeholder_figure("Loading map..."))
                    ])
                ], className="shadow-lg border-0 rounded-3"), md=6, className="mb-4"
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        dl.Map(
                            id='map',
                            center=[51.51, -0.13],
                            zoom=2,
                            style={'width': '100%', 'height': '70vh', 'borderRadius': '12px'},
                            children=[
                                dl.TileLayer(
                                    id='base-layer',
                                    url='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
                                ),
                                dl.LayerGroup(id='lines-layer', children=[]),
                                dl.LayerGroup(id='stations-layer', children=[]),
                            ]
                        )
                    ])
                ], className="shadow-lg border-0 rounded-3"), md=6, className="mb-4"
            )
        ]),
        #   dbc.Col(
        #         dbc.Card([
        #             dbc.CardBody([
        #                 dl.Map(
        #                   id='map',
        #                   center=[51.51, -0.13],
        #                   zoom=2,
        #                   style={'width': '100%', 'height': '70vh', 'borderRadius': '12px'},
        #                   children=[
        #                       dl.LayersControl(
        #                           [
        #                               dl.BaseLayer(
        #                                   dl.TileLayer(url='https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png'),
        #                                   name='Dark mode',
        #                                   checked=True
        #                               ),
        #                               dl.BaseLayer(
        #                                   dl.TileLayer(url='https://tiles.stadiamaps.com/tiles/alidade_smooth/{z}/{x}/{y}{r}.png'),
        #                                   name='Light mode',
        #                                   checked=False
        #                               ),
        #                               dl.Overlay(
        #                                   dl.LayerGroup(id='stations-layer'),
        #                                   name='Stations',
        #                                   checked=True
        #                               ),
        #                               dl.Overlay(
        #                                   dl.LayerGroup(id='lines-layer'),
        #                                   name='Lines',
        #                                   checked=True
        #                               ),
        #                           ]
        #                       )
        #                   ]
        #               )
        #             ])
        #         ], className="shadow-lg border-0 rounded-3 bg-dark-subtle"), md=6, className="mb-4"
        #     )
        # ]),

        # Summary Graph
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(
                          id='summarize', 
                          style={'height': '40vh'},
                          figure=create_placeholder_figure("Loading chart...")
                          )
                    ])
                ], className="shadow-lg border-0 rounded-3"),
                width=12
            )
        ], className="mb-4"),

        html.Hr(className="text-muted"),

        # Footer
        html.Footer([
            dbc.Row([
                dbc.Col(html.P("Created by: shaun.hoang@gmail.com",
                               className="text-center text-muted small"), width=6),
                dbc.Col(html.P([
                    "Data source: ",
                    html.A("Kaggle.com", href='https://www.kaggle.com/citylines/city-lines',
                           target="_blank", className="link-info")
                ], className="text-center text-muted small"), width=6)
            ])
        ], className="mt-3")
    ], fluid=True)
])


# Finally
if __name__ == "__main__":
    app.run(debug=True)