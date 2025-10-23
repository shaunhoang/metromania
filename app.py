## Project Metromania
#-by Shaun H. (shaun.hoang@gmail.com)

### Overview
# A dashboard that allows users to visualize the growth of chosen city's transit systems over time with a year slider. The user could also export them into KML files for use on other map applications like Google Maps and Earth

import os
import pandas as pd
import datetime
import requests
import plotly.express as px
import dash_leaflet as dl
from dash import dcc, html, Dash
from dash.dependencies import Input, Output, State
from simplekml import Kml
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


## Merge multiple datasets into STATIONS
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
                     'country','line_id','line_name','system_id','system_name']]

## Merge multiple datasets into TRACKS
tracks = pd.merge(tracks, track_lines_simpl,how='left',on='section_id')
tracks = pd.merge(tracks, lines_simpl, how='left', on='line_id')
tracks = pd.merge(tracks, cities_simpl, how='left', on='city_id')
tracks = pd.merge(tracks, systems_simpl, how='left', on='system_id')

# Define function to split coord from linestring object - two versions for different applications
def split_coord_lonlat(x):
    stripped_x = x.rstrip(')) ').lstrip(' MULTILINESTRING ((').strip() # strip non-numerical values from object 
    coord_list = []
    for point in stripped_x.split(','):
        coord = point.split(' ')                 # split into lon-lat 
        coord = [float(x.strip()) for x in coord]             # turn to float
        coord_list.append(coord)
    return coord_list

def split_coord_latlon(x):
    stripped_x = x.rstrip(')) ').lstrip(' MULTILINESTRING ((').strip() # strip non-numerical values from object 
    coord_list = []
    for point in stripped_x.split(','):
        coord = point.split(' ')                 # split into lon-lat 
        coord[0],coord[1] = coord[1],coord[0]     # swap to lat-lon
        coord = [float(x.strip()) for x in coord]             # turn to float
        coord_list.append(coord)
    return coord_list

# Split 'geometry' for each row into 'linestring' a list of coordinates to draw track lines
tracks['linestring_latlon'] = tracks.geometry.apply(split_coord_latlon)
tracks['linestring_lonlat'] = tracks.geometry.apply(split_coord_lonlat)

# Reorder columns
tracks = tracks[['section_id','geometry','linestring_latlon','linestring_lonlat','opening','closure',
                 'length','line_id','line_name','line_color',
                 'system_id','system_name','city_id','city','country']]


## Fill in NA and other data cleaning


stations['station_name'] = stations['station_name'].fillna('N.A.')
tracks['line_color'] = tracks['line_color'].fillna('#000000')
stations['closure'] = stations['closure'].fillna(999999)
tracks['closure'] = tracks['closure'].fillna(999999)
stations['line_id'] = stations['line_id'].fillna(0)
tracks['line_id'] = tracks['line_id'].fillna(0)
stations['line_name'] = stations['line_name'].fillna('N.A.')
tracks['line_name'] = tracks['line_name'].fillna('N.A.')


# Consideration for opening:
# - Stations: contains 73 'NULL' values, 1633 '0' values, and 42 '999999' values => 0
# - Tracks: contains 21 'NULL' values, 2937 '0' values, and 17 '999999' values => 0

stations['opening'] = stations['opening'].fillna(0)
tracks['opening'] = tracks['opening'].fillna(0)
stations.loc[stations.opening>2040, 'opening'] = 0
tracks.loc[tracks.opening>2040, 'opening'] = 0

# I'm just having fun here, trying to determine which city has the least stations without clean opening years

def wonk(x):
    if x==0 :
        wonk_or_good='wonk'
    else:
        wonk_or_good='good'
    return wonk_or_good

stations['wonk'] = stations.opening.apply(lambda x: wonk(x)) # add 'wonk' for opening==0
pivot_st = pd.pivot_table(stations,values='station_name',index='city',columns=['wonk'],aggfunc=len)
pivot_st = pivot_st.fillna(0)
pivot_st['wonkiness_st'] = (pivot_st['wonk']) / (pivot_st['good'] + pivot_st['wonk'])

tracks['wonk'] = tracks.opening.apply(lambda x: wonk(x)) # add 'wonk' for opening==0
pivot_tr = pd.pivot_table(tracks,values='section_id',index='city',columns=['wonk'],aggfunc=len)
pivot_tr = pivot_tr.fillna(0)
pivot_tr['wonkiness_tr'] = (pivot_tr['wonk']) / (pivot_tr['good'] + pivot_tr['wonk'])

# Will only show cities with wonk_score < median and more than 110 stations

wonk_table = pd.merge(pivot_st,pivot_tr,on='city')
wonk_table['wonk_score'] = (wonk_table['wonkiness_st'] + wonk_table['wonkiness_tr']) / (wonk_table['good_x'] + wonk_table['good_y'] + wonk_table['wonk_x'] + wonk_table['wonk_y'])
wonk_table = wonk_table[(wonk_table.wonk_score < wonk_table.wonk_score.median())
                       & ((wonk_table.good_x+wonk_table.wonk_x)>=110)]

wonk_table.columns.name = None              
wonk_table = wonk_table.reset_index()  
cities_list = wonk_table.city.tolist()


cities_list_other = stations.city.unique().tolist()
for x in cities_list:
  cities_list_other.remove(x)


# ---------------------------------
######## 2. Actually create mapping and plotting functions that take cities and year inputs

app = Dash(__name__)
server = app.server

# Get current year
currentDateTime = datetime.datetime.now()
currentDate = currentDateTime.date()
currentYear = float(currentDate.strftime("%Y"))

# Get geocoords from city input, to help center the plot and map
api_key = os.environ.get('GEO_API_KEY')
if not api_key:
    print("ERROR: GEO_API_KEY environment variable not set.")

@app.callback(Output('map','viewport'),[Input('dropdown','value')])    
def get_geocode(city):
    url = f'http://api.positionstack.com/v1/forward?access_key={api_key}&query={city}&limit=1'  
    response = requests.get(url)    
    geocode_data = requests.get(url).json()      
    lat = geocode_data['data'][0]['latitude']
    lon = geocode_data['data'][0]['longitude']
    lat = float(lat)
    lon = float(lon)
    viewport = {'center':[lat,lon],'zoom':12}
    return viewport


# Plot it function


@app.callback(Output('plot','figure'),[Input('dropdown','value'),Input('slider','value')])
def plot_it(city,year):
    
    my_stations = stations[(stations.city == city.title()) 
                           & (stations.opening <= year) 
                           & (stations.closure > year)]
    
    my_tracks = tracks[(tracks.city == city.title()) 
                       & (tracks.opening <= year) 
                       & (tracks.closure > year)]
    
    
    # Tracks: Extract linestring coords into lists, combine into a plottable df    
    long=[]
    lat=[]
    line_color=[]
    for sect in range(len(my_tracks)):
        linesegment = my_tracks.linestring_lonlat.iloc[sect]
        for point in linesegment:
            long.append(point[0])
            lat.append(point[1])
            line_color.append(my_tracks.line_color.iloc[sect])
    plot = pd.DataFrame({'x':long,
                         'y':lat,
                         'z':line_color})
    plot['x'] = plot['x'].astype(float)
    plot['y'] = plot['y'].astype(float)

    fig = px.scatter(plot, 
                     x="x", 
                     y="y" , 
                     color="z",
                     template="simple_white",
                    width=800, height=800)
    
    fig.update_yaxes(title_text="",showgrid=False,
                     showline=False,mirror=True, scaleanchor = "x",scaleratio = 1,
                     showticklabels=False,ticks='',automargin=True)
    fig.update_xaxes(title_text="",showgrid=False,
                     showline=False,mirror=True,
                     showticklabels=False,ticks='',automargin=True)
    fig.update_layout(showlegend=False,
                     autosize=False)
    
    return fig     # return plotly graph


# Map it function

@app.callback(Output('map','children'),[Input('dropdown','value'),Input('slider','value')])
def map_it(city,year):
    my_stations = stations[(stations.city == city) 
                           & (stations.opening <= year) 
                           & (stations.closure > year)]
    
    my_tracks = tracks[(tracks.city == city) 
                       & (tracks.opening <= year) 
                       & (tracks.closure > year)]
    
    url1 = 'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png'
    url2 = 'https://www.ign.es/wmts/mapa-raster?request=getTile&layer=MTN&TileMatrixSet=GoogleMapsCompatible&TileMatrix={z}&TileCol={x}&TileRow={y}&format=image/jpeg'
    attribution = '&copy; <a href="https://stadiamaps.com/">Stadia Maps</a> '

    markers = []
    for i in range(len(my_stations)):
        latlon = my_stations[['latitude', 'longitude']]
        latlonlist = latlon.values.tolist()        
        marker = dl.Marker(position=latlonlist[i],title=my_stations.station_name.iloc[i])
        markers.append(marker)
      
    lines = []
    for i in range(len(my_tracks)):
        linesegment = my_tracks.linestring_latlon.iloc[i]
        linecolor = my_tracks.line_color.iloc[i]
        line = dl.Polyline(positions=linesegment,color=linecolor)
        lines.append(line) 
        
    my_map = dl.LayersControl(
            [
                dl.BaseLayer(
                    dl.TileLayer(url=url1, maxZoom=20, attribution=attribution),
                    name='Dark mode',
                    checked=True
                ),
                dl.BaseLayer(
                    dl.TileLayer(opacity=0.5),
                    name="Light mode",
                    checked=False
                ),
                dl.BaseLayer(
                    dl.TileLayer(opacity=0.5,url=url2, maxZoom=20, attribution=attribution),
                    name='Retro mode',
                    checked=False
                ),
            ] + 
            [
                dl.Overlay(dl.LayerGroup(markers), name="Stations", checked=True),
                dl.Overlay(dl.LayerGroup(lines), name="Lines", checked=True)
            ]
        )
    return my_map


# Count it function - Snapshot of number of stations and track length as of a certain year

@app.callback(Output('count','children'),[Input('dropdown','value'),Input('slider','value')])
def count_it(city,year):
    my_stations = stations[(stations.city == city) 
                           & (stations.opening <= year) 
                           & (stations.closure > year)]
    my_tracks = tracks[(tracks.city == city) 
                       & (tracks.opening <= year) 
                       & (tracks.closure > year)]
    track_length_km = my_tracks.length.sum()/1000
    num_stations = len(my_stations)
    count_it_result = f'{city}\'s transit system in {year} had {num_stations} stations and {track_length_km} km total track length'
    return count_it_result


# Summarize it function - Evolution


@app.callback(Output('summarize','figure'),[Input('dropdown','value'),Input('slider','value')])
def summarize_it(city,year):
    my_stations = stations[(stations.city == city.title())]
    my_tracks = tracks[(tracks.city == city.title())]
    
    joint_df = pd.concat([my_tracks.opening,my_stations.opening])
    
    min_year = int(sorted(joint_df.unique(),reverse=False)[1])
    max_year = int(sorted(joint_df.unique(),reverse=True)[0])
    
    data = []
    for y in range(min_year, max_year):
        d = {'year': y,
             'track_length' : my_tracks[my_tracks.opening <= y].length.sum()/1000,
             'stations_num' : len(my_stations[my_stations.opening <= y])}
        data.append(d)
    dataset = pd.DataFrame(data)
    dataset.stations_num = dataset.stations_num.astype(float)
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=dataset.year, y=dataset.stations_num, name="stations"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=dataset.year, y=dataset.track_length, name="tracks"),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Number of stations", secondary_y=False,showgrid=False,zeroline=False) #Prim
    fig.update_yaxes(title_text="Track length", secondary_y=True,showgrid=False,zeroline=False) #Sec    
    
    fig.add_vline(x=year, line_width=2, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title={
            'text': "System growth over time",
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
    
    return fig         # return plotly graph


# Export it function - Into 2 KML files

@app.callback(
    [Output("download-kml-st", "data"),
     Output("download-kml-tr", "data")],
    [State('dropdown','value'),
     State('slider','value')],
    Input("export_button", "n_clicks"),
    prevent_initial_call=True,
)
def export_it(city,year,*args):
    my_stations = stations[(stations.city == city.title()) 
                            & (stations.opening <= year) 
                            & (stations.closure > year)]
    my_tracks = tracks[(tracks.city == city.title()) 
                        & (tracks.opening <= year) 
                        & (tracks.closure > year)]


    # --- Stations (in memory) ---
    kml_st = Kml(name='stations')
    list_st=[]
    for i in range(len(my_stations)):
        d = [my_stations.station_name.iloc[i],f'{my_stations.opening.iloc[i]:g}', my_stations.line_name.iloc[i],
             my_stations.latitude.iloc[i],my_stations.longitude.iloc[i]]
        list_st.append(d)

    for row in list_st:
        kml_st.newpoint(name=row[0], description=row[2],
                         coords=[(row[4], row[3])]) 

    # Create data dictionary for download
    station_data = dict(
        content=kml_st.kml(), 
        filename=f"stations_{city}_{year:g}.kml"
    )


    # --- Tracks (in memory) ---
    kml_tr = Kml(name='tracks')      
    list_tr=[]
    for i in range(len(my_tracks)):
        d = [my_tracks.line_name.iloc[i], my_tracks.linestring_lonlat.iloc[i],f'{my_tracks.opening.iloc[i]:g}']        
        list_tr.append(d)   

    for row in list_tr:
        kml_tr.newlinestring(name=row[0],description=row[2],coords=row[1])

    # Create data dictionary for download
    track_data = dict(
        content=kml_tr.kml(),
        filename=f"tracks_{city}_{year:g}.kml"
    )

    # Return the data dicts to the dcc.Download components
    return [station_data, track_data]

# ---------------------------------
######## 3. Create Dash with dropdown for Cities, and Year slider (with callbacks)


selection_items = []

for i in range(len(cities_list)):
    dict = {'label': f'{cities_list[i]}',
            'value':cities_list[i]}
    selection_items.append(dict)
for i in range(len(cities_list_other)):
    dict = {'label': f'{cities_list_other[i]}     - Note: Missing data',
            'value':cities_list_other[i]}
    selection_items.append(dict)
      
dropdown = dcc.Dropdown(id='dropdown', 
                         options=selection_items)

slider_marks = {}
for i in range(1850,2040,10):
    slider_marks[i] =  {'label': f'{i:g}'}

slider = dcc.Slider(id='slider',
                    min=1850,
                    max=2040,
                    step=1,
                    included = False,
                    marks=slider_marks,
                    tooltip={"placement": "bottom", "always_visible": True})

app.layout = html.Div(children=[
    html.H1(
        children=['Welcome to Metromania!'],
        style={'textAlign': 'center',}
        ),
    
    html.Plaintext(
        children=['Happy you are here, taking your first steps towards becoming a true metro historian'],
        style={'textAlign': 'center',}
        ),
    
    html.Div(children=[
        html.H3(
              children=['Let\'s get started... choose your city:'],
              style={'textAlign': 'center'},
        ),
        
        html.Br(),
        
        html.Div(children=[
            dropdown
        ],style={'width': '60%', 'margin': "auto", "display": "block",'justify': 'center'})
        ,
        html.Br(),
        
        html.H3(
            children=['Select your favourite year:'],                
            style={'textAlign': 'center'}
            ),
        
        html.Br(),
        slider,        
    ],style={'width': '80%', 'margin': "auto", "display": "block",'justify': 'center'}),
    
    html.Div(children=[  
        html.Div([
            dcc.Graph(id='plot')
        ],style={'width': '100%','display': 'flex','justify-content': 'center','align-items': 'center'}),
    ],style={'width': '80%', 'margin': "auto", "display": "block"}),

    html.Br(),
    html.Hr(),

    html.Div(children=[  
        html.H2(children=['Did you know...'],style={'display': 'flex','justify-content': 'center'}),
        html.Plaintext(id='count',style={'display': 'flex','justify-content': 'center'}),
    ],style={'width': '80%', 'margin': "auto", "display": "block"}), 
    
    html.Div(children=[
        dcc.Graph(id='summarize'),
    ],style={'width': '80%', 'margin': "auto", "display": "block"}
            ),
    
    html.H2(
        children=[f'Check it out on a map!'],
        style={'textAlign': 'center',}
        ),
    
    html.Div(children=[
        dl.Map(id='map')
    ],style={'width': '80%', 'height': '75vh', 'margin': "auto", "display": "block"}),
    
    html.Br(),
    
    html.Div(children=[
        html.Plaintext(children=['Export this map into (2) KML files \nto explore further in Google Earth'],style={'textAlign': 'center'}),
    ], style={'width': '100%','display': 'flex','justify-content': 'center','align-items': 'center'}
            ),
    
    html.Div(children=[
        html.Button('Export to KML', id='export_button',n_clicks=0),
        dcc.Download(id="download-kml-st"),
        dcc.Download(id="download-kml-tr")
    ], style={'width': '100%','display': 'flex','justify-content': 'center','align-items': 'center'}
            ),
    
    html.Br(),
    html.Hr(),
    
    html.Plaintext(children=['Created by: shaun.hoang@gmail.com'],
                   style={'textAlign': 'center'}),
    html.Plaintext(children=[
        'Data source:',
        html.A("Kaggle.com", href='https://www.kaggle.com/citylines/city-lines', target="_blank")
    ],style={'textAlign': 'center'})    
], style={'align-items': 'center','justify-content': 'center'})   

# Finally

if __name__ == "__main__":
    app.run_server(debug=True)