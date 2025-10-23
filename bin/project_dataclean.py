import pandas as pd 

##### 1 Preparation
# Read datasets into dataframes
cities = pd.read_csv("datasets/cities.csv")
stations = pd.read_csv("datasets/stations.csv")
tracks = pd.read_csv("datasets/tracks.csv")
lines = pd.read_csv("datasets/lines.csv")
track_lines = pd.read_csv("datasets/track_lines.csv")
station_lines = pd.read_csv("datasets/station_lines.csv")
systems = pd.read_csv("datasets/systems.csv")

##### 2 Joining datasets
### Trim the dataframes to key columns, prepared for merging

stations = stations.rename(columns={'id':'station_id','name':'station_name'})
tracks = tracks.rename(columns={'id':'section_id'})
cities_simpl = pd.DataFrame({'city_id':cities.id,'country':cities.country,'city':cities.name})
station_lines_simpl = pd.DataFrame({'station_id':station_lines.station_id,'line_id':station_lines.line_id})
lines_simpl = pd.DataFrame({'line_id':lines.id,'line_name':lines.name,'line_color':lines.color,'system_id':lines.system_id})
systems_simpl = pd.DataFrame({'system_id':systems.id,'system_name':systems.name})
track_lines_simpl = pd.DataFrame({'section_id':track_lines.section_id,'line_id':track_lines.line_id})

### Create new **stations** dataframe with more complete data from other datasets

# Merge multiple datasets into STATIONS
stations = pd.merge (stations, cities_simpl, how='left',on='city_id')
stations = pd.merge (stations, station_lines_simpl, how='left',on='station_id')
stations = pd.merge (stations, lines_simpl, how='left',on='line_id')
stations = pd.merge (stations, systems_simpl, how='left',on='system_id')

# Split 'geometry' into 'longitudes' and 'latitudes'
stations['longitude'] = stations['geometry'].apply(lambda x: x.split('POINT(')[1].split(' ')[0])
stations['latitude'] = stations['geometry'].apply(lambda x: x.split('POINT(')[1].split(' ')[1].split(')')[0])

# Reorder columns in STATIONS and clean up
stations = stations[['station_id','station_name','geometry','longitude',
                     'latitude','opening','closure','city_id','city',
                     'country','line_id','line_name','system_id','system_name']]
stations.head()
stations.head()

### Create new **tracks** dataframe with more complete data from other datasets

# Merge multiple datasets into TRACKS
tracks = pd.merge(tracks, track_lines_simpl,how='left',on='section_id')
tracks = pd.merge(tracks, lines_simpl, how='left', on='line_id')
tracks = pd.merge(tracks, cities_simpl, how='left', on='city_id')
tracks = pd.merge(tracks, systems_simpl, how='left', on='system_id')

# Define function to split coord from linestring object - two versions for different applications 
def split_coord_lonlat(x):
    stripped_x = x.rstrip(')').lstrip('LINESTRING(') # strip non-numerical values from object 
    coord_list = []
    for point in stripped_x.split(','):
        coord = point.split(' ')                 # split into lon-lat 
        coord = [float(x) for x in coord]             # turn to float
        coord_list.append(coord)
    return coord_list

def split_coord_latlon(x):
    stripped_x = x.rstrip(')').lstrip('LINESTRING(') # strip non-numerical values from object 
    coord_list = []
    for point in stripped_x.split(','):
        coord = point.split(' ')                 # split into lon-lat 
        coord[0],coord[1] = coord[1],coord[0]     # swap to lat-lon
        coord = [float(x) for x in coord]             # turn to float
        coord_list.append(coord)
    return coord_list

# Split 'geometry' for each row into 'linestring' a list of coordinates to draw track lines
tracks['linestring_latlon'] = tracks.geometry.apply(split_coord_latlon)
tracks['linestring_lonlat'] = tracks.geometry.apply(split_coord_lonlat)

# Reorder columns
tracks = tracks[['section_id','geometry','linestring_latlon','linestring_lonlat','opening','closure',
                 'length','line_id','line_name','line_color',
                 'system_id','system_name','city_id','city','country']]
tracks.head()
tracks.head()

##### 3 Data cleanup

stations['station_name'] = stations['station_name'].fillna('N.A.')
tracks['line_color'] = tracks['line_color'].fillna('#000000')

# Closing date: operational stations/tracks have both a value of 999999.0 and NaN. This needs to be cleaned up to all 999999
stations['closure'] = stations['closure'].fillna(999999)
tracks['closure'] = tracks['closure'].fillna(999999)

# Opening date to clean all missing values and '999999' values to 0
stations['station_name'] = stations['station_name'].fillna('N.A.')
tracks['line_color'] = tracks['line_color'].fillna('#000000')
stations['closure'] = stations['closure'].fillna(999999)
tracks['closure'] = tracks['closure'].fillna(999999)
stations['line_id'] = stations['line_id'].fillna(0)
tracks['line_id'] = tracks['line_id'].fillna(0)
stations['line_name'] = stations['line_name'].fillna('N.A.')
tracks['line_name'] = tracks['line_name'].fillna('N.A.')
stations['opening'] = stations['opening'].fillna(0)
tracks['opening'] = tracks['opening'].fillna(0)
stations['opening'] = stations['opening'].replace(999999,0)
tracks['opening'] = tracks['opening'].replace(999999,0)

# Export
stations.to_csv('stations_cmpd.csv',index=False)
tracks.to_csv('tracks_cmpd.csv',index=False)