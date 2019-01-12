# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 09:54:32 2018

@author: Alexandros Spiliotis
"""

# Folium

import folium

# Matplotlib

import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Numpy

import numpy as np

# Pandas

import pandas as pd

# Requests

import requests

# Seaborn

import seaborn as sns

#sys
import sys

# Geocoder Timed Out, for when Geocoder unexpectedly times out after importing the 632th of 633 addresses
from geopy.exc import GeocoderTimedOut
# Nominatim
from geopy.geocoders import Nominatim  # module to convert an address into latitude and longitude values


# Geopy
# Matplotlib and associated plotting modules
# K-Means Clustering
# Standard Scaler


def val_to_num(df):
    values = []
    for val in df.Value:
        try:
            values.append(float(val.replace(',', '')))
        except ValueError:
            values.append(float(df.Value.max().replace(',', '')))

    return values


def enumerate_years(df):
    # Collect all survey periods

    year_values = df.Year.unique()

    # Survey Periods dictionary

    df_survey = pd.DataFrame(year_values)
    df_survey.rename(columns={0: 'Survey Period'}, inplace=True)

    # Replace Categorical time values with enumeration

    for i, year in enumerate(year_values):
        df.replace(year, i, inplace=True)

    return df, df_survey


def create_ward_df(df):
    print('Initializing Wards Dataframe...')
    df_ward_ = {}
    names = (df.Ward_name + ', ' + df.Borough).unique()
    for i,name in enumerate(names):
        df_ward_[name] = df.loc[(df.Ward_name == name.split(', ')[0]) & (df.Borough == name.split(', ')[1])]
        df_ward_[name].reset_index(inplace=True, drop=True)
        print(f'{i / len(names) * 100:.2f} % completed', end='\r')

    sys.stdout.flush()
    print(f'100.0 % completed!', end='\n')
    return df_ward_, names


# With the following function, we will obtain all ward coordinates, except for the ones for which geocoder won't be able
# to find.
# When such an exception happens, the ward name will pass to an array ("exceptions"), which will be used to monitor which
# wards weren't accounted for.
# Due to the abundance of wards, even if we lose a few tens of them, the rest of them should be enough to draw a valid
# conclusion.



def geolocation(neighborhoods):
    coords = []
    exceptions = []
    print('Initializing Neighborhood Geolocation...')
    for names in neighborhoods:
        try:
            address = names + ', London, UK'
            geolocator = Nominatim(user_agent="my-application")
            location = geolocator.geocode(address, timeout=None)
            latitude = location.latitude
            longitude = location.longitude

        except (AttributeError, GeocoderTimedOut):
            exceptions.append(names)
            pass

        coords.append([latitude, longitude])
        sys.stdout.flush()
        print(f'{len(coords)/len(neighborhoods)*100:.2f} % completed', end='\r')
    print(f'100.0 % completed!', end='\n')
    return coords, exceptions


def drop_exceptions(df, coords, exceptions):
    index_to_drop = []

    # Find the positions of the missing wards

    for wards in exceptions.values:
        wards = str(wards)
        index_to_drop.append(df.loc[(df.Ward_name == wards[0]) & (df.Borough == wards[1])].index.values)

    index_to_drop = np.concatenate(index_to_drop)

    # ...and drop them from the dataframe:

    df.drop(index_to_drop, inplace=True)
    df.reset_index(inplace=True, drop=True)

    latitude = []
    longitude = []

    for i, crd in enumerate(coords.values):
        if i in index_to_drop:
            pass
        else:
            latitude.append(crd[0])
            longitude.append(crd[1])

    df['Latitude'] = latitude
    df['Longitude'] = longitude

    return df


def drop_unwanted_cols(df):
    df.drop(columns=['Code', 'Measure'], inplace=True)

    ward_borough_names_reshaped = (df.Ward_name + ', ' + df.Borough).unique()

    df['Neighborhood'] = ward_borough_names_reshaped

    df.drop(columns=['Ward_name', 'Borough'], inplace=True)

    cols = ['Neighborhood', 'Value', 'Latitude', 'Longitude', 'Year']

    df = df[cols]
    print('Columns Dropped!')
    return df


def add_distance(df, addr):
    address = addr
    geolocator = Nominatim(user_agent="my-application")
    trg_location = geolocator.geocode(address, timeout=5)
    trg_latitude = trg_location.latitude
    trg_longitude = trg_location.longitude

    target = []
    neigh_locations = []

    for neighs in zip(df.Latitude, df.Longitude):
        neigh_locations.append(neighs)
    #    print(np.array(neigh_crds))
    #    cdist(np.array(neigh_crds),target)

    for i in range(0, len(neigh_locations)):
        target.append([trg_latitude, trg_longitude])

    distances = []
    neigh_locations = np.array(neigh_locations)

    # approximate radius of earth in km
    R = 6373.0

    from math import sin, cos, sqrt, atan2, radians

    for trg, neigh in zip(target, neigh_locations):
        lat1 = radians(neigh[0])
        lon1 = radians(neigh[1])
        lat2 = radians(trg[0])
        lon2 = radians(trg[1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat / 2) ** 2 + cos(lat2) * cos(lat1) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distances.append(R * c)

    for i in range(0, len(distances)):
        distances[i] = round(distances[i], 2)

    return distances


def london_map(df):
    address = 'London, United Kingdom'
    geolocator = Nominatim(user_agent="my-application")
    Lnd_location = geolocator.geocode(address, timeout=5)
    Lnd_latitude = Lnd_location.latitude
    Lnd_longitude = Lnd_location.longitude

    lndloc = [Lnd_latitude, Lnd_longitude]
    map_Lnd = folium.Map(location=lndloc, zoom_start=10)

    for lat, lng, name in zip(df.Latitude,
                              df.Longitude,
                              df.Neighborhood
                              ):
        label = '{}'.format(name)  # ,lat,lng)

        label = folium.Popup(label, parse_html=True)
        folium.CircleMarker(
            [lat, lng],
            radius=5,
            popup=label,
            color='blue',
            fill=True,
            # fill_color='neighborhoods.borough',
            fill_opacity=0.7,
        ).add_to(map_Lnd)

    return map_Lnd


def clusters_map(df):
    from matplotlib.colors import rgb2hex

    address = 'London, United Kingdom'
    geolocator = Nominatim(user_agent="my-application")
    Lnd_location = geolocator.geocode(address, timeout=5)
    Lnd_latitude = Lnd_location.latitude
    Lnd_longitude = Lnd_location.longitude
    lndloc = [Lnd_latitude, Lnd_longitude]

    map_clusters = folium.Map(location=lndloc, zoom_start=10)
    kclusters = len(df['Cluster Labels'].unique())
    x = np.arange(kclusters)
    ys = [i + x + (i * x) ** 2 for i in range(kclusters)]

    colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
    rainbow = [rgb2hex(i) for i in colors_array]
    # markers_colors=[]
    for lat, lon, poi, cluster in zip(df.Latitude,
                                      df.Longitude,
                                      df.Neighborhood,
                                      df['Cluster Labels']):
        label = folium.Popup(str(poi) + ', Cluster ' + str(cluster), parse_html=True)
        folium.CircleMarker(
            [lat, lon],
            radius=5,
            popup=label,
            color=rainbow[cluster],
            fill_color=rainbow[cluster],
            fill=True,
            fill_opacity=0.7).add_to(map_clusters)

    return map_clusters


def reshape_onehot(df_onehot, df_venues):

    # This is an abomination of a function, but still the best way that I could figure out to manually categorize venue
    # categories so that we have 8 big categories instead of ~500 small.

    print('Reshaping One-hot Dataframe...')

    my_list = df_venues['Venue Category'].unique().tolist()

    stores = []
    restaurants = []
    places = []
    shops = []
    salons = []
    bars = []
    clubs = []
    gyms = []
    art = []
    joints = []
    groceries = []
    stations = []
    markets = []
    spots = []
    food = []
    misc = []
    theaters = []
    courts = []
    museums = []
    parks = []
    cafes = []
    airports = []

    misc.append(['Photography Studio', 'Recording Studio', 'Film Studio'])

    for rest in misc:
        try:
            my_list.remove(rest)
        except:
            pass

    for i, val in enumerate(my_list):
        if ('Store' in val) or ('store' in val):
            stores.append(val)

        if 'Restaurant' in val:
            restaurants.append(val)

        if 'Place' in val:
            places.append(val)

        if ('Shop' in val) or ('shop' in val):
            shops.append(val)

        if ('Salon' in val):
            salons.append(val)

        if ('club' in val) or ('Club' in val):
            clubs.append(val)

        if ('Bar' in val) or ('Pub' in val):
            bars.append(val)

        if ('Gym' in val) or ('Studio' in val) or ('Recreation Center' in val):
            gyms.append(val)

        if ('Grocery' in val):
            groceries.append(val)

        if ('Gallery' in val) or ('Art' in val):
            art.append(val)

        if ('Market' in val) or ('market' in val):
            markets.append(val)

        if ('Joint' in val):
            joints.append(val)

        if ('Spot' in val):
            spots.append(val)

        if ('Station' in val):
            stations.append(val)

        if ('Food' in val):
            food.append(val)

        if ('Theater' in val):
            theaters.append(val)

        if ('Museum' in val):
            museums.append(val)

        if ('Court' in val) or ('Ground' in val) or ('Course' in val) or ('Stadium' in val) or ('Field' in val):
            courts.append(val)

        if ('Garden' in val) or ('Park' in val) or ('Plaza' in val) or ('Playground') in val:
            parks.append(val)

        if ('Airport' in val):
            airports.append(val)

    bars_to_append = ['Beer Garden', 'Brewery', 'Lounge']
    rest_to_append = ['Gastropub', 'Steakhouse', 'Churrascaria', 'Diner', 'Noodle House']
    cafes = ['Café', 'Tea Room', 'Creperie', 'Bistro', 'Gaming Cafe']
    hotels = ['Hotel', 'Bed & Breakfast', 'Hostel']
    pharmacies = ['Pharmacy']
    stores_to_append = ['Bakery', 'Deli / Bodega']
    gyms_to_append = ['Spa', 'Pool', 'Athletics & Sports']

    for bar in bars_to_append:
        bars.append(bar)
    for gym in gyms_to_append:
        gyms.append(gym)

    for store in stores_to_append:
        stores.append(store)

    for rest in rest_to_append:
        restaurants.append(rest)

    restaurants = restaurants
    food = joints + spots + food + places
    art_venues = art + theaters + museums
    bars = bars + clubs
    stores = stores + groceries + shops + markets + pharmacies + salons
    outdoors = parks + courts
    for rest in (restaurants
                 + stores

                 + bars
                 + gyms
                 + stations
                 + art_venues
                 + food
                 + outdoors
                 + cafes
                 + airports
                 + hotels
    ):
        try:
            my_list.remove(rest)
        except:
            pass

    df_onehot_restaurants = df_onehot[restaurants].sum(1)
    df_onehot_stores = df_onehot[stores].sum(1)
    df_onehot_bars = df_onehot[bars].sum(1)
    df_onehot_gyms = df_onehot[gyms].sum(1)
    df_onehot_art_venues = df_onehot[art_venues].sum(1)
    df_onehot_food = df_onehot[food].sum(1)
    df_onehot_outdoors = df_onehot[outdoors].sum(1)
    df_onehot_cafes = df_onehot[cafes].sum(1)
    # df_onehot_hotels     = df_onehot[hotels].sum(1)

    df_onehot_new = pd.DataFrame(df_onehot.Neighborhood)
    df_onehot_new['Nightlife'] = df_onehot_bars.astype('uint8')
    df_onehot_new['Restaurants'] = df_onehot_restaurants.astype('uint8')
    df_onehot_new['Cafes'] = df_onehot_cafes.astype('uint8')
    df_onehot_new['Stores'] = df_onehot_stores.astype('uint8')
    df_onehot_new['Gyms'] = df_onehot_gyms.astype('uint8')
    df_onehot_new['Art Venues'] = df_onehot_art_venues.astype('uint8')
    df_onehot_new['Food Stalls'] = df_onehot_food.astype('uint8')
    df_onehot_new['Outdoors Activities'] = df_onehot_outdoors.astype('uint8')

    print('Done!')
    return df_onehot_new


def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']

    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    CLIENT_ID = 'HWG4GRWAET5CTID3P3CGEWPIMQJCZS5FEMQQWCDL5FCCSC5X'  # your Foursquare ID
    CLIENT_SECRET = '53CLPTEMYTZJZUKGRDBVD40WSJ54ZRILJ4NF43CU33HDBSH4'  # your Foursquare Secret
    VERSION = '20180605'

    LIMIT = 100

    venues_list = []
    for i,(name, lat, lng) in enumerate(zip(names, latitudes, longitudes)):
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID,
            CLIENT_SECRET,
            VERSION,
            lat,
            lng,
            radius,
            LIMIT)

        # make the GET request
        try:
            results = requests.get(url).json()["response"]['groups'][0]['items']
        except KeyError:
            results = []

        # return only relevant information for each nearby venue
        venues_list.append([(
            name,
            lat,
            lng,
            v['venue']['name'],
            v['venue']['location']['lat'],
            v['venue']['location']['lng'],
            v['venue']['categories'][0]['name']) for v in results])
        print(f'{i / len(names) * 100:.2f} % completed', end='\r')
    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood',
                             'Neighborhood Latitude',
                             'Neighborhood Longitude',
                             'Venue',
                             'Venue Latitude',
                             'Venue Longitude',
                             'Venue Category']
    print(f'100.0 % completed')
    return (nearby_venues)


def display_venue_numbers(df):
    print(' \nThere are {} unique venue categories in London.'.format(len(df['Venue Category'].unique())))

    return df['Venue Category'].value_counts().head(10)


def mean_vals_barplot(df, size, method='mean'):
    sns.set(style="white", context="talk")
    rs = np.random.RandomState(8)

    # Set up the matplotlib figure

    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=size, sharex=True)

    if method == 'median':

        fun = df.groupby('Cluster Labels').median()

    else:

        fun = df.groupby('Cluster Labels').mean()

    mean_value_cluster = fun.Value
    mean_distance_cluster = fun['Distance from Target']
    mean_restaurants_cluster = fun.Restaurants
    mean_bars_cluster = fun.Bars
    mean_stores_cluster = fun.Stores
    # =============================================================================
    #     mean_value_cluster          = df.groupby('Cluster Labels').fun.Value
    #     mean_distance_cluster       = df.groupby('Cluster Labels').fun['Distance from Target']
    #     mean_restaurants_cluster    = df.groupby('Cluster Labels').fun.Restaurants
    #     mean_bars_cluster           = df.groupby('Cluster Labels').fun.Bars
    #     mean_stores_cluster         = df.groupby('Cluster Labels').fun.Stores
    # =============================================================================

    x = np.array(range(0, len(mean_value_cluster)))
    y1 = mean_value_cluster
    sns.barplot(x=x, y=y1, palette="rocket", ax=ax1)
    ax1.axhline(0, color="k", clip_on=False)
    ax1.set_ylabel("Mean\n house value")

    y2 = mean_distance_cluster
    sns.barplot(x=x, y=y2, palette="vlag", ax=ax2)
    ax2.axhline(0, color="k", clip_on=False)
    ax2.set_ylabel("Mean\ndistance")
    ax2.set_ylim(0, 40)

    y3 = mean_restaurants_cluster
    sns.barplot(x=x, y=y3, palette="deep", ax=ax3)
    ax3.axhline(0, color="k", clip_on=False)
    ax3.set_ylabel("Mean\n# Restaurants")

    y4 = mean_bars_cluster
    sns.barplot(x=x, y=y4, palette="deep", ax=ax4)
    ax4.axhline(0, color="k", clip_on=False)
    ax4.set_ylabel("Mean\n# Bars")

    y5 = mean_stores_cluster
    sns.barplot(x=x, y=y5, palette="deep", ax=ax5)
    ax5.axhline(0, color="k", clip_on=False)
    ax5.set_ylabel("Mean\n# Stores")

    y4 = mean_bars_cluster
    sns.barplot(x=x, y=y4, palette="deep", ax=ax4)
    ax4.axhline(0, color="k", clip_on=False)
    ax4.set_ylabel("Mean\n# Stores")

    # Finalize the plot
    sns.despine(bottom=True)
    plt.setp(f.axes, yticks=[])
    plt.tight_layout(h_pad=2)

    return None
