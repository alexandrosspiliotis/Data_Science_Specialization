# Numpy

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import london_clustering_functions as london


class LondonClustering:

    def __init__(self, method=0, target_loc='Hammersmith Hospital, UK', n_clusters=8,
                 weights=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]):
        print('Initializing Parameters...')
        fileurl = 'land-registry-house-prices-ward.csv'
        self.df_house = pd.read_csv(fileurl)
        # self.venue_radius = venue_radius
        self.coordinates_method = method
        self.clusters_n = n_clusters
        self.weights = weights
        self.target = target_loc

        if method != 0 and method != 1:
            raise ValueError(
                'Parameter "geolocate" should be either 1 (geolocate neighborhoods and venues) or 0 (use csv).')

        if not isinstance(self.target, str):
            raise ValueError('Target Address should be a string.')

        if not (isinstance(n_clusters, int) and n_clusters > 0):
            raise ValueError('Number of Clusters should be a positive integer.')

        if len(self.weights) != 10:            raise ValueError(
            'Weights for all 10 parameters (restaurants, bars etc.) should be passed')

        for weight in self.weights:
            if not 1 >= weight > 0.01:
                raise ValueError('Weights should range from 0.01 to 1')

        print('Done!')

        self.house_data()
        self.create_venues()
        self.create_onehot()
        self.insert_location()
        self.create_grouped()
        self.clustering()
        self.elbow_analysis(40)
        print('Process Completed!')

    def house_data(self):

        print('Initializing Housing Values Dataframe...')
        df_house_med = self.df_house.loc[self.df_house['Measure'] == 'Median']

        # df_house_mean = df_house.loc[df_house['Measure'] == 'Mean']
        # df_house_sales = df_house.loc[df_house['Measure'] == 'Sales']

        self.df_house = df_house_med

        if self.df_house['Value'].dtypes != 'float64':
            vals = self.val_to_num()
            self.df_house['Value'] = vals

        self.df_house, self.survey_period = london.enumerate_years(self.df_house)

        print('Done!')

        self.df_house_ward, ward_borough_names = london.create_ward_df(self.df_house)

        self.df_house_curr = self.df_house.loc[self.df_house.Year == 88].reset_index(drop=True)

        if self.coordinates_method == 1:
            coords, exceptions = london.geolocation(ward_borough_names)

            np.savetxt("coordinates.csv", coords, delimiter=',')
            np.savetxt("exceptions.csv", exceptions, delimiter=';', fmt='%s')
        else:
            pass

        print('Neighborhood Coordinates Imported!')
        coords = pd.read_csv('coordinates.csv', header=None)
        exceptions = pd.read_csv('exceptions.csv', header=None)

        df_house_no_exceptions = london.drop_exceptions(self.df_house_curr, coords, exceptions)
        print('Exceptions Dropped!')
        df_house_no_exceptions = df_house_no_exceptions.loc[df_house_no_exceptions.Year == 88].reset_index(drop=True)
        self.df_house_cleaned = london.drop_unwanted_cols(df_house_no_exceptions)
        self.list_neigh = self.df_house_cleaned.Neighborhood

    def insert_location(self):
        self.df_house_cleaned['Distance from Target'] = london.add_distance(self.df_house_cleaned,
                                                                            self.target)

    def show_london_map(self):
        return london.london_map(self.df_house_cleaned)

    def create_venues(self):
        print('Importing Venues...')
        if self.coordinates_method == 1:
            df_venues = london.getNearbyVenues(
                self.df_house_cleaned.Neighborhood,
                self.df_house_cleaned.Latitude,
                self.df_house_cleaned.Longitude
            )

            df_venues.to_csv("venues.csv")

        df_venues = pd.read_csv('venues.csv')
        df_venues.drop(columns='Unnamed: 0', inplace=True)
        self.df_venues = df_venues
        print('Done!')
        # london.display_venue_numbers(self.df_venues)

    def val_to_num(self):
        values = []
        for val in self.df_house.Value:
            try:
                values.append(float(val.replace(',', '')))
            except ValueError:
                values.append(float(self.df_house.Value.max().replace(',', '')))
        return values

    def create_onehot(self):
        print('Initializing One-hot Dataframe...')
        self.df_onehot = pd.get_dummies(self.df_venues[['Venue Category']], prefix='', prefix_sep='')
        self.df_onehot['Neighborhood'] = self.df_venues['Neighborhood']

        # Let's find where the 'Neighborhood' column went..
        neigh_position = np.where(self.df_onehot.columns == 'Neighborhood')[0][0]

        # Rearrange the columns:

        fixed_columns = np.concatenate(
            [self.df_onehot.columns[neigh_position:]] + [self.df_onehot.columns[:neigh_position]])
        self.df_onehot = self.df_onehot[fixed_columns]
        print("Done!")
        self.df_onehot_reshaped = london.reshape_onehot(self.df_onehot, self.df_venues)

    def create_grouped(self):

        self.df_grouped = self.df_onehot_reshaped.groupby('Neighborhood').sum().reset_index()

        no_venues = \
            np.where(np.isin(self.df_house_cleaned.Neighborhood.values, self.df_grouped.Neighborhood.values) == False)[
                0].tolist()

        self.df_house_cleaned.drop(no_venues, inplace=True)
        self.df_house_cleaned.reset_index(inplace=True, drop=True)

        # self.df_grouped.drop(no_venues, inplace=True)
        # self.df_grouped.reset_index(inplace=True, drop=True)

        self.df_house_cleaned.sort_values(by='Neighborhood', inplace=True)
        self.df_house_cleaned.reset_index(drop=True, inplace=True)

        self.df_grouped.sort_values(by='Neighborhood', inplace=True)
        self.df_grouped.reset_index(drop=True, inplace=True)

        self.df_grouped['Value'] = self.df_house_cleaned['Value']
        self.df_grouped['Distance from Target'] = self.df_house_cleaned['Distance from Target']

    def clustering(self):

        print('Creating Clusters...')
        np.random.seed(9001)
        self.df_grouped_clustering = self.df_grouped.drop('Neighborhood', 1)

        cols = self.df_grouped_clustering.columns

        x = np.array(self.df_grouped_clustering)
        x = StandardScaler().fit_transform(x)

        self.df_grouped_clustering_unweighted = pd.DataFrame(x, columns=self.df_grouped_clustering.columns)

        self.x_weighted = np.array([x_obs * self.weights for x_obs in x])

        self.df_grouped_clustering = pd.DataFrame(self.x_weighted, columns=self.df_grouped_clustering.columns)

        kclusters = self.clusters_n
        self.k_means = KMeans(init='k-means++', n_clusters=kclusters, n_init=10, tol=10 ** -6).fit(
            self.x_weighted)

        # self.k_means = SpectralClustering(n_clusters=kclusters, affinity='nearest_neighbors',
        #                                   assign_labels='kmeans').fit(
        #     self.df_grouped_clustering)

        self.df_house_cleaned['Cluster Labels'] = self.k_means.labels_
        self.df_grouped['Cluster Labels'] = self.k_means.labels_

        self.df_merged = pd.merge(self.df_grouped, self.df_house_cleaned)

        self.clusternames = [self.df_merged.loc[self.df_merged['Cluster Labels'] == i].Neighborhood.index
                             for i in
                             range(0, kclusters)]
        print('Done!')

    def elbow_analysis(self, max_clusters):

        self.distortions = []
        self.K = range(1, max_clusters)

        for k in self.K:
            kmeans_elbow = KMeans(init='k-means++', n_clusters=k, n_init=10, tol=10 ** -6)
            kmeans_elbow.fit(self.x_weighted)
            self.distortions.append(kmeans_elbow.inertia_)

        #np.min(cdist(self.x_weighted, kmeans_elbow.cluster_centers_, 'euclidean'), axis=1)

    def show_elbow_analysis(self):
        import matplotlib.pyplot as plt

        # Plot the elbow
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(1,1,1)

        ax.plot(self.K, self.distortions, 'bx-')
        ax.set_xlabel('k')
        ax.set_ylabel('Distortion')
        ax.set_title('The Elbow Method showing the optimal k')


    def show_clusters_map(self):
        return london.clusters_map(self.df_merged)

    def show_barplots(self, size):
        return london.mean_vals_barplot(self.df_grouped, size, method='mean')

    def show_clusters_data(self, cluster):
        return self.df_merged.loc[self.clusternames[cluster]]

    def show_price_evolution(self, neigh, init=0, end=89, roll=1):

        import matplotlib.pyplot as plt

        surv_time_new = [' '.join(surv.split()[2:4]) for surv in self.survey_period.values.flatten()]

        plt.figure(figsize=(14, 14))
        plt.subplot(1, 1, 1)

        for ng in neigh:
            self.df_house_ward[ng].Year = surv_time_new

            time_trend = surv_time_new[init: end]
            x = pd.to_datetime(time_trend)
            y = self.df_house_ward[ng].Value[init:end].rolling(roll).mean()

            sns.lineplot(x=x, y=y, label=ng)

    def plot_data(self, param, kind='box', swarm=None):

        if kind == 'box':

            sns.boxplot(x="Cluster Labels", y=param, data=self.df_merged,
                        whis="range", palette="vlag")

            if swarm:
                sns.swarmplot(x="Cluster Labels", y=param, data=self.df_merged,
                              size=2, color=".3", linewidth=0)

            sns.despine(trim=True)

        elif kind == 'cat':
            sns.catplot(x="Cluster Labels", y=param, data=self.df_merged,
                        palette="vlag")
            sns.despine(trim=True)

        else:
            raise ValueError('The plot can be of kind "cat"(catplot) or "box"(boxplot)')

    def plot_multiple(self, params, swarm=False):

        import matplotlib.pyplot as plt
        plt.figure(figsize=(11, 11))

        for i, param in enumerate(params):
            plt.subplot(2, 2, i + 1)

            sns.boxplot(x="Cluster Labels", y=param, data=self.df_merged,
                        palette="vlag")

            if swarm:
                sns.swarmplot(x="Cluster Labels", y=param, data=self.df_merged,
                              size=2, color=".3", linewidth=0)

            sns.despine(trim=True)

    def show_heatmap(self):
        df_corr_spearman = self.df_grouped_clustering.corr(method='spearman')
        sns.heatmap(df_corr_spearman)