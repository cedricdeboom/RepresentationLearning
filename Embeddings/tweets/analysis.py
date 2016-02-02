__author__ = 'cedricdeboom'

#Analyse uncompressed twitter csv files

import os
import csv
import operator
import matplotlib
import numpy as np
from sklearn.cluster import DBSCAN

matplotlib.use('Agg')
import matplotlib.pyplot as plt


TEN_MINUTES_MS = 600000
FIVE_MINUTES_MS = 300000
ONE_MINUTE_MS = 60000


def determine_popular_hashtags_csv_file(input_file_name, top=30):
    ht_dict = {}

    with open(input_file_name, 'rb') as csv_f:
        csv_reader = csv.reader(csv_f, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in csv_reader: #returns a row as a list [id, text, hashtags, timestamp, retweeted?]
            if row[4] == '1':
                continue  #ignore retweets
            hashtags = row[2].split(',')
            for ht in hashtags:
                if len(ht) == 0:
                    continue
                if ht in ht_dict:
                    ht_dict[ht] += 1
                else:
                    ht_dict[ht] = 1

    return sorted(ht_dict.items(), key=operator.itemgetter(1), reverse=True)[0:top]


def graph_popular_hashtags(input_file_name, top_hashtag_tuples, time_division=FIVE_MINUTES_MS):
    _, bare_file_name = os.path.split(input_file_name)

    graph_dict = {}
    for ht_tuple in top_hashtag_tuples:
        graph_dict[ht_tuple[0]] = [] #init dict

    start = True
    boundary_time = 0

    temp_dict = {}
    for ht_tuple in top_hashtag_tuples:
        temp_dict[ht_tuple[0]] = 0 #init temporary dict

    with open(input_file_name, 'rb') as csv_f:
        csv_reader = csv.reader(csv_f, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in csv_reader: #returns a row as a list [id, text, hashtags, timestamp, retweeted?]
            if start:
                boundary_time = int(row[3])
                start = False
            if row[4] == '1':
                continue  #ignore retweets

            hashtags = row[2].split(',')
            current_time = int(row[3])

            if current_time % time_division == 0 and current_time > boundary_time:
                #update dict
                for (key, value) in temp_dict.items():
                    graph_dict[key].append(value)
                #clean temporary dict
                temp_dict = {}
                for ht_tuple in top_hashtag_tuples:
                    temp_dict[ht_tuple[0]] = 0
                #update boundary time
                boundary_time = current_time

            for ht in hashtags:
                if ht in temp_dict.keys():
                    temp_dict[ht] += 1

    #update dict
    for (key, value) in temp_dict.items():
        graph_dict[key].append(value)

    #create graphs
    i = 0
    print 'Creating graphs ...'
    for (ht, number) in top_hashtag_tuples:
        vector = graph_dict[ht]
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(ht + " " + bare_file_name.replace('.csv', '') + " " + str(number))
        x = np.linspace(0, len(vector)-1, len(vector))
        plt.plot(x, vector, 'b-', label='0')
        plt.savefig('graphs_uncompressed/' + bare_file_name.replace('.csv', '.' + str(i) + '.png'))
        plt.close()
        i += 1


def create_cooccurrence_matrix(input_file_name, top_hashtag_tuples):
    top_hashtags = []
    for ht_tuple in top_hashtag_tuples:
        top_hashtags.append(ht_tuple[0])

    cooc_matrix = np.zeros((len(top_hashtags), len(top_hashtags)))

    with open(input_file_name, 'rb') as csv_f:
        csv_reader = csv.reader(csv_f, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in csv_reader: #returns a row as a list [id, text, hashtags, timestamp, retweeted?]
            if row[4] == '1':
                continue  #ignore retweets

            #construct matrix
            hashtags = row[2].split(',')
            for ht_1 in hashtags:
                if ht_1 in top_hashtags:
                    for ht_2 in hashtags:
                        if ht_2 in top_hashtags:
                            if ht_1 != ht_2:
                                cooc_matrix[top_hashtags.index(ht_1), top_hashtags.index(ht_2)] += 1.0

    #normalize
    for i in xrange(len(top_hashtags)):
        s = np.sum(cooc_matrix[i])
        if s == 0.0:
            cooc_matrix[i] = 0.0
        else:
            cooc_matrix[i] /= np.sum(cooc_matrix[i])

    #create similarities
    sim_matrix = np.zeros((len(top_hashtags), len(top_hashtags)))
    for i in xrange(len(top_hashtags)):
        for j in xrange(len(top_hashtags)):
            sim_matrix[i, j] = (cooc_matrix[i, j] + cooc_matrix[j, i]) / 2.0

    return sim_matrix, top_hashtags


def create_clusters(DBSCAN_clusters, top_hashtags):
    clusters = []
    hashtags = np.array(top_hashtags)
    for i in xrange(0, np.max(DBSCAN_clusters)+1):
        clusters.append(set(hashtags[DBSCAN_clusters == i]))
    rest = set(hashtags[DBSCAN_clusters == -1])
    for ht in rest:
        clusters.append(set([ht]))
    return clusters

def friendly_format_clusters(clusters):
    clusters = str(clusters)
    clusters = clusters.replace(', set(', '\nset(')
    return clusters

if __name__ == '__main__':
    #example usage script
    csv_file = 'twitter_uncompressed/20131001.csv'
    #determine most popular hashtags
    popular = determine_popular_hashtags_csv_file(csv_file, top=100)
    #create a graph of the most popular hashtags
    graph_popular_hashtags(csv_file, popular, time_division=ONE_MINUTE_MS)
    #construct co-occurrence matrix
    cooc_matrix, top = create_cooccurrence_matrix(csv_file, popular)
    #turn similarity matrix into distance matrix
    cooc_matrix = 1.0 - cooc_matrix
    #cluster with DBSCAN
    d = DBSCAN(eps=0.86, metric='precomputed')
    clusters = d.fit_predict(cooc_matrix)
    textual_clusters = create_clusters(clusters, top)
    print friendly_format_clusters(textual_clusters)