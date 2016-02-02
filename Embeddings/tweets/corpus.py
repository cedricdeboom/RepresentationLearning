__author__ = 'cedricdeboom'

import os
import csv
import numpy as np

from tweet import Tweet

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


FORBIDDEN_HASHTAGS = {'#0', '#breaking', '#update', '#tweet', '#topstories', '#', '#photos', '#news',
                      '#potusplaylist', '#yoursay', '#viralvideos', '#longreads', '#c0news', '#ssnalerts',
                      '#sayfie', '#flapol', '#aljazeera', '#lbcblogs', '#video', '#breakingnews',
                      '#bbcgofigure', '#npreads', '#nbc0ny', '#abcnews0', '#0abc', '#nbc0today', '#nyccw0',
                      '#cbs0', '#developing', '#theinsider', '#bbc', '#tomorrowspaperstoday', '#cnnfc',
                      '#buffalo', '#twcnews', '#nbc0', '#cny', '#nbc0dc', '#asktheglobe', '#cambridge',
                      '#cambridgeshire', '#a0', '#fortuneinsider', '#fortunelive', '#fortunetech', '#foxnews',
                      '#global0', '#medhat', '#expressfrontage', '#expressopinion', '#india', '#beyondthenews',
                      '#sundayeye', '#freetimeoutny', '#wordonthestreet', '#aljazeeramag', '#aljazeerzworld',
                      '#jpost', '#foxinthefastlane', '#guardiangothenburg', '#bbcafricalive', '#bbcfocusafrica',
                      '#proverb', '#africa', '#bbcbreakfast', '#huffpostnow', '#washweek', '#onthisday',
                      '#philly', '#local', '#periscope', '#glasto0', '#exploreny', '#budget0', '#icymi', '#obama',
                      '#lateline', '#tca0', '#washweeklive', '#myrtlebeach', '#surfsidebeach', '#mugshots',
                      '#go2mb'}
ONE_DAY = 86400000L

def load_tweet_corpus(csv_dir):
    corpus = []

    for file in os.listdir(csv_dir):
        print 'Processing ' + file + '...'
        with open(os.path.join(csv_dir, file), 'rb') as csv_f:
            csv_reader = csv.reader(csv_f, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in csv_reader: #returns a row as a list [id, text, hashtags, timestamp, retweeted?]
                hashtags = row[2].split(',')
                ht_set = set()
                if len(hashtags) == 1 and len(hashtags[0]) == 0:
                    #there are no hashtags in the current tweet
                    pass
                else:
                    for ht in hashtags:
                        if len(ht) == 0 or ht in FORBIDDEN_HASHTAGS:
                            continue
                        ht_set.add(ht)

                corpus.append(Tweet(long(row[0]), row[1].split(), ht_set, long(row[3]), bool(row[4])))

    print 'Sorting...'
    corpus.sort()
    print 'Done.'

    return corpus

def generate_time_slots(corpus_list, min_timestamp=1433969373000L, time_division=ONE_DAY):
    tweets = []
    temp_tweets = 0
    start = True
    boundary_time = 0
    i = 0

    #create time-frequency table
    while i < len(corpus_list):
        if start:
            start = False
            boundary_time = corpus_list[0].timestamp - corpus_list[0].timestamp % time_division + time_division
        while i < len(corpus_list) and corpus_list[i].timestamp < boundary_time:
            temp_tweets += 1
            i += 1
        if i >= len(corpus_list) or corpus_list[i].timestamp >= min_timestamp:
            tweets.append(temp_tweets)
        temp_tweets = 0
        boundary_time += time_division

    return tweets

def generate_time_slots_tweets(corpus_list, min_timestamp=1433969373000L, time_division=ONE_DAY):
    tweets = []
    temp_tweets = []
    start = True
    boundary_time = 0
    i = 0

    #create time-frequency table
    while i < len(corpus_list):
        if start:
            start = False
            boundary_time = corpus_list[0].timestamp - corpus_list[0].timestamp % time_division + time_division
        while i < len(corpus_list) and corpus_list[i].timestamp < boundary_time:
            temp_tweets.append(corpus_list[i])
            i += 1
        if i >= len(corpus_list) or corpus_list[i].timestamp >= min_timestamp:
            tweets.append(temp_tweets)
        temp_tweets = []
        boundary_time += time_division

    return tweets

def graph_tweet_corpus(corpus_list, output_file, time_division=ONE_DAY):
    tweets = generate_time_slots(time_division)

    #create graph
    print 'Creating graphs ...'
    plt.clf()
    fig = plt.figure()
    fig.add_subplot(111)
    x = np.linspace(0, len(tweets)-1, len(tweets))
    plt.plot(x, tweets, 'b-', label='0')
    plt.savefig(output_file)
    plt.close()