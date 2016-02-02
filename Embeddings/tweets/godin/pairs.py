__author__ = 'cedricdeboom'

#Create pairs and non-pairs of tweets

import os
import csv
import random
import numpy as np

from ..tweet import Tweet

TEN_MINUTES_MS = 600000
FIVE_MINUTES_MS = 300000
ONE_MINUTE_MS = 60000


def naive_approach(input_csv_file_name, number_of_pairs=50, time_division=FIVE_MINUTES_MS):
    #divide twitter stream in bags of 5 minutes
    #choose random seed in bag, and find most similar tweet to this seed
    #most similar = closest in time with most overlap in hashtags

    tweets = []

    start = True
    boundary_time = 0
    temp_tweets = []

    with open(input_csv_file_name, 'rb') as csv_f:
        csv_reader = csv.reader(csv_f, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in csv_reader: #returns a row as a list [id, text, hashtags, timestamp, retweeted?]
            if row[4] == '1':
                continue  #ignore retweets
            if start:
                boundary_time = int(row[3])
                start = False

            hashtags = row[2].split(',')
            if len(hashtags) == 1 and len(hashtags[0]) == 0:
                continue #there are no hashtags in the current tweet
            ht_set = set()
            for ht in hashtags:
                if len(ht) == 0:
                    continue
                ht_set.add(ht)

            current_time = int(row[3])
            t = Tweet(int(row[0]), row[1], ht_set, current_time, bool(row[4]))

            if current_time % time_division == 0 and current_time > boundary_time:
                #append gathered tweets
                tweets.append(temp_tweets)
                #clean temporary tweets
                temp_tweets = []
                #update boundary time
                boundary_time = current_time

            temp_tweets.append(t)

    #generate number_of_pairs pairs
    i = 0
    while i < number_of_pairs:
        #pick bag at random
        r = random.randint(0, len(tweets)-1)
        bag = tweets[r]

        #pick seed tweet from bag at random
        r = random.randint(0, len(bag)-1)
        seed = bag[r]

        #find most similar tweet to seed
        max_overlap = 0.0
        most_similar = None
        forward = True
        backward = True
        forward_i = r+1
        backward_i = r-1
        forward_time = seed.timestamp
        backward_time = seed.timestamp

        while forward or backward:
            while forward:
                if forward_i >= len(bag):
                    forward = False
                    break
                else:
                    if bag[forward_i].timestamp == forward_time:
                        #count overlapping hashtags
                        overlap = score_overlapping_hashtags(seed, bag[forward_i])
                        if overlap > max_overlap:
                            max_overlap = overlap
                            most_similar = bag[forward_i]
                            if abs(overlap - 1.0) < 1E-9:
                                forward = False
                                backward = False
                                break
                        forward_i += 1
                    else:
                        forward_time = bag[forward_i].timestamp
                        break

            while backward:
                if backward_i < 0:
                    backward = False
                    break
                else:
                    if bag[backward_i].timestamp == backward_time:
                        #count overlapping hashtags
                        overlap = score_overlapping_hashtags(seed, bag[backward_i])
                        if overlap > max_overlap:
                            max_overlap = overlap
                            most_similar = bag[backward_i]
                            if abs(overlap - 1.0) < 1E-9:
                                forward = False
                                backward = False
                                break
                        backward_i -= 1
                    else:
                        backward_time = bag[backward_i].timestamp
                        break

        if max_overlap >= 0.5:
            i += 1
            print seed.text + "\t ;; \t" + most_similar.text


def predefined_hashtags(input_csv_file_name, number_of_pairs=50, hashtag_set=set(), time_division=FIVE_MINUTES_MS):
    #divide twitter stream in bags of 5 minutes
    #choose random seed in bag, and find most similar tweet to this seed
    #most similar = closest in time with most overlap in hashtags

    tweets = []

    start = True
    boundary_time = 0
    temp_tweets = []

    with open(input_csv_file_name, 'rb') as csv_f:
        csv_reader = csv.reader(csv_f, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in csv_reader: #returns a row as a list [id, text, hashtags, timestamp, retweeted?]
            if row[4] == '1':
                continue  #ignore retweets
            if start:
                boundary_time = int(row[3])
                start = False

            hashtags = row[2].split(',')
            if len(hashtags) == 1 and len(hashtags[0]) == 0:
                continue #there are no hashtags in the current tweet
            ht_set = set()
            ok = False
            for ht in hashtags:
                if len(ht) == 0:
                    continue
                ht_set.add(ht)
                if ht in hashtag_set:
                    ok = True
            if not ok:
                continue

            current_time = int(row[3])
            t = Tweet(int(row[0]), row[1], ht_set, current_time, bool(row[4]))

            if current_time % time_division == 0 and current_time > boundary_time:
                #append gathered tweets
                tweets.append(temp_tweets)
                #clean temporary tweets
                temp_tweets = []
                #update boundary time
                boundary_time = current_time

            temp_tweets.append(t)

    #generate number_of_pairs pairs
    i = 0
    while i < number_of_pairs:
        #pick bag at random
        r = random.randint(0, len(tweets)-1)
        bag = tweets[r]

        #pick seed tweet from bag at random
        r = random.randint(0, len(bag)-1)
        seed = bag[r]

        #find most similar tweet to seed
        max_overlap = 0.0
        most_similar = None
        forward = True
        backward = True
        forward_i = r+1
        backward_i = r-1
        forward_time = seed.timestamp
        backward_time = seed.timestamp

        while forward or backward:
            while forward:
                if forward_i >= len(bag):
                    forward = False
                    break
                else:
                    if bag[forward_i].timestamp == forward_time:
                        #count overlapping hashtags
                        overlap = score_overlapping_hashtags(seed, bag[forward_i])
                        if overlap > max_overlap:
                            max_overlap = overlap
                            most_similar = bag[forward_i]
                            if abs(overlap - 1.0) < 1E-9:
                                forward = False
                                backward = False
                                break
                        forward_i += 1
                    else:
                        forward_time = bag[forward_i].timestamp
                        break

            while backward:
                if backward_i < 0:
                    backward = False
                    break
                else:
                    if bag[backward_i].timestamp == backward_time:
                        #count overlapping hashtags
                        overlap = score_overlapping_hashtags(seed, bag[backward_i])
                        if overlap > max_overlap:
                            max_overlap = overlap
                            most_similar = bag[backward_i]
                            if abs(overlap - 1.0) < 1E-9:
                                forward = False
                                backward = False
                                break
                        backward_i -= 1
                    else:
                        backward_time = bag[backward_i].timestamp
                        break

        if most_similar is not None:
            i += 1
            print seed.text + "\t ;; \t" + most_similar.text



def count_overlapping_hashtags(tweet_1, tweet_2):
    res = 0
    for ht_2 in tweet_2.hashtags:
        if ht_2 in tweet_1.hashtags:
            res += 1
    return res

def score_overlapping_hashtags(tweet_1, tweet_2):
    overlap = count_overlapping_hashtags(tweet_1, tweet_2)
    return overlap / (float)(len(tweet_1.hashtags) + len(tweet_2.hashtags) - overlap)