__author__ = 'cedricdeboom'

import random

import corpus
from tweet import Tweet


ONE_DAY = 86400000L
ONE_HOUR = 3600000L
QUARTER = 900000L

HASHTAG_JACCARD_THRESH = 0.5
WORD_JACCARD_THRESH = 0.5
WORD_COUNT_THRESH = 5

def generate_random_pairs(tweet_corpus, output_file='pairs.txt', number_of_pairs=50, min_timestamp=1433969373000L, time_division=QUARTER):
    #generate semantically similar pairs by maximizing hashtag overlap within a certain time slot
    time_slots = corpus.generate_time_slots_tweets(tweet_corpus, min_timestamp=min_timestamp, time_division=time_division)
    #generate number_of_pairs pairs
    i = 0
    f = open(output_file, 'w')
    while i < number_of_pairs:
        #pick bag at random
        r = random.randint(0, len(time_slots)-1)
        bag = time_slots[r]
        if len(bag) == 0:
            continue

        #pick seed tweet from bag at random
        seed = None
        tries = 0
        while True:
            r = random.randint(0, len(bag)-1)
            seed = bag[r]
            tries += 1
            if len(seed.hashtags) > 0 or tries > 30:
                break
        if tries > 30 or count_veritable_words(seed) < WORD_COUNT_THRESH:
            continue

        #find most similar tweet to seed
        max_overlap = 0.0
        most_similar = None
        for candidate in bag:
            if candidate.id == seed.id:
                continue
            score = score_overlapping_hashtags(seed, candidate)
            if score > max_overlap:
                max_overlap = score
                most_similar = candidate

        if max_overlap < HASHTAG_JACCARD_THRESH:
            continue

        if most_similar is not None and count_veritable_words(most_similar) >= WORD_COUNT_THRESH \
                and score_word_overlap(seed, most_similar) < WORD_JACCARD_THRESH:
            i += 1
            def_1, def_2 = remove_overlapping_hashtags(seed, most_similar)
            f.write(' '.join(def_1) + ";" + ' '.join(def_2) + '\n')
    f.close()

def generate_all_pairs(tweet_corpus, output_file='pairs.txt', min_timestamp=1433969373000L, time_division=QUARTER):
    #generate semantically similar pairs by maximizing hashtag overlap within a certain time slot
    time_slots = corpus.generate_time_slots_tweets(tweet_corpus, min_timestamp=min_timestamp, time_division=time_division)
    #generate all possible pairs
    i = 0
    f = open(output_file, 'w')

    for slot in time_slots:
        if len(slot) == 0:
            continue

        for seed in slot:
            if len(seed.hashtags) == 0 or count_veritable_words(seed) < WORD_COUNT_THRESH:
                continue

            for candidate in slot:
                #do not compare with previous tweets -> leads to double pairs
                if candidate.id <= seed.id:
                    continue

                #similararity tweet to seed
                score = score_overlapping_hashtags(seed, candidate)
                if score < HASHTAG_JACCARD_THRESH:
                    continue

                if count_veritable_words(candidate) >= WORD_COUNT_THRESH \
                        and score_word_overlap(seed, candidate) < WORD_JACCARD_THRESH:
                    i += 1
                    def_1, def_2 = remove_overlapping_hashtags(seed, candidate)
                    f.write(' '.join(def_1) + ";" + ' '.join(def_2) + '\n')
    f.close()

def generate_random_non_pairs(tweet_corpus, output_file='non_pairs.txt', number_of_pairs=48645, min_timestamp=1433969373000L, time_division=QUARTER):
    #generate semantically similar pairs by maximizing hashtag overlap within a certain time slot
    time_slots = corpus.generate_time_slots_tweets(tweet_corpus, min_timestamp=min_timestamp, time_division=time_division)
    #generate number_of_pairs non-pairs
    i = 0
    f = open(output_file, 'w')
    while i < number_of_pairs:
        #pick bag at random
        r = random.randint(0, len(time_slots)-1)
        bag = time_slots[r]
        if len(bag) < 5:
            continue

        #pick seed tweet from bag at random
        seed = None

        while True:
            r = random.randint(0, len(bag)-1)
            seed = bag[r]
            if count_veritable_words(seed) >= WORD_COUNT_THRESH:
                break

        #find a non-pair
        dissimilar = None
        tries = 0
        while True:
            r = random.randint(0, len(bag)-1)
            candidate = bag[r]
            tries += 1
            if tries >= 30:
                break
            if candidate.id == seed.id:
                continue
            score = score_overlapping_hashtags(seed, candidate)
            if (score < 0.000001 and count_veritable_words(candidate) >= WORD_COUNT_THRESH and \
                            score_word_overlap(seed, candidate) < WORD_JACCARD_THRESH) or tries >= 30:
                dissimilar = candidate
                break
        if tries >= 30:
            continue

        if dissimilar is not None:
            i += 1
            def_1, def_2 = remove_overlapping_hashtags(seed, dissimilar)
            f.write(' '.join(def_1) + ";" + ' '.join(def_2) + '\n')
    f.close()

def count_overlapping_hashtags(tweet_1, tweet_2):
    res = 0
    for ht_2 in tweet_2.hashtags:
        if ht_2 in tweet_1.hashtags:
            res += 1
    return res

def score_overlapping_hashtags(tweet_1, tweet_2):
    overlap = count_overlapping_hashtags(tweet_1, tweet_2)
    if overlap == 0:
        return 0.0
    return overlap / (float)(len(tweet_1.hashtags) + len(tweet_2.hashtags) - overlap)

def score_word_overlap(tweet_1, tweet_2):
    res = 0
    for word in tweet_2.text:
        if word in tweet_1.text:
            res += 1
    return res / (float)(len(tweet_1.text) + len(tweet_2.text) - res)

def count_veritable_words(tweet):
    count = 0
    for word in tweet.text:
        if word.startswith('#') or word.startswith('@') or word == '0' or word == 'url':
            pass
        else:
            count += 1
    return count

def remove_overlapping_hashtags(tweet_1, tweet_2):
    overlap = tweet_1.hashtags.intersection(tweet_2.hashtags)
    tweet_1_filtered = [i for i in tweet_1.text if i not in overlap]
    tweet_2_filtered = [i for i in tweet_2.text if i not in overlap]
    return tweet_1_filtered, tweet_2_filtered