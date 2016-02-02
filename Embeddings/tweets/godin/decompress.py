#!/bin/bash/python3
##### THIS IS A PYTHON3 FILE !!! ####

import sys
import os
import json
import re
import datetime
import unicodedata
import lzma
import tarfile
import csv
import html.parser


def process_xz_dir(input_dir_name, output_dir_name):
    for input_file in os.listdir(input_dir_name):
        if input_file.endswith('.xz'):
            output_file = input_file.replace('.xz', '.csv') #save in csv format
            if os.path.isfile(os.path.join(output_dir_name, output_file)): #do not overwrite existing files
                continue
            process_xz_file(os.path.join(input_dir_name, input_file),
                            os.path.join(output_dir_name, output_file))

def process_tgz_dir(input_dir_name, output_dir_name):
    for input_file in os.listdir(input_dir_name):
        if input_file.endswith('.tgz'):
            output_file = input_file.replace('.tgz', '.csv') #save in csv format
            if os.path.isfile(os.path.join(output_dir_name, output_file)): #do not overwrite existing files
                continue
            process_tgz_file(os.path.join(input_dir_name, input_file),
                            os.path.join(output_dir_name, output_file))

def process_xz_file(input_xz_file_name, output_csv_file_name):
    i = 0
    with open(output_csv_file_name, 'w', newline='') as csv_f:
        csv_writer = csv.writer(csv_f, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        with lzma.open(input_xz_file_name) as f:
            for line in f:
                process_json_tweet(line, csv_writer)
                if i % 100000 == 0:
                    print("Storing tweet %s in %s" % (str(i), output_csv_file_name))
                i += 1

def process_tgz_file(input_tgz_file_name, output_csv_file_name):
    i = 0
    with open(output_csv_file_name, 'w', newline='') as csv_f:
        csv_writer = csv.writer(csv_f, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        with tarfile.open(input_tgz_file_name) as tar:
            for m in tar.getmembers():
                f = tar.extractfile(m)
                for line in f:
                    process_json_tweet(line, csv_writer)
                    if i % 100000 == 0:
                        print("Storing tweet %s in %s" % (str(i), output_csv_file_name))
                    i += 1
                f.close()

def process_json_tweet(tweet, csv_writer):
    parsed = json.loads(tweet.decode('utf-8'))
    if 'lang' not in parsed:
        return
    if parsed['lang'] != 'en':
        return

    t_id = parsed['id']
    t_hashtags = []
    t_retweeted = 0
    t_timestamp = int(unix_time_millis(datetime.datetime.strptime(parsed['created_at'],
                                                                    '%a %b %d %H:%M:%S +0000 %Y')))
    if 'retweeted_status' not in parsed:
        temp_text = unicodedata.normalize('NFKD', parsed['text']).encode('ascii', 'ignore').decode('utf-8')
    else:
        temp_text = unicodedata.normalize('NFKD', parsed['retweeted_status']['text'])\
            .encode('ascii', 'ignore').decode('utf-8')
        t_retweeted = 1
    temp_text = re.sub('\n', ' ', temp_text)
    temp_text = re.sub('http\:\/\/t\.co\/[A-Za-z0-9]+', ' url ', temp_text)
    temp_text = re.sub('https\:\/\/t\.co\/[A-Za-z0-9]+', ' url ', temp_text)
    temp_text = html.parser.HTMLParser().unescape(temp_text)
    temp_text = re.sub('[^A-Za-z0-9\s#@$£&\']', ' ', temp_text)
    temp_text = re.sub(r"[-+]?[\d,]*\.*\d+", '0', temp_text)
    temp_text = temp_text.lower()
    temp_text = temp_text.split()

    if len(temp_text) == 0:
        return

    word_bag = []
    for word in temp_text:
        if word[0] == '#':
            hashtag_bag = word.split('#')
            for ht in range(1, len(hashtag_bag)):
                t_hashtags.append('#' + hashtag_bag[ht])
                word_bag.append('#' + hashtag_bag[ht])
        elif word == "\'":
            pass
        else:
            word_bag.append(word)
        if word == 'rt':
            t_retweeted = 1

    t_text = ' '.join(word_bag).strip()
    tweet = [t_id, t_text, ','.join(t_hashtags), t_timestamp, t_retweeted]
    csv_writer.writerow(tweet)


#date helper functions
def unix_time(dt):
    epoch = datetime.datetime.utcfromtimestamp(0)
    delta = dt - epoch
    return delta.total_seconds()

def unix_time_millis(dt):
    return unix_time(dt) * 1000.0


if __name__ == '__main__':
    input_dir_name = sys.argv[1]
    output_dir_name = sys.argv[2]

    process_tgz_dir(input_dir_name, output_dir_name)