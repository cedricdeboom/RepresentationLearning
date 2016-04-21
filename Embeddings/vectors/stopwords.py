"""
Stopwords experiments on Wikipedia
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2016 February 10th"


def read_stopwords(input_file='stopwords.txt'):
    stopwords = set()

    f = open(input_file, 'r')
    for line in f:
        a = line.split('\'')
        for item in a:
            if len(item[0:-1])>0:
                stopwords.add(item[0:-1])
    f.close()
    return stopwords


def count_fraction_stopwords(stopwords, wiki_pairs_file='pairs.txt', lines=100000):
    f = open(wiki_pairs_file, 'r')
    current_line = 0
    mean = 0.0
    for line in f:
        p = line.split(';')
        for pi in p:
            count = 0
            words = pi.split(' ')
            for word in words:
                if word in stopwords:
                    count += 1
            mean += count
        current_line += 1
    mean /= (float(current_line) * 2)
    f.close()

    return mean


if __name__ == '__main__':
    stop_file = '../data/wiki/model/stopwords.txt'
    wiki_file = '../data/wiki/pairs/sets/enwiki_no_pairs_20-train.txt'

    s = read_stopwords(stop_file)
    m = count_fraction_stopwords(s, wiki_file, 200000)

    print m