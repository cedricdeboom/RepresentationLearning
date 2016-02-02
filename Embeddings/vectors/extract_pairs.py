"""
Extract semantically-related pairs of short texts out of wikipedia textual corpus (processed with cleanwiki script),
and word2vec model
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2015 April 14th"

from w2v import w2v
from random import shuffle, randint

def extract(words_per_text=30,
            skip=2,
            n_pairs=5000000,
            corpus='../data/enwiki.txt',
            w2v_model='../data/model/minimal',
            pairs_file='../data/pairs/enwiki_pairs_30.txt',
            no_pairs_file='../data/pairs/enwiki_no_pairs_30.txt'):

    w = w2v()
    print 'Loading model...'
    w.load_minimal(w2v_model)
    print 'Done.'
    o = open(pairs_file, 'w')
    n = open(no_pairs_file, 'w')

    pool = []
    add1 = 0
    add2 = 10

    current_pair = 0.0

    f = open(corpus, 'r')
    for line in f:
        words = line.split() #get words of current paragraph

        if len(words) >= words_per_text*2+words_per_text/2: #no of words needs to be sufficiently high
            pair1 = []
            pair2 = []
            while len(pair1) < words_per_text and len(words) > 0: #add words to first part of pair
                current = words.pop(0)
                if w.exists_word(current):
                    pair1.append(current)
            for s in xrange(skip):
                if len(words) > 0:
                    words.pop(0)
            while len(pair2) < words_per_text and len(words) > 0: #add words to second part of pair
                current = words.pop(0)
                if w.exists_word(current):
                    pair2.append(current)
            if len(pair1) == words_per_text and len(pair2) == words_per_text:
                if add1 == 0: #add pair1 to the pool
                    pool.append(pair1)
                if add2 == 0: #add pair2 to the pool
                    pool.append(pair2)
                add1 = (add1 + 1) % 20
                add2 = (add2 + 1) % 20
                o.write(' '.join(pair1) + ';' + ' '.join(pair2) + '\n') #write pairs to output file
                current_pair += 1.0

        if len(pool) >= 100: #process 'no pairs'
            print 'Progress %.3f%%' % (100.0*current_pair / n_pairs)
            for i in xrange(10):
                shuffle(pool)
                for p in xrange(len(pool)-1):
                    n.write(' '.join(pool[p]) + ';' + ' '.join(pool[p+1]) + '\n') #write pairs to output file
            pool = []
            o.flush()
            n.flush()
            if current_pair >= n_pairs:
                break

    f.close()
    o.close()

def extract_random(max_words_per_text=30,
            skip=2,
            n_pairs=5000000,
            corpus='../data/enwiki.txt',
            w2v_model='../data/model/minimal',
            pairs_file='../data/pairs/enwiki_pairs_r.txt',
            no_pairs_file='../data/pairs/enwiki_no_pairs_r.txt'):

    w = w2v()
    print 'Loading model...'
    w.load_minimal(w2v_model)
    print 'Done.'
    o = open(pairs_file, 'w')
    n = open(no_pairs_file, 'w')

    pool = []
    add1 = 0
    add2 = 10

    current_pair = 0.0

    f = open(corpus, 'r')
    for line in f:
        words = line.split() #get words of current paragraph

        words_per_text_1 = randint(10, max_words_per_text)
        words_per_text_2 = randint(10, max_words_per_text)

        if len(words) >= words_per_text_1 + words_per_text_2 + (words_per_text_1 + words_per_text_2)/4: #no of words needs to be sufficiently high
            pair1 = []
            pair2 = []
            while len(pair1) < words_per_text_1 and len(words) > 0: #add words to first part of pair
                current = words.pop(0)
                if w.exists_word(current):
                    pair1.append(current)
            for s in xrange(skip):
                if len(words) > 0:
                    words.pop(0)
            while len(pair2) < words_per_text_2 and len(words) > 0: #add words to second part of pair
                current = words.pop(0)
                if w.exists_word(current):
                    pair2.append(current)
            if len(pair1) == words_per_text_1 and len(pair2) == words_per_text_2:
                if add1 == 0: #add pair1 to the pool
                    pool.append(pair1)
                if add2 == 0: #add pair2 to the pool
                    pool.append(pair2)
                add1 = (add1 + 1) % 20
                add2 = (add2 + 1) % 20
                o.write(' '.join(pair1) + ';' + ' '.join(pair2) + '\n') #write pairs to output file
                current_pair += 1.0

        if len(pool) >= 100: #process 'no pairs'
            print 'Progress %.3f%%' % (100.0*current_pair / n_pairs)
            for i in xrange(10):
                shuffle(pool)
                for p in xrange(len(pool)-1):
                    n.write(' '.join(pool[p]) + ';' + ' '.join(pool[p+1]) + '\n') #write pairs to output file
            pool = []
            o.flush()
            n.flush()
            if current_pair >= n_pairs:
                break

    f.close()
    o.close()

if __name__ == '__main__':
    extract_random()