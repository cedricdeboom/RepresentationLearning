"""
Data processor for tweets
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2015 February 25th"


import re
import operator
import cPickle
import random
import os


#HASHTAGS = {'#isis', '#weather', '#prideofbritainawards', '#eastenders', '#gbbo', '#britishmuseum', '#xfactor'}
HASHTAGS = {'#britishmuseum'}


class Processor:
    def __init__(self, filename):
        self.filename = filename
        self.hashtag_counts = dict()
        self.hashtag_tweets = dict()

    def readFile(self):
        tweet = 0
        with open(self.filename) as f:
            for line in f:

                if tweet%10000 == 0:
                    print "Processing tweet " + str(tweet)
                if tweet == 200000:
                    break

                #Example line: id="516577332946288641";cr="1411996608380";text="Wie o wie";us="365633279";lat="50.854306";lon="4.401551";repl="-1";retw="false";
                indexCR2 = line.index('\";text=\"')
                indexText2 = line.index('\";us=\"', indexCR2+1)
                text = line[indexCR2+8:indexText2].decode('utf8')

                #Find hashtags
                tokens = re.split('\s+', text)
                for token in tokens:
                    if token.startswith('#'):
                        hts = token.split('#')
                        for i in range(1,len(hts)):
                            if ('#'+hts[i]) in self.hashtag_counts.keys():
                                self.hashtag_counts['#'+hts[i]] += 1
                                #self.hashtag_tweets['#'+hts[i]].append(text)
                            else:
                                self.hashtag_counts['#'+hts[i]] = 1
                                #self.hashtag_tweets['#'+hts[i]] = list()

                tweet += 1

    def printHashtagCounts(self):
        sorted_hashtags = sorted(self.hashtag_counts.items(), key=operator.itemgetter(1))
        for t in sorted_hashtags:
            if t[1] > 20:
                print t[0] + "\t\t" + str(t[1])

    def collectTweets(self, output):
        tweet = 0
        with open(self.filename) as f:
            for line in f:

                if tweet%10000 == 0:
                    print "Processing tweet " + str(tweet)
                if tweet == 200000:
                    break

                #Example line: id="516577332946288641";cr="1411996608380";text="Wie o wie";us="365633279";lat="50.854306";lon="4.401551";repl="-1";retw="false";
                indexCR2 = line.index('\";text=\"')
                indexText2 = line.index('\";us=\"', indexCR2+1)
                text = line[indexCR2+8:indexText2].decode('utf8').lower()

                #Find hashtags
                tokens = re.split('\s+', text)
                for token in tokens:
                    if token.startswith('#'):
                        hts = token.split('#')
                        for i in range(1,len(hts)):
                            if ('#'+hts[i]) in HASHTAGS:
                                filtered = re.sub('#'+hts[i], ' ', text)
                                filtered = re.sub('\s+', ' ', filtered)
                                if ('#'+hts[i]) in self.hashtag_counts.keys():
                                    self.hashtag_counts['#'+hts[i]] += 1
                                    self.hashtag_tweets['#'+hts[i]].append(filtered)
                                else:
                                    self.hashtag_counts['#'+hts[i]] = 1
                                    self.hashtag_tweets['#'+hts[i]] = [filtered]

                tweet += 1

        f = file(output, 'wb')
        cPickle.dump(self.hashtag_counts, f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(self.hashtag_tweets, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def scorePairs(self, inputfile, count=150):
        f = file(inputfile, 'r')
        self.hashtag_counts = cPickle.load(f)
        self.hashtag_tweets = cPickle.load(f)
        f.close()

        for hashtag in HASHTAGS:
            print "\n\nHASHTAG " + hashtag + "\n"
            counts = list()
            previous_1 = 0
            previous_2 = 0
            for pair in xrange(count):
                tweet1_index = 0
                tweet2_index = 0
                while(tweet1_index == tweet2_index or (tweet1_index, tweet2_index) in counts or (tweet2_index, tweet1_index) in counts):
                    tweet1_index = random.randint(0, self.hashtag_counts[hashtag] - 1)
                    tweet2_index = random.randint(0, self.hashtag_counts[hashtag] - 1)

                counts.append((tweet1_index, tweet2_index))
                print self.hashtag_tweets[hashtag][tweet1_index]
                print self.hashtag_tweets[hashtag][tweet2_index]

                s1 = set(self.hashtag_tweets[hashtag][tweet1_index].split(' '))
                s2 = set(self.hashtag_tweets[hashtag][tweet2_index].split(' '))
                s = s1.intersection(s2)
                print "Common words: " + str(s)

                sim = raw_input("Sim: ")

                if sim == 'r':
                    os.remove('sims/' + hashtag + "_" + str(previous_1) + "_" + str(previous_2))
                    counts.pop()
                    print 'Removed previous record.'
                    sim = raw_input("Sim: ")

                sim = float(sim)/10.0
                f = file('sims/' + hashtag + "_" + str(tweet1_index) + "_" + str(tweet2_index), 'w')
                f.write(self.hashtag_tweets[hashtag][tweet1_index] + '\n')
                f.write(self.hashtag_tweets[hashtag][tweet2_index] + '\n')
                f.write(str(sim))
                f.close()

                previous_1 = tweet1_index
                previous_2 = tweet2_index


    def dataAugmentation(self, inputfile):
        f = open(inputfile, 'r')
        self.hashtag_counts = cPickle.load(f)
        self.hashtag_tweets = cPickle.load(f)
        f.close()

        while True:
            hashtag = raw_input("Hashtag: ")
            if hashtag in self.hashtag_counts.keys():
                break

        for tweet in self.hashtag_tweets[hashtag]:
            print tweet

        l = os.listdir('sims/')
        number = 0
        for file in l:
            if file.startswith('extra_'):
                number += 1

        while True:
            t1 = raw_input("Tweet 1: ")
            t2 = raw_input("Tweet 2: ")
            sim = raw_input("Sim: ")
            sim = float(sim)/10.0
            f = open('sims/extra_' + str(number), 'w')
            f.write(t1 + '\n')
            f.write(t2 + '\n')
            f.write(str(sim))
            f.close()
            number += 1

    def processSims(self, output):
        tweets = []
        sims = []

        for file in os.listdir('sims/'):
            if file.startswith('.'):
                continue
            print file
            f = open('sims/' + file, 'r')
            t1 = f.readline().strip()
            t2 = f.readline().strip()
            s = float(f.readline().strip())
            f.close()

            tweets.append((t1, t2))
            sims.append(s)

        f = open(output, 'wb')
        cPickle.dump(tweets, f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(sims, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

if __name__ == '__main__':
    p = Processor('london2.txt')
    #p.collectTweets(output='london2.dump')
    #p.scorePairs('london2.dump')
    #p.dataAugmentation('london2.dump')
    p.processSims('sims.dump')