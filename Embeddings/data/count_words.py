"""
Count number of words in tweet collection
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2015 March 31st"


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def count(collection=''):
    counts = dict()

    tweet = 0
    with open(collection) as f:
        for line in f:
            if tweet%10000 == 0:
                print "Processing tweet " + str(tweet)
            if tweet == 500000:
                break

            #Example line: id="516577332946288641";cr="1411996608380";text="Wie o wie";us="365633279";lat="50.854306";lon="4.401551";repl="-1";retw="false";
            indexCR2 = line.index('\";text=\"')
            indexText2 = line.index('\";us=\"', indexCR2+1)
            text = line[indexCR2+8:indexText2].decode('utf8')
            words = text.split()

            if len(words) in counts.keys():
                counts[len(words)] += 1
            else:
                counts[len(words)] = 1

            tweet += 1

    plt.clf()
    m = max(counts.keys())
    x = np.linspace(m-20, m, 21)
    y = []
    for i in xrange(m-20, m+1):
        if i in counts.keys():
            y.append(counts[i])
        else:
            y.append(0)
    plt.plot(x, y)
    plt.savefig('counts.png')


if __name__ == '__main__':
    count('london2.txt')