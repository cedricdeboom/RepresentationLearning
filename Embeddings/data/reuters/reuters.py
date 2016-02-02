"""
Data processor for Reuters dataset
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2015 March 12th"


import re
import json
import copy
import random

TWEETDIM = 140
MIN_LENGTH = 50

random.seed(94612)

"""
Process Reuters data file to extract pairs of 'tweets'.
"""
def processReuters(file="reuters.json", output="outputa", topics={"GENT", "GFAS", "GTOUR"}):
    fo1 = open(output+"_1.txt", 'w')
    fo2 = open(output+"_2.txt", 'w')

    f = open(file, 'r')
    for line in f:
        A = json.loads(line)
        hasTopic = False
        for t in A['topics']:
            if t in topics:
                hasTopic = True
                break
        if not hasTopic:
            continue

        pairs = extractPairs2(A['title'], A['body'])
        for i in xrange(len(pairs[0])):
            fo1.write(pairs[0][i] + '\n')
            fo2.write(pairs[1][i] + '\n')

        fo1.flush()
        fo2.flush()

    f.close()

    fo1.close()
    fo2.close()

"""
PRIVATE -- Extract pairs out of textual body and title (sentence-wise).
"""
def extractPairs(title="", body=""):
    lines = getSentences(body)
    pairs = [list(), list()]

    p1 = filter(removeNR(getTrimmedText(title)))
    p2 = filter(removeNR(getTrimmedText(lines[0])))
    pairs[0].append(p1)
    pairs[1].append(p2)

    j = 1
    while True:
        while True:
            if(j+1 >= len(lines)):
                break
            p1 = removeNR(getTrimmedText(lines[j]))
            p2 = removeNR(getTrimmedText(lines[j+1]))
            if len(p1) >= MIN_LENGTH and len(p2) >= MIN_LENGTH:
                break
            j += 1

        if(j+1 >= len(lines)):
            break

        pairs[0].append(p1)
        pairs[1].append(p2)
        j += 2

    return pairs


"""
PRIVATE -- Extract pairs out of textual body and title (word-wise).
"""
def extractPairs2(title="", body="", aug=True):
    words = filter(removeNR(body)).split()
    pairs = [list(), list()]

    p1 = filter(removeNR(getTrimmedText(title)))
    (p2, words) = getTrimmedList(words)
    if aug:
        pp = augment(p1, 10)
        for i in xrange(len(pp)):
            pairs[0].append(pp[i])
            pairs[1].append(' '.join(p2))
    else:
        pairs[0].append(p1)
        pairs[1].append(' '.join(p2))
    words[0:30] = []

    while True:
        if len(words) < 1:
            break
        (p1, words) = getTrimmedList(words)
        if len(words) < 1:
            break
        (p2, words) = getTrimmedList(words)
        p1 = ' '.join(p1)
        p2 = ' '.join(p2)
        if len(p2) < MIN_LENGTH or len(p1) < MIN_LENGTH:
            break

        if aug:
            pp = augment(p1, 10)
            for i in xrange(len(pp)):
                pairs[0].append(pp[i])
                pairs[1].append(p2)
        else:
            pairs[0].append(p1)
            pairs[1].append(p2)

        words[0:30] = []

    return pairs

"""
PRIVATE -- Augment pairs by randomly switching words
"""
def augment(text="", amount=10):
    words = text.split()
    res = list()
    for i in xrange(amount):
        random.shuffle(words)
        s = ' '.join(words)
        res.append(s)
    return res


"""
PRIVATE -- Remove new lines and carriage returns
"""
def removeNR(text=""):
    r = re.sub('\n', ' ', text)
    r = re.sub('\r', '', r)
    return r

"""
PRIVATE -- Get sentence lines
"""
def getSentences(body=""):
    lines = body.split('\n')
    processed = list()
    for l in lines:
        processed.append(re.sub('\r', ' ', l))
    result = list()
    current = ""
    for i in xrange(len(processed)):
        if(len(current) != 0):
            current += " "
        current += processed[i]
        if not current[-1].isalpha():
            result.append(filter(current))
            current = ""
    return result

"""
PRIVATE -- Filtering (lowercase, replacing numbers)
"""
def filter(text=""):
    r = text.lower()
    return re.sub(r"[-+]?[\d,]*\.*\d+", '0', r)

"""
PRIVATE -- Trim text to max 140 chars
"""
def getTrimmedText(text=""):
    words = text.split()
    res = list()
    i = 0
    length = -1
    while True:
        if length + len(words[i]) + 1 > TWEETDIM:
            break
        res.append(words[i])
        length += (len(words[i]) + 1)
        i += 1
        if i >= len(words):
            break
    return ' '.join(res)

"""
PRIVATE -- Trim list of words to max 140 chars
"""
def getTrimmedList(words=list()):
    res = list()
    res2 = copy.deepcopy(words)
    i = 0
    length = -1
    while True:
        if length + len(words[i]) + 1 > TWEETDIM:
            break
        res.append(words[i])
        res2.pop(0)
        length += (len(words[i]) + 1)
        i += 1
        if i >= len(words):
            break
    return (res, res2)


if __name__ == '__main__':
    processReuters()