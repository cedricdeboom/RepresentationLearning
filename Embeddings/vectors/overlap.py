__author__ = 'cedricdeboom'


import sys
import random

len_voc_1 = int(sys.argv[1]) - 1
words_in_sentence = int(sys.argv[2])
repetitions = int(sys.argv[3])

overlap = 0
different_sentences = set()

for i in xrange(repetitions):
    s1 = set()
    s2 = set()
    for w in xrange(words_in_sentence):
        s1.add(random.randint(0, len_voc_1))
        s2.add(random.randint(0, len_voc_1))
    overlap += (len(s1.intersection(s2)) > 0)

print float(overlap) / float(repetitions)

# for i in xrange(repetitions):
#     s1 = []
#     for w in xrange(words_in_sentence):
#         s1.append(random.randint(0, len_voc_1))
#     occ = set()
#     for i in s1:
#         occ.add(s1.count(i))
#     if occ == {2}:
#         different_sentences.add(tuple(s1))
#
# print len(different_sentences)