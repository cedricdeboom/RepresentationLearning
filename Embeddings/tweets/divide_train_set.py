__author__ = 'cedricdeboom'

import os

def divide(pairs_file, no_pairs_file, lines, parts=4):
    p = []
    np = []
    for i in xrange(parts):
        p.append(open(os.path.splitext(pairs_file)[0] + '.' + str(i) + '.txt', 'w'))
        np.append(open(os.path.splitext(no_pairs_file)[0] + '.' + str(i) + '.txt', 'w'))

    pf = open(pairs_file, 'r')
    npf = open(no_pairs_file, 'r')

    for i in xrange(lines):
        to_exclude = i // (lines / parts)
        p_line = pf.next()
        np_line = npf.next()
        for j in xrange(parts):
            if j != to_exclude:
                p[j].write(p_line)
                np[j].write(np_line)

    for i in xrange(parts):
        p[i].close()
        np[i].close()
    pf.close()
    npf.close()