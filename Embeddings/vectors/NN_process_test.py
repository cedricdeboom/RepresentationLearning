__author__ = 'cedricdeboom'



import NN_process
from threading import Thread
import time
import copy


NO_DATA = 4900000
BATCH_SIZE = 100
BATCHES = NO_DATA / BATCH_SIZE
EMBEDDING_DIM = 400
WORDS = 20
INPUT_SHAPE = (400, 20)
OUTPUT_SHAPE = (400, 1)


processor = NN_process.PairProcessor('../data/pairs/enwiki_pairs_20.txt', '../data/pairs/enwiki_no_pairs_20.txt',
                                  '../data/model/docfreq.npy', '../data/model/minimal', WORDS, EMBEDDING_DIM, BATCH_SIZE)
t = Thread(target=processor.process)
t.start()
print 'Start processor thread'

processor.new_epoch()
processor.lock.acquire()
while not processor.ready:
    processor.lock.wait()
processor.lock.release()

processor.lock.acquire()
processor.cont = True
processor.ready = False
processor.lock.notifyAll()
processor.lock.release()

for b in xrange(BATCHES):
    time.sleep(1)

    processor.lock.acquire()
    while not processor.ready:
        processor.lock.wait()
    processor.lock.release()

    print processor.y
    print processor.z

    processor.lock.acquire()
    processor.cont = True
    processor.ready = False
    if b == BATCHES-1:
        processor.stop = True
    processor.lock.notifyAll()
    processor.lock.release()