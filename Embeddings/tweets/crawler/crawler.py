__author__ = 'cedricdeboom'



from GetTrending import *
import time
import sys
import os
import subprocess

LAST_ID = 1200000000000000000

GET_PARAMS = {'screen_name': '',
              'max_id': str(LAST_ID),
              'since_id': 500000000000000000,
              'count': '200',
              'trim_user': 'true',
              }

class Crawler:
    def __init__(self, username):
        self.filename = os.path.join('crawls/', username)
        self.username = username
        self.firstid = 0
        self.account = 1

    def crawl(self):
        ts = GetTrending()
        processed = 0
        retryCount = 0
        GET_PARAMS['screen_name'] = self.username
        while True:
            print 'Crawling '+self.username+' ('+str(processed)+')'
            ret = ts.start(GET_PARAMS)
            if ret == 0:
                retryCount = 0
                processed += 1
                minID = self.handleIncoming(ts.data)
                GET_PARAMS['max_id'] = str(minID - 1)
                if minID < 0:
                    break
            elif(ret == 2):
                #BACK OFF
                print 'BACKING OFF...'
                retryCount = 0
                time.sleep(10)
                print 'SWITCHING ACCOUNT'
                if self.account == 1:
                    ts.oauth_token = oauth.Token(key=OAUTH_KEYS2['access_token_key'], secret=OAUTH_KEYS2['access_token_secret'])
                    ts.oauth_consumer = oauth.Consumer(key=OAUTH_KEYS2['consumer_key'], secret=OAUTH_KEYS2['consumer_secret'])
                    self.account = 2
                elif self.account == 2:
                    ts.oauth_token = oauth.Token(key=OAUTH_KEYS3['access_token_key'], secret=OAUTH_KEYS3['access_token_secret'])
                    ts.oauth_consumer = oauth.Consumer(key=OAUTH_KEYS3['consumer_key'], secret=OAUTH_KEYS3['consumer_secret'])
                    self.account = 3
                else:
                    ts.oauth_token = oauth.Token(key=OAUTH_KEYS['access_token_key'], secret=OAUTH_KEYS['access_token_secret'])
                    ts.oauth_consumer = oauth.Consumer(key=OAUTH_KEYS['consumer_key'], secret=OAUTH_KEYS['consumer_secret'])
                    self.account = 1
            else:
                #retry
                print 'RETRYING'
                retryCount += 1
                if retryCount > 4:
                    break
                time.sleep(1)

    def write(self, data):
        print 'Writing...'
        f = open(self.filename + '.' + str(self.firstid) + '.txt', 'a')
        for status in data:
            f.write(json.dumps(status)+'\n')
            f.flush()
        f.close()

    def handleIncoming(self, data):
        if data is None:
            print "0"
            return -1
        print len(data)
        minID = long(GET_PARAMS['max_id'])

        for status in data:
            id = long(status['id'])
            if self.firstid == 0:
                self.firstid = id
            if id < minID:
                minID = id
        if minID >= long(GET_PARAMS['max_id']):
            return -1

        self.write(data)
        if len(data) < 100:
            return -1
        return minID


if __name__ == '__main__':
    account = 1
    for line in sys.stdin:
        un = line.strip()

        #find largest crawled id
        ids = set()
        for f in os.listdir('crawls/'):
            if f.startswith(un):
                ids.add(long(f.split('.')[-2]))
        if len(ids) == 0:
            GET_PARAMS['since_id'] = 500000000000000000
        else:
            GET_PARAMS['since_id'] = str(max(ids))

        GET_PARAMS['max_id'] = LAST_ID
        c = Crawler(un)
        c.account = account
        c.crawl()
        account = c.account

        full_file_name = c.filename + '.' + str(c.firstid)
        subprocess.call("tar -zcvf " + full_file_name + ".tgz " + full_file_name + ".txt", shell=True)
        subprocess.call("rm " + full_file_name + ".txt", shell=True)