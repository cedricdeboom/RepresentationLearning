
import time
import pycurl
import urllib
import json
import oauth2 as oauth
# get all settings, see http://www.arngarden.com/2012/11/07/consuming-twitters-streaming-api-using-python-and-curl/
from configWorld import *
import logging
import re
import nltk
import optparse
import sys

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2014 September 30th"

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger=logging.getLogger("TwitterStream")

class GetTrending:
    def __init__(self, timeout=False):
        self.oauth_token = oauth.Token(key=OAUTH_KEYS['access_token_key'], secret=OAUTH_KEYS['access_token_secret'])
        self.oauth_consumer = oauth.Consumer(key=OAUTH_KEYS['consumer_key'], secret=OAUTH_KEYS['consumer_secret'])
        self.conn = None
        self.buffer = ''
        self.timeout = timeout
        self.data = None

    def setup_connection(self, get_params):
        """ Create persistent HTTP connection to Streaming API endpoint using cURL.
        """
        if self.conn:
            self.conn.close()
            self.buffer = ''
        self.conn = pycurl.Curl()
        # Restart connection if less than 1 byte/s is received during "timeout" seconds
        if isinstance(self.timeout, int):
            self.conn.setopt(pycurl.LOW_SPEED_LIMIT, 1)
            self.conn.setopt(pycurl.LOW_SPEED_TIME, self.timeout)
        #self.conn.setopt(pycurl.PROXY, 'http://proxy.atlantis.ugent.be:8080/')
        #self.conn.setopt(pycurl.PROXY, 'http://proxy.test:8080/')
        self.conn.setopt(pycurl.URL, API_ENDPOINT_URL + '?' + urllib.urlencode(get_params))
        self.conn.setopt(pycurl.USERAGENT, USER_AGENT)
        # Using gzip is optional but saves us bandwidth.
        self.conn.setopt(pycurl.ENCODING, 'deflate, gzip')
        self.conn.setopt(pycurl.HTTPGET, 1)
        self.conn.setopt(pycurl.HTTPHEADER, ['Authorization: %s' % self.get_oauth_header(get_params)])
        # self.handle_trending is the method that are called when new tweets arrive
        self.conn.setopt(pycurl.WRITEFUNCTION, self.handle_trending)

    def get_oauth_header(self, get_params):
        """ Create and return OAuth header.
        """
        params = {'oauth_version': '1.0',
                  'oauth_nonce': oauth.generate_nonce(),
                  'oauth_timestamp': int(time.time())}
        req = oauth.Request(method='GET', parameters=params, url='%s?%s' % (API_ENDPOINT_URL,
                                                                             urllib.urlencode(get_params)))
        req.sign_request(oauth.SignatureMethod_HMAC_SHA1(), self.oauth_consumer, self.oauth_token)
        return req.to_header()['Authorization'].encode('utf-8')

    def start(self, get_params):
        """ Start listening to Streaming endpoint.
        Handle exceptions according to Twitter's recommendations.
        """
        backoff_network_error = 0.25
        backoff_http_error = 5
        backoff_rate_limit = 60
        self.setup_connection(get_params)
        try:
            self.conn.perform()
        except:
            # Network error, use linear back off up to 16 seconds
            logger.error('Network error: %s' % self.conn.errstr())
            logger.error('Waiting %s seconds before trying again' % backoff_network_error)
            time.sleep(backoff_network_error)
            backoff_network_error = min(backoff_network_error + 1, 16)
            return 1
        # HTTP Error
        sc = self.conn.getinfo(pycurl.HTTP_CODE)
        if sc == 420 or sc == 429:
            # Rate limit, use exponential back off starting with 1 minute and double each attempt
            logger.error('Rate limit, waiting %s seconds' % backoff_rate_limit)
            #time.sleep(backoff_rate_limit)
            #backoff_rate_limit *= 2
            return 2
        elif sc != 200:
            # HTTP error, use exponential back off up to 320 seconds
            logger.error('HTTP error %s, %s' % (sc, self.conn.errstr()))
            logger.error('Waiting %s seconds' % backoff_http_error)
            time.sleep(backoff_http_error)
            backoff_http_error = min(backoff_http_error * 2, 320)
            return 2
        return 0

    def handle_trending(self, data):
        """ This method is called when data is received through Streaming endpoint.
        """
        self.buffer += data
        if data.endswith('}]') and self.buffer.strip():
            # complete message received
            try:
                message = json.loads(self.buffer)
                self.SaveMsg(message)
            except:
                pass
    
    def SaveMsg(self, data):
        self.data = data