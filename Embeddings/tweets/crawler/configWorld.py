#!/usr/bin/env python
"""
@author:    Matthias Feys (matthiasfeys@gmail.com), IBCN (Ghent University)
@date:      Fri Nov 29 13:17:21 2013
"""

#API_ENDPOINT_URL = 'https://api.twitter.com/1.1/trends/place.json'
API_ENDPOINT_URL = 'https://api.twitter.com/1.1/statuses/user_timeline.json'
USER_AGENT = 'TwitterStream 1.0' # This can be anything really

# You need to replace these with your own values
OAUTH_KEYS = {'consumer_key': 'gCHXyanIfYm8aRx1Z1daGxuH4',
              'consumer_secret': '1TotkwwSzvbE8KkRBv4THowCK5JcYJZnMOmi8hehvEdjIm1Fkx',
              'access_token_key': '2794817867-PTyxbqRC1UqhQ4fu8P52GRB3bYyvk1Qr4Dj5X6s',
              'access_token_secret': 'leHdGYXUMZhNQTBC1a6VP5zGmOfGKI3yGsn2g0W0yxXwx'
              }

OAUTH_KEYS2 = {'consumer_key': 'A6l8OCJJ7zaZNqc9YnjCKMtzN',
              'consumer_secret': 'ZT0eP6UhadFr234HG9NI5FQo3tshpy2Ehw2XiGZaQRZq1yBolM',
              'access_token_key': '2813604911-XXa74eUTzvfu2GI4U1UFrf7rlUypAFC1jDJdugz',
              'access_token_secret': 'KdYyervChI3yL5lsisM3yNrlXNKCZVQqY1lOMI6IbPj16'
              }

OAUTH_KEYS3 = {'consumer_key': 'yubl8UflSddfgemsIdWpPxMic',
              'consumer_secret': '7NTNFRkt5rqG6MwoK7WmsUD1g4OizCrrskJkbjWQEmkpa5wybN',
              'access_token_key': '2814433305-kcHN4rG70D8134ZofvI8fkNfsI1kW6lxsmykB7I',
              'access_token_secret': '5cnWzShzuCqV7fQq3tkhkReOiilu3AL9kVXHUxHBYN411'
              }

#GET_PARAMS = {'id': '23424757'}
# GET_PARAMS = {'q': '#goedvolk',
#                'since_id': '500000000000000000',
#                'max_id': '517218770793025536',
#                'until': '2014-10-02',
#                'count': '100'
#                }
               
               
# ***************** END of CONFIGURATION ****************