

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2014 October 1st"


class Tweet:
    def __init__(self, id, text, hashtags, timestamp, retweeted):
        self.id = id
        self.text = text
        self.timestamp = timestamp
        self.hashtags = hashtags
        self.retweeted = retweeted
        
    def __str__(self):
        return 'id=\"' + str(self.id) + \
            '\";cr=\"' + str(self.timestamp) + \
            '\";text=\"' + self.text + '\";'
        
    def __eq__(self, other):
        if isinstance(other, Tweet):
            return self.id == other.id
        return NotImplemented
    
    def __ne__(self, other):
        if isinstance(other, Tweet):
            return self.id != other.id
        return NotImplemented
    
    def __lt__(self, other):
        if isinstance(other, Tweet):
            return self.timestamp < other.timestamp
        return NotImplemented
    
    def __gt__(self, other):
        if isinstance(other, Tweet):
            return self.timestamp > other.timestamp
        return NotImplemented
    
    def __le__(self, other):
        if isinstance(other, Tweet):
            return self.timestamp <= other.timestamp
        return NotImplemented
    
    def __ge__(self, other):
        if isinstance(other, Tweet):
            return self.timestamp >= other.timestamp
        return NotImplemented
    
    def __hash__(self):
        return hash(self.id)