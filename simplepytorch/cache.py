from collections import namedtuple


class Cache(dict):
    """Simple cache to store data.  It's just a dict with extra methods.

        cache = Cache()
        cache.add('b', 100)
        cache.streaming_mean('c', 100)
        cache['a'] = 1
        cache.clear()
    """

    def add(self, key, value):
        try:
            self[key] += value
        except KeyError:
            self[key] = value

    _streaming_mean_type = namedtuple('streaming_mean', ['mean', 'count'])

    def streaming_mean(self, key, value, n=1):
        """Update mean in rolling avg fashion, storing a total count along with
        the mean"""
        try:
            tup = self[key]
            nn = tup.count + n
            self[key] = self._streaming_mean_type(
                tup.count/nn*tup.mean + n/nn*value, nn)
        except KeyError:
            self[key] = self._streaming_mean_type(value, n)
