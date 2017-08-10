import time

# http://preshing.com/20110924/timing-your-code-using-pythons-with-statement/


class Timer(object):

    def __enter__(self):
        self.start_time = time.clock()
        return self

    def __exit__(self, *args):
        self.end_time = time.clock()
        self.interval = self.end_time - self.start_time
