import time
from contextlib import contextmanager

@contextmanager
def compile_timer(name):
    start_time = time.time()
    print('Compiling %s... ' % name)
    yield 
    elapsed = time.time()-start_time
    print('\t took %.3f sec' % elapsed)
