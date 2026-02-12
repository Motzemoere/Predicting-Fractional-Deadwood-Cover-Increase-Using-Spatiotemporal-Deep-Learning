"""
Parallel processing utilities using joblib.
Provides convenient wrappers for multi-core computation with progress tracking.
"""

from joblib import Parallel, delayed
import joblib
from typing import List, Iterable, Callable
from tqdm import tqdm
import contextlib
import warnings
import os

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    # credits:
    # https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def paral(function: Callable,
          iters: List[Iterable],
          num_cores=-1,
          progress_bar=True,
          backend="loky",
          return_as="list",
          total=None):
    """ compute function parallel with arguments in iters.
    function(iters[0][0],iters[0][1],...)"""

    if total is None:
        total = len(iters[0])

    with tqdm_joblib(
            tqdm(desc=function.__name__,
                 unit="jobs",
                 dynamic_ncols=True,
                 total=total,
                 disable=not progress_bar), ) as progress_bar:
        # backend can be loky or threading (or maybe something else)
        # Set LOKY_MAX_CPU_COUNT to avoid worker startup issues
        os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(os.cpu_count() or 1))
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", 
                                  message=".*worker stopped while some jobs were given.*",
                                  category=UserWarning,
                                  module="joblib")
            return Parallel(n_jobs=num_cores,
                            batch_size=1,
                            backend=backend,
                            return_as=return_as,
                            timeout=300)(delayed(function)(*its)
                                         for its in zip(*iters))