from typing import List
import abc
import atexit
import csv
import datetime as dt
import gzip
import logging
import pandas as pd
import pickle
import os
import os.path as osp

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class DataLoggerException(Exception):
    pass


class _LogRotate_LazyPassthrough:
    """Transparent class for handling log rotation only when a method or
    attribute from the logger is accessed"""
    def __init__(self, kls, init_data_logger_func, utcnow, *args, **kwargs):
        self._kls = kls
        self._args = args
        self._kwargs = kwargs
        self._initialized_logger_instance = None
        self.init_data_logger_func = init_data_logger_func
        self.utcnow = utcnow

    def __getattribute__(self, attr_name):
        ga = object.__getattribute__
        if ga(self, '_initialized_logger_instance') is None:
            args, kwargs = ga(self, '_args'), ga(self, '_kwargs')
            utcnow = ga(self, 'utcnow')
            ga(self, 'init_data_logger_func')(utcnow, *args, **kwargs)
            setattr(self, '_initialized_logger_instance', ga(self, '_kls')(
                *args, **kwargs))
        return getattr(
            ga(self,'_initialized_logger_instance'), attr_name)

class LogRotate:
    """Store a history of any data logger to avoid overwriting old results

    log = LogRotate(CsvLogger)(...)

    If the log filepath is "./a/b/c.csv" then it will get renamed to
    ./a/b/log/{utc_timestamp}_c.csv and a symlink to './a/b/c.csv' will point
    to the file.

    :lazy_init:  If True, don't perform log rotate or initialize the logger
    class until an attribute or method is invoked from it.  This avoids writing
    to disk or doing rotation until the moment the file is written to.
    Even if lazy loading causes the log file to be created several minutes late,
    the start time used to define the log filename will be correct.
    """
    def __init__(self, data_logger_kls, lazy_init=True):
        self.kls = data_logger_kls
        self.lazy_init = lazy_init

    def __call__(self, *args, **kwargs):
        utcnow = dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')
        if self.lazy_init:
            return _LogRotate_LazyPassthrough(
                self.kls, self.init_data_logger, utcnow, *args, **kwargs)
        else:
            return self.init_data_logger(utcnow, log_fp, *args, **kwargs)

    def init_data_logger(self, utcnow, log_fp, *args, **kwargs):
        replacement_log_fp = f'{osp.dirname(log_fp)}/log/{utcnow}_{osp.basename(log_fp)}'

        if osp.islink(log_fp):
            os.remove(log_fp)
        rv = self.kls(replacement_log_fp, *args, **kwargs)
        os.symlink(f'log/{osp.basename(replacement_log_fp)}', log_fp)
        return rv


class DataLogger(abc.ABC):
    """
    Abstract base class for logging data.  Define a subclass and then use
    like this:

    >>> dlog = DataLoggerChildClass("./myfile", ['col1', 'col2'])
    >>> dlog.writerow({'a': 1, 'b': 2})
    >>> dlog.writerow([1, 2])  # does same thing
    """

    @abc.abstractmethod
    def _init_file_handler(self, log_fp, **kwargs):
        # this function expects to receive args and kwargs from __init__

        # self.fh = open(log_fp, 'w')
        raise NotImplementedError("subclasses should implement this")

    @abc.abstractmethod
    def _write_to_file_handler(self, rowdict):
        # actually write data to the file handler.

        #  self.fh.write(rowdict)
        raise NotImplementedError("subclasses should implement this")

    @abc.abstractmethod
    def flush(self):
        #  self.fh.flush()
        raise NotImplementedError("subclasses should implement this")

    @abc.abstractmethod
    def close(self):
        #  self.fh.close()
        raise NotImplementedError("subclasses should implement this")

    # below here are default DataLogger functions

    def __init__(self, log_fp: str, header: List[str], close_atexit=True, **extra_kwargs):
        os.makedirs(osp.dirname(osp.abspath(log_fp)), exist_ok=True)
        self.header = header
        self._header_set = set(header)
        self._init_file_handler(log_fp, header=header, close_atexit=close_atexit, **extra_kwargs)

        log.info("DataLogger writing to file: %s" % log_fp)

        if close_atexit:
            atexit.register(self.close)

    def _clean_rowdict(self, rowdict, ignore_missing, raise_if_extra_keys):
        # convert to dict
        if not isinstance(rowdict, dict):
            rowdict = dict(zip(self.header, rowdict))
        if raise_if_extra_keys:
            ds = set(rowdict).difference(self._header_set)
            if ds: raise DataLoggerException(
                f"Tried to write row, but found extra key(s): {ds}")
        else: # filter only the relevant keys
            rowdict = {
                k: v for k, v in rowdict.items() if k in self._header_set}

        # handle missing keys
        missing_keys = self._header_set.difference(rowdict.keys())
        if not ignore_missing and len(missing_keys) != 0:
            raise DataLoggerException(
                "Tried to write row, but missing key(s): %s" % missing_keys)
        else:  # okay to ignore missing keys. set missing keys to None
            rowdict = {k: rowdict.get(k)
                       for k in self.header}
        return rowdict

    def writerow(self, rowdict, ignore_missing=False, raise_if_extra_keys=True):
        """Write a row to the file handler.  if `ignore_missing`, allow some
        values to be empty.  if raise_if_extra_keys, do not allow unrecognized keys in the dict"""
        rowdict = self._clean_rowdict(rowdict, ignore_missing, raise_if_extra_keys)
        self._write_to_file_handler(rowdict)


class CsvLogger(DataLogger):
    def _init_file_handler(self, log_fp, header, **kwargs):
        if log_fp.endswith('.csv'):
            self.fh = open(log_fp, 'w')
        elif log_fp.endswith('.csv.gz'):
            self.fh = gzip.open(log_fp, 'wt')
        else:
            raise Exception("Unrecognized CsvLogger suffix: %s" % log_fp)
        self.writer = csv.DictWriter(self.fh, fieldnames=header)
        self.writer.writeheader()

    def _write_to_file_handler(self, rowdict):
        self.writer.writerow(rowdict)

    def flush(self):
        self.fh.flush()

    def close(self):
        self.fh.close()


class HDFLogger(DataLogger):
    """Write a dict where values are pandas dataframes to HDF file.

    Write:
        log = HDFLogger('test.h5', ['df1', ...], compression_level=0)
        log.writerow({"df1": pd.DataFrame({'col1': {('index1', 0): [1,2,3]}}), ...})

    Read:
        pd.HDFStore('test.h5', 'df1').keys()
    """
    def _init_file_handler(self, log_fp: str, compression_level=5, *args, **kwargs):
        self.complevel = compression_level
        assert log_fp.endswith('.h5')
        self.log_fp = log_fp
        self.fh = pd.HDFStore(log_fp, 'w', complevel=compression_level)

    def _write_to_file_handler(self, rowdict):
        for k, v in rowdict.items():
            if v is None: continue
            self.fh.append(k, v)

    def flush(self):
        self.fh.flush()

    def close(self):
        self.fh.close()


class PickleLogger(DataLogger):
    def _init_file_handler(self, log_fp: str, **kwargs):
        assert log_fp.endswith('.pickle')
        self.fh = open(log_fp, 'wb')

    def _write_to_file_handler(self, rowdict):
        pickle.dump(rowdict, self.fh)

    def flush(self):
        self.fh.flush()

    def close(self):
        self.fh.close()

    @staticmethod
    def unpickler(fp):
        return unpickler(fp)

def unpickler(fp):
    """
    Example function to get data from PickleLogger
    >>> dcts = list(unpickler('data/results/test/grads.pickle'))
    """
    with open(fp, 'rb') as file_handler:
        while file_handler.peek(1):
            yield pickle.load(file_handler)


if __name__ == "__main__":

    # for TESTING

    a = CsvLogger('a.csv.gz', ['a', 'b'])
    b = CsvLogger('b.csv', ['a', 'b'])

    for x in [a, b]:
        x.writerow({'a': 1, 'b': 2})
        x.writerow([1, 2])

    # optional close, since done atexit by default
    a.close()

    # check results on cmdline
    # zcat a.csv.gz
    # cat b.csv

    c = PickleLogger('c.pickle', ['a', 'b'])

    for x in [c]:
        x.writerow({'a': 1, 'b': 2})
        x.writerow([3, 4])

    # optional close, since done atexit by default
    c.close()

    _h = open('c.pickle', 'rb')
    print(pickle.load(_h))
    print(pickle.load(_h))
