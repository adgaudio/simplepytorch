from collections import namedtuple


BestPerf = namedtuple('BestPerf', ['value', 'epoch'])


class EarlyStopping:
    """Convenience code to apply early stopping.
    The `is_best_performing_epoch` method can be useful to choose whether to
    save a checkpoint.
    """
    def __init__(self, keys_to_monitor, patience):
        """
        `keys_to_monitor` - names of measures that slowly increase, like
            validation accuracy.
        `patience` - how many epochs to look back
        """
        #  self.keys_to_monitor = keys_to_monitor
        self._best_perf = {k: BestPerf(-2, 0) for k in keys_to_monitor}
        self.patience = patience

    def is_best_performing_epoch(self, cur_epoch, perf_dct=None) -> bool:
        """
        Call this once per epoch.
        Record whether the current epoch is the best one. Return True or False.

        If perf_dct is given, update the knowledge of best performing epoch.
        """
        if perf_dct is not None:
            assert set(perf_dct) == set(self._best_perf), \
                'bug: perf_dct has wrong keys'

        if perf_dct is not None:
            for key, val in self._best_perf.items():
                cur_perf = BestPerf(perf_dct[key], cur_epoch)
                if val.value < cur_perf.value:
                    self._best_perf[key] = cur_perf
        for key in self._best_perf:
            if cur_epoch == self._best_perf[key].epoch:
                return True
        return False

    def should_stop(self, cur_epoch) -> bool:
        should_stop = all(
            (cur_epoch - perf.epoch) > self.patience
            for perf in self._best_perf.values())
        return should_stop
