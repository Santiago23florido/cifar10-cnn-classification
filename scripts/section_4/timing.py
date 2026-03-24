# Section 4 timing callback used by notebook experiments.
from .runtime import Callback, time


class TimingHistory(Callback):
    def on_train_begin(self, logs=None):
        self.batch_times = []
        self.epoch_times = []
        self._batch_start = None
        self._epoch_start = None

    def on_train_batch_begin(self, batch, logs=None):
        self._batch_start = time.perf_counter()

    def on_train_batch_end(self, batch, logs=None):
        if self._batch_start is None:
            return
        self.batch_times.append(time.perf_counter() - self._batch_start)

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        if self._epoch_start is None:
            return
        self.epoch_times.append(time.perf_counter() - self._epoch_start)
