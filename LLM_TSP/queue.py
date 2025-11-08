from queue import Empty
import multiprocessing as mp

class SemaphoreQueue:
    """Portable mp.Queue wrapper with an item-count semaphore."""
    def __init__(self, maxsize: int | None = None):
        size = 0 if maxsize in (None, 0) else maxsize
        self._queue = mp.Queue(size)
        self._sem = mp.Semaphore(0)

    def put(self, item) -> None:
        self._queue.put(item)
        self._sem.release()

    def get(self, block: bool = True, timeout: float | None = None):
        if not self._sem.acquire(block, timeout=timeout):
            raise Empty
        try:
            return self._queue.get_nowait()
        except Exception:
            self._sem.release()
            return None

    def get_nowait(self):
        return self.get(block=False)

    def try_get(self):
        if self._sem.acquire(False):
            try:
                return self._queue.get_nowait()
            except Exception:
                self._sem.release()
                return None
        return None

    def put_back_bulk(self, items) -> None:
        for item in items:
            self.put(item)

    def has_items(self) -> bool:
        if self._sem.acquire(False):
            self._sem.release()
            return True
        return False

    def empty(self) -> bool:
        return not self.has_items()

    def __getattr__(self, name):
        return getattr(self._queue, name)