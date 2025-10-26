import os
import time
import shelve
import random
from collections import deque, defaultdict
from threading import RLock, Condition, Thread
from queue import Queue
from urllib.parse import urlparse

from utils import get_logger, get_urlhash, normalize
from scraper import is_valid


def _domain_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


class TaskBoard:
    """Single-threaded dispatcher with per-domain politeness.

    - One dispatcher thread assigns URLs to idle workers via per-worker inbox queues.
    - Guarantees >= 500ms between requests to the same domain (or max(0.5, POLITENESS)).
    - Uses shelve for persistence, same format as base Frontier.
    """

    def __init__(self, config, restart):
        self.logger = get_logger("TASKBOARD", "TASKBOARD")
        self.config = config
        self._lock = RLock()
        self._cv = Condition(self._lock)

        self._delay = max(0.5, float(getattr(self.config, "time_delay", 0.0)))

        # State
        self._queues = defaultdict(deque)          # domain -> deque[url]
        self._next_allowed = defaultdict(float)    # domain -> ts
        self._pending = 0

        # Workers
        self._worker_inbox: dict[int, Queue] = {}
        self._idle_workers: set[int] = set()
        self._worker_count = 0

        # Termination flag
        self._done = False

        # Shelve persistence
        if not os.path.exists(self.config.save_file) and not restart:
            self.logger.info(
                f"Did not find save file {self.config.save_file}, starting from seed.")
        elif os.path.exists(self.config.save_file) and restart:
            self.logger.info(
                f"Found save file {self.config.save_file}, deleting it.")
            os.remove(self.config.save_file)
        self.save = shelve.open(self.config.save_file)

        if restart:
            for url in self.config.seed_urls:
                self.add_url(url)
        else:
            self._parse_save_file()
            if not self.save:
                for url in self.config.seed_urls:
                    self.add_url(url)

        # Start dispatcher thread
        self._dispatcher = Thread(target=self._dispatch_loop, name="TaskDispatcher", daemon=True)
        self._dispatcher.start()

    # --------------------------- Persistence ---------------------------
    def _parse_save_file(self):
        total_count = len(self.save)
        tbd_count = 0
        with self._lock:
            for url, completed in self.save.values():
                if not completed and is_valid(url):
                    self._enqueue(url)
                    tbd_count += 1
        self.logger.info(
            f"Found {tbd_count} urls to be downloaded from {total_count} total urls discovered.")

    def add_url(self, url):
        url = normalize(url)
        urlhash = get_urlhash(url)
        with self._lock:
            if urlhash not in self.save:
                self.save[urlhash] = (url, False)
                self.save.sync()
                self._enqueue(url)

    def mark_url_complete(self, url):
        urlhash = get_urlhash(url)
        with self._lock:
            if urlhash not in self.save:
                self.logger.error(
                    f"Completed url {url}, but have not seen it before.")
            self.save[urlhash] = (url, True)
            self.save.sync()

    # ---------------------------- Workers -----------------------------
    def register_worker(self, worker_id: int) -> Queue:
        with self._lock:
            if worker_id not in self._worker_inbox:
                self._worker_inbox[worker_id] = Queue()
                self._worker_count += 1
        return self._worker_inbox[worker_id]

    def get_task(self, worker_id: int):
        """Worker blocks here to receive a task. Upon entry, worker is marked idle
        so dispatcher can immediately assign the next task. Returns None to signal termination.
        """
        inbox = self.register_worker(worker_id)
        with self._lock:
            # Mark idle and notify dispatcher
            self._idle_workers.add(worker_id)
            self._cv.notify()
        task = inbox.get()  # blocking; dispatcher put() url or None
        return task

    # ----------------------- Internal helpers ------------------------
    def _enqueue(self, url: str):
        dom = _domain_of(url)
        self._queues[dom].append(url)
        self._pending += 1
        _ = self._next_allowed[dom]
        self._cv.notify()

    def _eligible_domains(self, now: float):
        for dom, q in self._queues.items():
            if q and self._next_allowed[dom] <= now:
                yield dom

    def _try_dispatch(self) -> bool:
        """Try assign one task. Returns True if a dispatch occurred."""
        if not self._idle_workers or self._pending == 0:
            return False
        now = time.monotonic()
        eligible = [d for d in self._eligible_domains(now)]
        if not eligible:
            return False
        dom = random.choice(eligible)
        url = self._queues[dom].popleft()
        self._pending -= 1
        self._next_allowed[dom] = now + self._delay

        wid = random.choice(list(self._idle_workers))
        self._idle_workers.remove(wid)
        inbox = self._worker_inbox.get(wid)
        if inbox is not None:
            inbox.put(url)
        return True

    def _maybe_finish(self) -> bool:
        """If no pending work and all workers idle, declare done."""
        if self._pending == 0 and len(self._idle_workers) == self._worker_count and not self._done:
            self._done = True
            # Broadcast termination sentinel
            for wid, inbox in self._worker_inbox.items():
                try:
                    inbox.put(None)
                except Exception:
                    pass
            return True
        return False

    def _dispatch_loop(self):
        while True:
            with self._lock:
                if self._done:
                    break
                # Dispatch as much as possible right now
                progressed = False
                while self._try_dispatch():
                    progressed = True
                # Maybe finish
                if self._maybe_finish():
                    break
                # Compute next wait
                if not progressed:
                    waits = [self._next_allowed[d] for d, q in self._queues.items() if q]
                    now = time.monotonic()
                    timeout = None
                    if waits:
                        timeout = max(0.0, min(waits) - now)
                    # wait for new idle worker or new url or time slot
                    self._cv.wait(timeout=timeout)
