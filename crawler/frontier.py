from collections import defaultdict, deque
import os
import shelve

from threading import Condition, Thread, RLock
from queue import Queue, Empty
import time
from urllib.parse import urlparse

from utils import get_logger, get_urlhash, normalize
from utils.utilities import get_robots_delay
from scraper import is_valid

class Frontier(object):
    def __init__(self, config, restart):
        self.logger = get_logger("FRONTIER")
        self.config = config
        self.to_be_downloaded = list()

        self._lock = RLock()
        self._cv = Condition(self._lock)

        # politeness per domain: at least 0.5s between requests to the same domain
        # Note: robots.txt delays will be checked per-URL and override this if higher
        self.domain_delay = max(0.5, float(getattr(self.config, "time_delay", 0.0)))

        # per-domain queues and next-allowed times
        self._queues = defaultdict(deque)          # domain -> deque[url]
        self._next_allowed = defaultdict(float)    # domain -> monotonic ts
        self._pending = 0                          # total items in all queues
        
        if not os.path.exists(self.config.save_file) and not restart:
            # Save file does not exist, but request to load save.
            self.logger.info(
                f"Did not find save file {self.config.save_file}, "
                f"starting from seed.")
        elif os.path.exists(self.config.save_file) and restart:
            # Save file does exists, but request to start from seed.
            self.logger.info(
                f"Found save file {self.config.save_file}, deleting it.")
            os.remove(self.config.save_file)

        # Load existing save file, or create one if it does not exist.
        self.save = shelve.open(self.config.save_file)       # shelve = dict, key = urlhash, value = (url, completed = bool)
        if restart:
            for url in self.config.seed_urls:
                self.add_url(url)
        else:
            # Set the frontier state with contents of save file.
            self._parse_save_file()
            if not self.save:
                for url in self.config.seed_urls:
                    self.add_url(url)

    @staticmethod
    def _domain_of(url: str) -> str:
        try:
            return (urlparse(url).hostname or "").lower()
        except Exception:
            return ""

    def _enqueue(self, url: str):
        url = normalize(url)
        dom = self._domain_of(url)
        self._queues[dom].append(url)
        self._pending += 1
        # ensure next_allowed has a key; default 0 allows immediate pick if time passed
        _ = self._next_allowed[dom]
        self._cv.notify()  # wake a waiting worker


    def _parse_save_file(self):
        ''' This function can be overridden for alternate saving techniques. '''
        total_count = len(self.save)
        tbd_count = 0
        with self._lock:
            for url, completed in self.save.values():
                if not completed and is_valid(url):
                    self._enqueue(url)
                    tbd_count += 1
        self.logger.info(
            f"Found {tbd_count} urls to be downloaded from {total_count} "
            f"total urls discovered.")

    def get_tbd_url(self):
        with self._lock:
            while True:
                # Check pending URLs
                if self._pending == 0:
                    return None
                # Get Time and compare with next allowed times
                now = time.monotonic()
                chosen_dom = None
                # scan for any domain eligible now
                for dom, q in list(self._queues.items()):
                    if not q:
                        continue
                    if self._next_allowed[dom] <= now:
                        chosen_dom = dom
                        break

                if chosen_dom is None:
                    # No domain is ready yet; wait until the soonest one is.
                    soonest_time = min(self._next_allowed[dom] for dom, q in self._queues.items() if q)
                    wait_time = max(0, soonest_time - now)
                    self._cv.wait(timeout=wait_time)
                    continue

                # Found a domain ready to go.
                url = self._queues[chosen_dom].popleft()
                self._pending -= 1
                
                # Get robots.txt delay for this URL, enforce minimum 500ms
                robots_delay = get_robots_delay(url, getattr(self.config, "user_agent", "*"))
                actual_delay = max(self.domain_delay, robots_delay)
                
                # Update next allowed time for this domain
                self._next_allowed[chosen_dom] = now + actual_delay
                return url
            

    def add_url(self, url):
        url = normalize(url)
        if not is_valid(url):
            return
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
            # This should not happen.
                self.logger.error(
                    f"Completed url {url}, but have not seen it before.")

            self.save[urlhash] = (url, True)
            self.save.sync()
