from threading import Thread
from inspect import getsource
from utils.download import download
from utils import get_logger
import scraper


class TaskBoardWorker(Thread):
    """Worker for TaskBoard push model.

    The worker calls taskboard.get_task(worker_id) which blocks until a URL is assigned.
    Upon completion, it loops and calls get_task again (which marks the worker idle),
    so dispatcher can立即分配下一个任务。
    """

    def __init__(self, worker_id, config, taskboard):
        self.logger = get_logger(f"Worker-{worker_id}", "Worker_TB")
        self.config = config
        self.taskboard = taskboard
        self.worker_id = worker_id
        # basic check for requests in scraper
        assert {getsource(scraper).find(req) for req in {"from requests import", "import requests"}} == {-1}, "Do not use requests in scraper.py"
        assert {getsource(scraper).find(req) for req in {"from urllib.request import", "import urllib.request"}} == {-1}, "Do not use urllib.request in scraper.py"
        super().__init__(daemon=True)

    def run(self):
        while True:
            tbd_url = self.taskboard.get_task(self.worker_id)
            if not tbd_url:
                self.logger.info("TaskBoard signaled completion. Stopping Worker.")
                break
            resp = download(tbd_url, self.config, self.logger)
            self.logger.info(
                f"Downloaded {tbd_url}, status <{resp.status}>, "
                f"using cache {self.config.cache_server}.")
            scraped_urls = scraper.scraper(tbd_url, resp)
            for scraped_url in scraped_urls:
                self.taskboard.add_url(scraped_url)
            self.taskboard.mark_url_complete(tbd_url)
