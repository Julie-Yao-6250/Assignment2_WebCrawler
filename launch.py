from configparser import ConfigParser
from argparse import ArgumentParser

from utils.server_registration import get_cache_server
from utils.config import Config
from crawler import Crawler
from crawler.frontier_mt import PoliteFrontier
from crawler.worker_mt import PoliteWorker
from crawler.taskboard import TaskBoard
from crawler.worker_tb import TaskBoardWorker


def main(config_file, restart):
    cparser = ConfigParser()
    cparser.read(config_file)
    config = Config(cparser)
    config.cache_server = get_cache_server(config, restart)
    # Select multithread-capable frontier/worker when threads_count > 1
    if config.threads_count > 1:
        # Use TaskBoard push model by default for multithreading
        crawler = Crawler(config, restart, frontier_factory=TaskBoard, worker_factory=TaskBoardWorker)
    else:
        crawler = Crawler(config, restart)
    crawler.start()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--restart", action="store_true", default=False)
    parser.add_argument("--config_file", type=str, default="config.ini")
    args = parser.parse_args()
    main(args.config_file, args.restart)
