import logging
import sys
from pathlib import Path
import os
import datetime
import functools
import atexit

from threading import Thread
from torch.multiprocessing import Queue
from logging import Filter
from logging.handlers import QueueHandler, QueueListener
from utils import comm

# @functools.lru_cache()
# def get_logger(
#         log_dir='./log',
#         log_level=logging.DEBUG,
#         name='hae',
#         distributed_rank=0
# ):
#     """
#         initialize and get a logger
#         make log file for day
#         Args:
#              log_dir: directory name to manage log file
#              log_level: default info
#         Returns:
#             logging.Logger
#     """
#
#     Path(log_dir).mkdir(exist_ok=True, parents=True)
#
#     logger = logging.getLogger(name)
#     logger.setLevel(log_level)
#     logger.propagate = False
#
#     formatter = logging.Formatter(
#         '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
#         datefmt='%Y-%m-%d:%H:%M:%S')
#
#     today = datetime.datetime.now()
#     now_str = today.strftime("%Y%m%d")
#
#     stream_handler = logging.StreamHandler()
#     stream_handler.setFormatter(formatter)
#     logger.addHandler(stream_handler)
#
#     file_handler = logging.FileHandler(filename=os.path.join(log_dir, f"{now_str}.log"))
#     file_handler.setFormatter(formatter)
#     logger.addHandler(file_handler)
#
#     return logger

@functools.lru_cache()
def get_logger(
        log_dir='./log',
        log_level=logging.DEBUG,
        name='hae',
        distributed_rank=0
):
    Path(log_dir).mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    log_queue = Queue(-1)
    today = datetime.datetime.now()
    now_str = today.strftime("%Y%m%d")
    file_handler = logging.FileHandler(filename=os.path.join(log_dir, f"{now_str}.log"))
    file_handler.setLevel(log_level)

    listener = QueueListener(log_queue, file_handler, respect_handler_level=True)
    listener.start()
    return log_queue

class WorkerLogFilter(Filter):
    def __init__(self, rank=-1):
        super().__init__()
        self._rank = rank

    def filter(self, record):
        if self._rank != -1:
            record.msg = f"Rank {self._rank} | {record.msg}"
        return True

def setup_logger(rank: int, log_queue: Queue, cfg):
    queue_handler = QueueHandler(log_queue)

    formatter = logging.Formatter(
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S')

    worker_filter = WorkerLogFilter(rank)
    queue_handler.addFilter(worker_filter)
    queue_handler.setLevel(cfg.logs.level)
    queue_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(cfg.logs.level)
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger(cfg.logs.name)
    root_logger.addHandler(queue_handler)
    root_logger.addHandler(stream_handler)
    root_logger.setLevel(cfg.logs.level)



class Logger:
    def __init__(self, cfg):
        self.cfg = cfg
        self._logger = logging.getLogger(cfg.logs.name)
        self.setup()

    @property
    def logger(self):
        return self._logger

    def start(self):
        self.listener.start()

    def stop(self):
        self.listener.stop()

    def setup(self):
        Path(self.cfg.logs.root_dir).mkdir(exist_ok=True, parents=True)
        log_queue = Queue(-1)

        self.listener = QueueListener(log_queue, respect_handler_level=True)

        queue_handler = QueueHandler(log_queue)

        worker_filter = WorkerLogFilter(comm.get_local_rank())

        queue_handler.addFilter(worker_filter)
        queue_handler.setLevel(self.cfg.logs.level)
        queue_handler.setFormatter(self.formatter)

        today = datetime.datetime.now()
        now_str = today.strftime("%Y%m%d")
        file_handler = logging.FileHandler(filename=os.path.join(self.cfg.logs.root_dir, f"{now_str}.log"))
        file_handler.setLevel(self.cfg.logs.level)
        file_handler.setFormatter(self.formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(self.cfg.logs.level)
        stream_handler.setFormatter(self.formatter)

        self._logger.addHandler(queue_handler)
        self._logger.addHandler(stream_handler)
        self._logger.addHandler(file_handler)
        self._logger.setLevel(self.cfg.logs.level)

    @property
    def formatter(self):
        return logging.Formatter(
            '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S'
        )
