import logging
import os

def get_logger(name=None):
    name = os.path.basename(name)
    name = os.path.splitext(name)[0]
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(name)-18s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger
